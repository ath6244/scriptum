"""RQ4 descriptive features: how do the families' outputs differ, measured directly?

Per family (all 9,665 rows, LLM_output_masked): median word count, type-token ratio on the first
200 words, mean sentence length, markdown per 1k words (headers, bullets, bold, code blocks),
emoji per 1k words, refusal rate. Plus the 15 most distinctive word features per family from the
TF-IDF + LR output-only model (refit on train with its saved best C).

Usage (from the project root):
    python scripts/rq4_features.py

Outputs:
    results/rq4_features.csv          one row per family
    results/rq4_top_words.csv         top 15 word/bigram features per family with their coefficients
    figures/rq4_features.png          one small panel per feature
"""

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of scripts/)
sys.path.insert(0, str(ROOT))  # so "from src... / from scripts... import" works when run as a script
from scripts.baselines import tfidf_vectorizers  # noqa: E402
from src.data import FAMILIES, load_data, load_splits  # noqa: E402
from src.transforms import style_features  # noqa: E402

SEED = 448
TOP_K = 15
FAMILY_COLORS = dict(zip(FAMILIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))  # fixed order
PANELS = [  # (column in rq4_features.csv, panel title, number format)
    ("median_words", "Median words", "{:.0f}"),
    ("ttr_200", "Type-token ratio (first 200 words)", "{:.3f}"),
    ("mean_sentence_len", "Mean sentence length (words)", "{:.1f}"),
    ("headers_per_1k", "Headers per 1k words", "{:.1f}"),
    ("bullets_per_1k", "Bullets / numbered items per 1k words", "{:.1f}"),
    ("bold_per_1k", "Bold spans per 1k words", "{:.1f}"),
    ("code_blocks_per_1k", "Code blocks per 1k words", "{:.2f}"),
    ("emoji_per_1k", "Emoji per 1k words", "{:.2f}"),
    ("refusal_pct", "Refusal rate (% of outputs)", "{:.1f}%"),
]


def family_features(df):
    features = pd.DataFrame([style_features(text) for text in df["LLM_output_masked"]])
    features["LLM_name"] = df["LLM_name"].to_numpy()
    rows = []
    for family in FAMILIES:
        f = features[features["LLM_name"] == family]
        total_words = f["words"].sum()
        rows.append({
            "LLM_name": family,
            "n_outputs": len(f),
            "median_words": f["words"].median(),
            "ttr_200": f["ttr_200"].mean(),
            "ttr_200_n": f["ttr_200"].notna().sum(),  # only outputs with >= 200 words have a TTR
            "mean_sentence_len": f["mean_sentence_len"].mean(),
            # Pooled rates: all markdown items of the family / all its words, x 1000.
            "headers_per_1k": f["headers"].sum() / total_words * 1000,
            "bullets_per_1k": f["bullets"].sum() / total_words * 1000,
            "bold_per_1k": f["bold"].sum() / total_words * 1000,
            "code_blocks_per_1k": f["code_blocks"].sum() / total_words * 1000,
            "emoji_per_1k": f["emoji"].sum() / total_words * 1000,
            "share_with_emoji": (f["emoji"] > 0).mean(),
            "refusal_rate": f["refusal"].mean(),
            "refusal_pct": f["refusal"].mean() * 100,
        })
    return pd.DataFrame(rows)


def top_words(data):
    """Refit TF-IDF + LR (output only, best C from baselines_runs.csv) and read the word-feature weights."""
    runs = pd.read_csv(ROOT / "results" / "baselines_runs.csv")
    best_c = runs.query("baseline == 'tfidf_lr' and condition == 'output_only'")["best_C"].iloc[0]
    random.seed(SEED)
    np.random.seed(SEED)
    texts = data["train"]["LLM_output_masked"]
    word, char = (vectorizer.fit(texts) for vectorizer in tfidf_vectorizers())
    from scipy.sparse import hstack
    x_train = hstack([word.transform(texts), char.transform(texts)]).tocsr()
    model = LogisticRegression(C=best_c, max_iter=5000, random_state=0).fit(x_train, data["train"]["LLM_name"])
    n_word = len(word.vocabulary_)
    names = word.get_feature_names_out()
    rows = []
    for k, family in enumerate(model.classes_):
        # Multinomial LR has one weight vector per class; a large positive weight on a word means
        # "this word makes the model more confident the text is from this family".
        weights = model.coef_[k, :n_word]  # the word block comes first, then the character block
        for rank, i in enumerate(np.argsort(weights)[::-1][:TOP_K], start=1):
            rows.append({"LLM_name": family, "rank": rank, "feature": names[i], "weight": weights[i]})
    return pd.DataFrame(rows), best_c


def plot_features(features, path):
    n_cols = 3
    n_rows = int(np.ceil(len(PANELS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.2 * n_rows))
    for ax, (column, title, fmt) in zip(axes.flat, PANELS):
        values = features[column].to_numpy()
        bars = ax.bar(features["LLM_name"], values, width=0.6,
                      color=[FAMILY_COLORS[f] for f in features["LLM_name"]])
        for bar, value in zip(bars, values):  # value labels: identity never relies on color alone
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(value),
                    ha="center", va="bottom", fontsize=8, color="#333333")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(values.max() * 1.18, 1e-9))
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=8, colors="#666666")
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(PANELS):]:
        ax.axis("off")
    fig.suptitle("RQ4: style features by model family (all 9,665 masked outputs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = load_data()
    features = family_features(df)
    features.round(4).to_csv(ROOT / "results" / "rq4_features.csv", index=False)
    plot_features(features, ROOT / "figures" / "rq4_features.png")
    print("Per-family features (saved to results/rq4_features.csv, figures/rq4_features.png):")
    print(features.round(3).T.to_string(header=False))

    words, best_c = top_words(load_splits(df))
    words.round(4).to_csv(ROOT / "results" / "rq4_top_words.csv", index=False)
    print(f"\nTop {TOP_K} word features per family (TF-IDF + LR, C={best_c}; saved to results/rq4_top_words.csv):")
    for family, rows in words.groupby("LLM_name"):
        print(f"  {family:7s} " + ", ".join(rows["feature"]))


if __name__ == "__main__":
    main()

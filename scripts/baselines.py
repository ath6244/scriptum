"""Baselines: majority class, length-only logistic regression, TF-IDF logistic regression.

Each baseline is run on three input conditions (output only, input only, input + output).
Outputs are the masked column (LLM_output_masked). Train on the train split, pick the
regularization strength C on val (by macro-F1), then report on test. Every run uses 3 seeds.
5 balanced classes, so chance = 20% accuracy.

Requires dataset.csv and splits.json (run scripts/prepare_arena.py, then scripts/make_splits.py).

Usage (from the project root):
    python scripts/baselines.py

Outputs:
    results/baselines_runs.csv                 one row per (baseline, condition, seed)
    results/baselines.csv                      mean and std over seeds
    results/confusion_tfidf_output.csv         confusion matrix, TF-IDF output-only, first seed
    figures/confusion_tfidf_output.png         same matrix as a figure
"""

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of scripts/)
sys.path.insert(0, str(ROOT))  # so "from src.data import ..." works when run as a script
from src.data import FAMILIES, load_data, load_splits  # noqa: E402

SEEDS = [0, 1, 2]
CHANCE = 1 / len(FAMILIES)  # 0.20 with 5 balanced classes
LABEL = "LLM_name"
CONDITIONS = ["output_only", "input_only", "input_output"]
BASELINES = ["majority", "length_lr", "tfidf_lr"]
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]  # values of C tried on val


def set_seeds(seed):
    # torch is not used in this script, so only python and numpy need seeding.
    random.seed(seed)
    np.random.seed(seed)


def get_text(df, condition):
    """Return the text the model is allowed to see under each input condition."""
    if condition == "output_only":
        return df["LLM_output_masked"]
    if condition == "input_only":
        return df["LLM_Input"]
    if condition == "input_output":
        return df["LLM_Input"] + " [SEP] " + df["LLM_output_masked"]
    raise ValueError(condition)


def length_features(texts):
    """4 numbers per text: characters, words, lines, newline count."""
    texts = pd.Series(texts)
    chars = texts.str.len()
    words = texts.str.split().str.len()
    newlines = texts.str.count("\n")
    lines = newlines + 1
    features = np.column_stack([chars, words, lines, newlines]).astype(float)
    # Lengths are heavily right-skewed; log(1 + x) makes them easier for a linear model.
    return np.log1p(features)


def make_features(baseline, x_train, x_val, x_test):
    """Fit the feature extractor on train only, then transform all three splits.

    The features don't depend on C or the seed, so they are built once per (baseline, condition).
    """
    if baseline == "majority":
        return x_train, x_val, x_test  # DummyClassifier ignores the inputs
    if baseline == "length_lr":
        scaler = StandardScaler().fit(length_features(x_train))
        return [scaler.transform(length_features(x)) for x in (x_train, x_val, x_test)]
    if baseline == "tfidf_lr":
        word, char = (vectorizer.fit(x_train) for vectorizer in tfidf_vectorizers())
        return [hstack([word.transform(x), char.transform(x)]).tocsr() for x in (x_train, x_val, x_test)]
    raise ValueError(baseline)


def tfidf_vectorizers():
    """Unfitted word 1-2-gram and character 3-5-gram TF-IDF vectorizers (the word block comes first)."""
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=100_000)
    return word, char


def make_model(baseline, C, seed):
    if baseline == "majority":
        return DummyClassifier(strategy="most_frequent")
    return LogisticRegression(C=C, max_iter=5000, random_state=seed)


def scores(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=FAMILIES, zero_division=0),
    }


def run_one(baseline, condition, seed, features, labels):
    """Tune C on val (if the baseline has one), then score the chosen model on test."""
    set_seeds(seed)
    f_train, f_val, f_test = features
    y_train, y_val, y_test = labels

    c_values = [None] if baseline == "majority" else C_GRID
    best = None
    for C in c_values:
        model = make_model(baseline, C, seed).fit(f_train, y_train)
        val_scores = scores(y_val, model.predict(f_val))
        if best is None or val_scores["macro_f1"] > best["val_scores"]["macro_f1"]:
            best = {"C": C, "model": model, "val_scores": val_scores}

    test_pred = best["model"].predict(f_test)
    test_scores = scores(y_test, test_pred)
    row = {
        "baseline": baseline, "condition": condition, "seed": seed, "best_C": best["C"],
        "val_acc": best["val_scores"]["acc"], "val_macro_f1": best["val_scores"]["macro_f1"],
        "test_acc": test_scores["acc"], "test_macro_f1": test_scores["macro_f1"],
    }
    return row, test_pred


def save_confusion(y_true, y_pred, csv_path, png_path):
    matrix = confusion_matrix(y_true, y_pred, labels=FAMILIES)
    pd.DataFrame(matrix, index=[f"true_{f}" for f in FAMILIES],
                 columns=[f"pred_{f}" for f in FAMILIES]).to_csv(csv_path)

    # Row-normalize so each cell is "% of this true family predicted as X".
    row_pct = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(FAMILIES)):
        for j in range(len(FAMILIES)):
            color = "white" if row_pct[i, j] > 0.6 else "#222222"
            ax.text(j, i, f"{row_pct[i, j]:.0%}\n({matrix[i, j]})",
                    ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(len(FAMILIES)), FAMILIES)
    ax.set_yticks(range(len(FAMILIES)), FAMILIES)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.set_title("TF-IDF + LR, output only (test set)\nrow % (count)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Share of true family")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    df = load_data()
    data = load_splits(df)  # rebuilt from splits.json, which is the source of truth
    assert sum(len(part) for part in data.values()) == len(df), "some rows are in no split"
    for name, part in data.items():
        print(f"{name}: {len(part)} rows, {part['prompt_id'].nunique()} prompts")
    labels = [data[name][LABEL] for name in ["train", "val", "test"]]

    rows = []
    for baseline in BASELINES:
        for condition in CONDITIONS:
            texts = [get_text(data[name], condition) for name in ["train", "val", "test"]]
            features = make_features(baseline, *texts)
            for seed in SEEDS:
                row, test_pred = run_one(baseline, condition, seed, features, labels)
                rows.append(row)
                print(f"{baseline:10s} {condition:13s} seed={seed}  C={row['best_C']}  "
                      f"test acc={row['test_acc']:.3f}  macro-F1={row['test_macro_f1']:.3f}")
                if baseline == "tfidf_lr" and condition == "output_only" and seed == SEEDS[0]:
                    save_confusion(labels[2], test_pred,
                                   ROOT / "results" / "confusion_tfidf_output.csv",
                                   ROOT / "figures" / "confusion_tfidf_output.png")

    runs = pd.DataFrame(rows)
    runs.to_csv(ROOT / "results" / "baselines_runs.csv", index=False)

    summary = runs.groupby(["baseline", "condition"], sort=False).agg(
        test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
        test_macro_f1_mean=("test_macro_f1", "mean"), test_macro_f1_std=("test_macro_f1", "std"),
        val_macro_f1_mean=("val_macro_f1", "mean"),
    ).reset_index()
    summary["chance_acc"] = CHANCE
    summary = summary.round(4)
    summary.to_csv(ROOT / "results" / "baselines.csv", index=False)
    print(f"\nSummary (chance = {CHANCE:.0%}; saved to results/baselines.csv):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

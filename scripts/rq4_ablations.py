"""RQ4 ablations: retrain on transformed outputs to see which signals the classifiers rely on.

Output only. TF-IDF + LR and TextCNN, same hyperparameters as before. Variants of LLM_output_masked
(train, val and test all transformed the same way):
    no_markdown                  headers, bold/italics, list markers, code fences, tables, rules removed
    no_opener                    first sentence dropped
    length_matched               every output cut to the dataset's 25th-percentile word count
    no_markdown_length_matched   both of the above
    unmasked                     the original LLM_output (model names NOT masked)
TF-IDF + LR only:
    no_markdown_no_emoji_punct   no_markdown + emoji removed + punctuation and whitespace normalized
    word_only                    masked output, but TF-IDF uses word 1-2-grams only (no character n-grams)
Length-only LR:
    length_matched               how much length signal is left after truncation
The "masked" reference is the main output-only result (results/preds/ from scripts/stats_rq1_rq2.py).

TextCNN runs seeds 448, 449, 450. TF-IDF + LR and length-only LR are deterministic on fixed splits
(the seed changes nothing), so they run once. Runs already saved in results/rq4_runs.csv are skipped.

Usage (from the project root, after scripts/stats_rq1_rq2.py):
    python scripts/rq4_ablations.py --models tfidf_lr length_lr     # CPU
    python scripts/rq4_ablations.py --models textcnn      # GPU

Outputs:
    results/rq4_runs.csv                       one row per run
    results/preds/rq4_preds_<model>.csv        test predictions per run
    results/rq4_ablations.csv                  acc / macro-F1 with bootstrap CIs, drop vs masked with CIs
    figures/rq4_ablation_drops.png             drop vs masked per variant, both models
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of scripts/)
sys.path.insert(0, str(ROOT))  # so "from src... / from scripts... import" works when run as a script
from scripts.baselines import make_features, run_one, tfidf_vectorizers  # noqa: E402
from scripts.rq3_cross_domain import predict_neural, predict_tfidf  # noqa: E402
from src.data import FAMILIES, load_data, load_splits  # noqa: E402
from src.stats import acc_and_macro_f1, bootstrap, ci, confusion  # noqa: E402
from src.transforms import (drop_first_sentence, strip_emoji_and_normalize_punct,  # noqa: E402
                            strip_markdown, truncate_words)

MODELS = ["tfidf_lr", "textcnn", "length_lr"]
MAIN_VARIANTS = ["no_markdown", "no_opener", "length_matched", "no_markdown_length_matched", "unmasked"]
VARIANTS_BY_MODEL = {
    "tfidf_lr": MAIN_VARIANTS + ["no_markdown_no_emoji_punct", "word_only"],
    "textcnn": MAIN_VARIANTS,
    "length_lr": ["length_matched"],
}
VARIANTS = MAIN_VARIANTS + ["no_markdown_no_emoji_punct", "word_only"]  # display order
SEEDS = {"tfidf_lr": [448], "textcnn": [448, 449, 450], "length_lr": [448]}
NAMES = {"tfidf_lr": "TF-IDF + LR", "textcnn": "TextCNN", "length_lr": "Length-only LR"}
COLORS = {"tfidf_lr": "#2a78d6", "textcnn": "#eb6834", "length_lr": "#1baf7a"}  # categorical slots 1-3
RUNS_FILE = ROOT / "results" / "rq4_runs.csv"
PRED_DIR = ROOT / "results" / "preds"


def transform(df, variant, n_words):
    """Return a copy of df whose LLM_output_masked column holds the transformed text."""
    df = df.copy()
    masked = df["LLM_output_masked"]
    if variant == "no_markdown":
        df["LLM_output_masked"] = masked.map(strip_markdown)
    elif variant == "no_opener":
        df["LLM_output_masked"] = masked.map(drop_first_sentence)
    elif variant == "length_matched":
        df["LLM_output_masked"] = masked.map(lambda t: truncate_words(t, n_words))
    elif variant == "no_markdown_length_matched":
        df["LLM_output_masked"] = masked.map(lambda t: truncate_words(strip_markdown(t), n_words))
    elif variant == "no_markdown_no_emoji_punct":
        df["LLM_output_masked"] = masked.map(lambda t: strip_emoji_and_normalize_punct(strip_markdown(t)))
    elif variant == "unmasked":
        df["LLM_output_masked"] = df["LLM_output"]  # the models read this column, so it now holds raw text
    elif variant == "word_only":
        pass  # text unchanged; only the TF-IDF features change (see predict_word_only)
    else:
        raise ValueError(variant)
    return df


def predict_word_only(train, val, test, seed):
    """TF-IDF + LR with the word 1-2-gram block only, C tuned on val as usual."""
    word, _ = tfidf_vectorizers()
    word.fit(train["LLM_output_masked"])
    features = [word.transform(part["LLM_output_masked"]).tocsr() for part in (train, val, test)]
    labels = [part["LLM_name"] for part in (train, val, test)]
    _, test_pred = run_one("tfidf_lr", "output_only", seed, features, labels)
    return np.asarray(test_pred)


def predict_length(train, val, test, seed):
    """Length-only LR (4 log-length features), C tuned on val, as in scripts/baselines.py."""
    features = make_features("length_lr", *[part["LLM_output_masked"] for part in (train, val, test)])
    labels = [part["LLM_name"] for part in (train, val, test)]
    _, test_pred = run_one("length_lr", "output_only", seed, features, labels)
    return np.asarray(test_pred)


def append_rows(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def load_predictions(test):
    """{(model, variant): [integer prediction arrays, one per seed]}, including the masked reference."""
    label_id = {f: i for i, f in enumerate(FAMILIES)}
    sources = [pd.read_csv(PRED_DIR / "baseline_preds.csv")
               .query("method in ['tfidf_lr', 'length_lr'] and condition == 'output_only'")
               .rename(columns={"method": "model"}).assign(variant="masked"),
               pd.read_csv(PRED_DIR / "neural_preds.csv").query("method == 'textcnn' and condition == 'output_only'")
               .rename(columns={"method": "model"}).assign(variant="masked")]
    sources += [pd.read_csv(PRED_DIR / f"rq4_preds_{m}.csv") for m in MODELS if (PRED_DIR / f"rq4_preds_{m}.csv").exists()]
    preds = pd.concat(sources, ignore_index=True)
    out = {}
    for (model, variant), rows in preds.groupby(["model", "variant"]):
        out[(model, variant)] = []
        for _, run in rows.groupby("seed"):
            run = run.sort_values("test_row")
            assert (run["y_true"].to_numpy() == test["LLM_name"].to_numpy()).all(), "prediction rows out of order"
            out[(model, variant)].append(run["y_pred"].map(label_id).to_numpy())
    return out


def summarize(test):
    """Accuracy / macro-F1 with grouped bootstrap CIs, and the paired drop vs the masked reference."""
    y_true = test["LLM_name"].map({f: i for i, f in enumerate(FAMILIES)}).to_numpy()
    groups = test["prompt_id"]
    preds = load_predictions(test)
    rows = []
    for model in MODELS:
        if (model, "masked") not in preds:
            continue
        # Same BOOT_SEED for every call -> the same resamples, so subtracting gives a paired bootstrap.
        reference = bootstrap(y_true, preds[(model, "masked")], groups, len(FAMILIES))
        reference_point = np.mean([acc_and_macro_f1(confusion(y_true, p, len(FAMILIES))) for p in preds[(model, "masked")]], axis=0)
        for variant in ["masked"] + VARIANTS:
            if (model, variant) not in preds:
                continue
            seed_scores = np.array([acc_and_macro_f1(confusion(y_true, p, len(FAMILIES))) for p in preds[(model, variant)]])
            point = seed_scores.mean(axis=0)
            samples = bootstrap(y_true, preds[(model, variant)], groups, len(FAMILIES))
            drop = reference - samples  # positive = the variant is worse than masked
            row = {"model": NAMES[model], "variant": variant, "n_seeds": len(seed_scores),
                   "acc": point[0], "acc_seed_std": seed_scores[:, 0].std(ddof=1) if len(seed_scores) > 1 else 0.0}
            row["acc_ci_low"], row["acc_ci_high"] = ci(samples[:, 0])
            row["macro_f1"] = point[1]
            row["macro_f1_ci_low"], row["macro_f1_ci_high"] = ci(samples[:, 1])
            row["drop_acc"] = reference_point[0] - point[0]
            row["drop_acc_ci_low"], row["drop_acc_ci_high"] = ci(drop[:, 0])
            row["drop_macro_f1"] = reference_point[1] - point[1]
            row["drop_macro_f1_ci_low"], row["drop_macro_f1_ci_high"] = ci(drop[:, 1])
            # The drop is clear if its 95% CI does not include 0.
            row["drop_ci_excludes_0"] = (variant != "masked") and (row["drop_acc_ci_low"] > 0 or row["drop_acc_ci_high"] < 0)
            rows.append(row)
    table = pd.DataFrame(rows).round(4)
    table.to_csv(ROOT / "results" / "rq4_ablations.csv", index=False)
    plot_drops(table, ROOT / "figures" / "rq4_ablation_drops.png")
    return table


def plot_drops(table, path):
    """Dot + 95% CI per variant; one row per variant, the two models side by side."""
    table = table[table["variant"] != "masked"]
    variants = [v for v in VARIANTS if v in set(table["variant"])]
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(variants) + 1.6))
    for k, model in enumerate(MODELS):
        rows = table[table["model"] == NAMES[model]].set_index("variant").reindex(variants)
        y = np.arange(len(variants)) + (k - 1) * 0.25  # 3 models side by side
        x = rows["drop_acc"] * 100
        err = np.vstack([x - rows["drop_acc_ci_low"] * 100, rows["drop_acc_ci_high"] * 100 - x])
        ax.errorbar(x, y, xerr=err, fmt="o", markersize=8, color=COLORS[model], ecolor=COLORS[model],
                    elinewidth=2, capsize=0, label=NAMES[model])
    ax.axvline(0, color="#888888", linewidth=1)
    ax.set_yticks(range(len(variants)), variants)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy drop vs masked output (points; < 0 = better)")
    ax.set_title("RQ4 ablations: accuracy removed by each transformation\n"
                 "(dot = mean, bar = grouped bootstrap 95% CI)", fontsize=11)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper right")  # the top-right corner has no data
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_data()
    data = load_splits(df)
    # 25th percentile of output length (words) over the whole dataset.
    n_words = int(df["LLM_output_masked"].str.split().str.len().quantile(0.25))
    print(f"length_matched cuts every output to {n_words} words")

    done = set()
    if RUNS_FILE.exists():
        saved = pd.read_csv(RUNS_FILE)
        done = set(zip(saved["model"], saved["variant"], saved["seed"]))

    jobs = [] if args.summarize_only else [(m, v) for m in args.models for v in VARIANTS_BY_MODEL[m]]
    for model_name, variant in jobs:
        parts = {name: transform(part, variant, n_words) for name, part in data.items()}
        if variant == "no_opener":
            empty = (parts["train"]["LLM_output_masked"].str.strip() == "").mean()
            print(f"  no_opener leaves {empty:.1%} of train outputs empty (single-sentence outputs)")
        for seed in SEEDS[model_name]:
            if (model_name, variant, seed) in done:
                print(f"skip {model_name} / {variant} / seed {seed}")
                continue
            print(f"\n=== {model_name} / {variant} / seed {seed} ===")
            start = time.time()
            if model_name == "tfidf_lr" and variant == "word_only":
                y_pred = predict_word_only(parts["train"], parts["val"], parts["test"], seed)
            elif model_name == "tfidf_lr":
                y_pred = predict_tfidf(parts["train"], parts["val"], parts["test"], seed)
            elif model_name == "length_lr":
                y_pred = predict_length(parts["train"], parts["val"], parts["test"], seed)
            else:
                y_pred = predict_neural(model_name, parts["train"], parts["val"], parts["test"], seed, device)
            y_true = parts["test"]["LLM_name"].to_numpy()
            label_id = {f: i for i, f in enumerate(FAMILIES)}
            acc, f1 = acc_and_macro_f1(confusion(np.array([label_id[f] for f in y_true]),
                                                 np.array([label_id[f] for f in y_pred]), len(FAMILIES)))
            append_rows(PRED_DIR / f"rq4_preds_{model_name}.csv",
                        [{"model": model_name, "variant": variant, "seed": seed, "test_row": i, "y_true": t, "y_pred": p}
                         for i, (t, p) in enumerate(zip(y_true, y_pred))])
            append_rows(RUNS_FILE, [{"model": model_name, "variant": variant, "seed": seed, "acc": acc,
                                     "macro_f1": f1, "train_time_s": round(time.time() - start, 1)}])
            print(f"  acc {acc:.3f}  macro-F1 {f1:.3f}  ({time.time() - start:.0f}s)")

    table = summarize(data["test"])
    print("\nSaved results/rq4_ablations.csv and figures/rq4_ablation_drops.png")
    print(table[["model", "variant", "n_seeds", "acc", "acc_ci_low", "acc_ci_high", "drop_acc",
                 "drop_acc_ci_low", "drop_acc_ci_high", "drop_ci_excludes_0"]].to_string(index=False))


if __name__ == "__main__":
    main()

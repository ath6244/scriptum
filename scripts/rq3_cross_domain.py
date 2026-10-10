"""RQ3: do LLM fingerprints generalize across domains? Output only (LLM_output_masked).

For each domain D (coding, math, creative_writing, general):
  lodo       train on the other 3 domains, test on D           (D never seen in training)
  in_domain  train on all 4 domains,       test on the same D  (D seen in training)
Test rows are always the rows of D in the original test split; train/val rows come from the
original train/val splits, so prompt grouping is unchanged.

Size control: every training set is subsampled to N rows per family (N = smallest per-family
train count over the 4 LODO folds; the subsample depends on the seed), so the only difference
between in_domain and lodo is whether D was in training, not how much data there was.

Models and hyperparameters are the same as scripts/baselines.py (TF-IDF + LR) and
scripts/run_neural.py (TextCNN, BiLSTM). Seeds 448, 449, 450.

Runs already saved in results/rq3.csv are skipped, so the script can be restarted after an interruption.

Usage (from the project root):
    python scripts/rq3_cross_domain.py
    python scripts/rq3_cross_domain.py --models tfidf_lr      # only one model (e.g. CPU while the GPU is busy)

Outputs:
    results/rq3.csv                    one row per run
    results/rq3_per_family_runs.csv    per-family recall, one row per run x family
    results/rq3_summary.csv            mean ± std per model x domain x condition, drop = in_domain - lodo
    results/rq3_per_family.csv         per-family recall per condition, n_test, "low n" flag
    figures/rq3_drop_heatmap.png       model x held-out domain, cell = mean accuracy drop
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of scripts/)
sys.path.insert(0, str(ROOT))  # so "from src... / from scripts... import" works when run as a script
from scripts.baselines import make_features, run_one  # noqa: E402
from src.data import FAMILIES, build_vocab, encode, load_data, load_splits  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import make_loader, predict, set_seeds, train_model  # noqa: E402

MODELS = ["tfidf_lr", "textcnn", "bilstm"]
DOMAINS = ["coding", "math", "creative_writing", "general"]
CONDITIONS = ["in_domain", "lodo"]
SEEDS = [448, 449, 450]
LOW_N = 30  # per-family test cells smaller than this are marked "low n"
MODEL_NAMES = {"tfidf_lr": "TF-IDF + LR", "textcnn": "TextCNN", "bilstm": "BiLSTM"}
RUNS_FILE = ROOT / "results" / "rq3.csv"
FAMILY_FILE = ROOT / "results" / "rq3_per_family_runs.csv"


def subsample(rows, n_per_family, seed):
    """Keep n_per_family random rows of each family."""
    rng = np.random.default_rng(seed)
    parts = [group.iloc[rng.permutation(len(group))[:n_per_family]] for _, group in rows.groupby("LLM_name")]
    return pd.concat(parts).reset_index(drop=True)


def predict_tfidf(train, val, test, seed):
    """TF-IDF + LR exactly as in scripts/baselines.py: C tuned on val by macro-F1."""
    texts = [part["LLM_output_masked"] for part in (train, val, test)]
    features = make_features("tfidf_lr", *texts)
    labels = [part["LLM_name"] for part in (train, val, test)]
    _, test_pred = run_one("tfidf_lr", "output_only", seed, features, labels)
    return np.asarray(test_pred)


def predict_neural(model_name, train, val, test, seed, device):
    """TextCNN / BiLSTM exactly as in scripts/run_neural.py, with the vocab built from this training set."""
    vocab = build_vocab(pd.concat([train["LLM_Input"], train["LLM_output_masked"]]))
    label_id = {family: i for i, family in enumerate(FAMILIES)}
    loaders = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        sequences = encode(part, "output_only", vocab)
        labels = [label_id[f] for f in part["LLM_name"]]
        loaders[name] = make_loader(sequences, labels, shuffle=(name == "train"), seed=seed)
    set_seeds(seed)  # before building the model, so the initial weights depend on the seed
    model = build_model(model_name, len(vocab), len(FAMILIES)).to(device)
    train_model(model, loaders["train"], loaders["val"], device)  # restores the best epoch
    y_pred, _ = predict(model, loaders["test"], device)
    return np.array([FAMILIES[i] for i in y_pred])


def append_rows(path, rows):
    """Append rows to a CSV (header only when the file is new). Safe to call after every run."""
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def summarize():
    """Build the summary tables and the heatmap from everything saved so far."""
    runs = pd.read_csv(RUNS_FILE)
    keys = ["model", "held_out_domain"]

    # Drop per seed (paired: same seed's in_domain minus lodo), then mean and std over seeds.
    wide = runs.pivot_table(index=keys + ["seed"], columns="condition", values=["acc", "macro_f1"]).dropna()
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide["drop_acc"] = wide["acc_in_domain"] - wide["acc_lodo"]
    wide["drop_macro_f1"] = wide["macro_f1_in_domain"] - wide["macro_f1_lodo"]
    summary = wide.groupby(keys).agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary["n_seeds"] = wide.groupby(keys).size()
    summary["n_test"] = runs.groupby(keys)["n_test"].first()
    # A drop is only meaningful if it is at least as big as its std over seeds.
    summary["drop_acc_meaningful"] = summary["drop_acc_mean"].abs() >= summary["drop_acc_std"]
    summary["drop_macro_f1_meaningful"] = summary["drop_macro_f1_mean"].abs() >= summary["drop_macro_f1_std"]
    summary = summary.reset_index()
    summary.round(4).to_csv(ROOT / "results" / "rq3_summary.csv", index=False)

    # Per-family recall in each condition.
    fam = pd.read_csv(FAMILY_FILE)
    per_family = fam.pivot_table(index=keys + ["family"], columns="condition", values="recall",
                                 aggfunc=["mean", "std"])
    per_family.columns = [f"recall_{condition}_{stat}" for stat, condition in per_family.columns]
    per_family["n_test"] = fam.groupby(keys + ["family"])["n_test"].first()
    per_family["low_n"] = np.where(per_family["n_test"] < LOW_N, "low n", "")
    if {"recall_in_domain_mean", "recall_lodo_mean"} <= set(per_family.columns):
        per_family["drop"] = per_family["recall_in_domain_mean"] - per_family["recall_lodo_mean"]
    per_family.reset_index().round(4).to_csv(ROOT / "results" / "rq3_per_family.csv", index=False)

    plot_drop_heatmap(summary, ROOT / "figures" / "rq3_drop_heatmap.png")
    return summary, per_family


def plot_drop_heatmap(summary, path):
    models = [m for m in MODELS if m in set(summary["model"])]
    drop = summary.pivot(index="model", columns="held_out_domain", values="drop_acc_mean").reindex(index=models, columns=DOMAINS)
    std = summary.pivot(index="model", columns="held_out_domain", values="drop_acc_std").reindex(index=models, columns=DOMAINS)
    n_test = summary.groupby("held_out_domain")["n_test"].first().reindex(DOMAINS)

    # Diverging: red = accuracy dropped when the domain was unseen, blue = it went up, gray = no change.
    cmap = LinearSegmentedColormap.from_list("drop", ["#2a78d6", "#f0efec", "#e34948"])
    limit = max(np.nanmax(np.abs(drop.to_numpy())), 0.01) * 100
    fig, ax = plt.subplots(figsize=(8, 1.2 + 1.0 * len(models)))
    image = ax.imshow(drop.to_numpy() * 100, cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                      aspect="auto")
    for i, j in itertools.product(range(len(models)), range(len(DOMAINS))):
        d, s = drop.iloc[i, j] * 100, std.iloc[i, j] * 100
        if np.isnan(d):
            continue
        label = f"{d:+.1f} pts\n± {s:.1f}" + ("\n(n.s.)" if abs(d) < s else "")
        ax.text(j, i, label, ha="center", va="center", color="#222222", fontsize=9)
    ax.set_xticks(range(len(DOMAINS)), [f"{d}\n(n_test={n})" for d, n in zip(DOMAINS, n_test)])
    ax.set_yticks(range(len(models)), [MODEL_NAMES[m] for m in models])
    ax.set_xlabel("Held-out domain")
    ax.set_title("Accuracy drop when the test domain is unseen in training\n"
                 "(in-domain − leave-one-domain-out, mean ± std over 3 seeds; n.s. = |drop| < std)")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="Drop (pts)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_splits(load_data())
    train, val, test = data["train"], data["val"], data["test"]

    # N = the smallest per-family training count over the 4 LODO folds.
    n_per_family = min(train[train["domain"] != d]["LLM_name"].value_counts().min() for d in DOMAINS)
    print(f"Size control: every training set = {n_per_family} rows per family ({n_per_family * len(FAMILIES)} total)")

    done = set()
    if RUNS_FILE.exists():
        saved = pd.read_csv(RUNS_FILE)
        done = set(zip(saved["model"], saved["held_out_domain"], saved["condition"], saved["seed"]))

    for model_name, domain, condition, seed in itertools.product(args.models, DOMAINS, CONDITIONS, args.seeds):
        if (model_name, domain, condition, seed) in done:
            print(f"skip {model_name} / {domain} / {condition} / seed {seed}")
            continue
        seen = (lambda rows: rows) if condition == "in_domain" else (lambda rows: rows[rows["domain"] != domain])
        fold_train = subsample(seen(train), n_per_family, seed)
        fold_val = seen(val).reset_index(drop=True)
        fold_test = test[test["domain"] == domain].reset_index(drop=True)

        print(f"\n=== {model_name} / held out {domain} / {condition} / seed {seed}: "
              f"train {len(fold_train)}, val {len(fold_val)}, test {len(fold_test)} ===")
        start = time.time()
        if model_name == "tfidf_lr":
            y_pred = predict_tfidf(fold_train, fold_val, fold_test, seed)
        else:
            y_pred = predict_neural(model_name, fold_train, fold_val, fold_test, seed, device)
        y_true = fold_test["LLM_name"].to_numpy()

        run = {"model": model_name, "held_out_domain": domain, "condition": condition, "seed": seed,
               "acc": accuracy_score(y_true, y_pred),
               "macro_f1": f1_score(y_true, y_pred, labels=FAMILIES, average="macro", zero_division=0),
               "n_test": len(fold_test), "n_train": len(fold_train), "train_time_s": round(time.time() - start, 1)}
        recalls = recall_score(y_true, y_pred, labels=FAMILIES, average=None, zero_division=0)
        family_rows = [{"model": model_name, "held_out_domain": domain, "condition": condition, "seed": seed,
                        "family": family, "n_test": int((y_true == family).sum()), "recall": recall}
                       for family, recall in zip(FAMILIES, recalls)]
        append_rows(RUNS_FILE, [run])
        append_rows(FAMILY_FILE, family_rows)
        print(f"  acc {run['acc']:.3f}  macro-F1 {run['macro_f1']:.3f}  ({run['train_time_s']:.0f}s)")

    summary, _ = summarize()
    print("\nSaved results/rq3_summary.csv, results/rq3_per_family.csv, figures/rq3_drop_heatmap.png")
    print(summary[["model", "held_out_domain", "acc_in_domain_mean", "acc_lodo_mean", "drop_acc_mean",
                   "drop_acc_std", "drop_acc_meaningful", "n_seeds"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Train TextCNN and BiLSTM on 3 input conditions x 3 seeds (18 runs) and save all results.

Requires dataset.csv, splits.json and results/baselines.csv
(scripts/prepare_arena.py -> scripts/make_splits.py -> scripts/baselines.py).

Usage (from the project root):
    python scripts/run_neural.py
    python scripts/run_neural.py --models textcnn --conditions output_only --seeds 448 --max-epochs 2   # quick check

Runs already saved in results/neural_runs.csv are skipped, so an interrupted grid can simply be
restarted. Use --fresh to start over.

Outputs:
    results/neural_runs.csv          one row per run
    results/neural_preds.csv         test prediction for every row of every run (figures are rebuilt from it)
    results/neural_history.csv       train loss / val scores per epoch, every run
    results/neural_summary.csv       mean and std per model x condition
    results/acc_by_length.csv        output-only test accuracy per output-length quartile
    results/comparison.csv           baselines + neural models, mean ± std, all conditions
    results/comparison_flags.csv     pairs whose difference is smaller than the larger std
    figures/curves_<model>_<condition>.png   train loss + val macro-F1 per epoch (seed 448)
    figures/cm_<model>_output.png            test confusion matrix, output only (seed 448)
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of scripts/)
sys.path.insert(0, str(ROOT))  # so "from src... import" works when run as a script
from src.data import CONDITIONS, FAMILIES, build_vocab, encode, load_data, load_splits  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import make_loader, predict, set_seeds, train_model  # noqa: E402

MODELS = ["textcnn", "bilstm"]
SEEDS = [448, 449, 450]
PLOT_SEED = 448  # curves and confusion matrices are drawn for this seed
CHANCE = 1 / len(FAMILIES)
METHOD_NAMES = {"majority": "Majority", "length_lr": "Length-only LR", "tfidf_lr": "TF-IDF + LR",
                "textcnn": "TextCNN", "bilstm": "BiLSTM"}


def plot_curves(history, model_name, condition, best_epoch, path):
    history = pd.DataFrame(history)
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(history["epoch"], history["train_loss"], marker="o", color="#2a78d6")
    ax_loss.set_title("Train loss")
    ax_f1.plot(history["epoch"], history["val_macro_f1"], marker="o", color="#eb6834")
    ax_f1.axhline(CHANCE, color="#888888", linestyle=":", label="chance (20%)")
    ax_f1.set_title("Validation macro-F1")
    ax_f1.set_ylim(0, 1)
    for ax in (ax_loss, ax_f1):
        ax.axvline(best_epoch, color="#333333", linestyle="--", linewidth=1, label=f"best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.grid(color="#e5e5e5", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_f1.legend(frameon=False)
    fig.suptitle(f"{METHOD_NAMES[model_name]}, {condition} (seed {PLOT_SEED})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, model_name, path):
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(FAMILIES)))
    row_pct = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)  # each row sums to 100%
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(FAMILIES)):
        for j in range(len(FAMILIES)):
            color = "white" if row_pct[i, j] > 0.6 else "#222222"
            ax.text(j, i, f"{row_pct[i, j]:.0%}\n({matrix[i, j]})", ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(len(FAMILIES)), FAMILIES)
    ax.set_yticks(range(len(FAMILIES)), FAMILIES)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.set_title(f"{METHOD_NAMES[model_name]}, output only (test set, seed {PLOT_SEED})\nrow % (count)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Share of true family")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def accuracy_by_length(test_df, y_true, y_pred, model_name, seed):
    """Test accuracy inside each quartile of output length (in words)."""
    words = test_df["LLM_output"].str.split().str.len()
    quartile = pd.qcut(words, 4, labels=["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"])
    rows = []
    for q in quartile.cat.categories:
        inside = (quartile == q).to_numpy()
        rows.append({"model": model_name, "seed": seed, "quartile": q,
                     "words_min": words[inside].min(), "words_max": words[inside].max(), "n": inside.sum(),
                     "acc": accuracy_score(y_true[inside], y_pred[inside]),
                     "macro_f1": f1_score(y_true[inside], y_pred[inside], average="macro")})
    return rows


def comparison_tables(neural_summary):
    """Put the baselines and the neural models in one table, and flag differences that are within noise."""
    baselines = pd.read_csv(ROOT / "results" / "baselines.csv").rename(columns={"baseline": "method"})
    neural = neural_summary.rename(columns={"model": "method"})
    columns = ["method", "condition", "test_acc_mean", "test_acc_std", "test_macro_f1_mean", "test_macro_f1_std"]
    both = pd.concat([baselines[columns], neural[columns]], ignore_index=True)
    both["method"] = both["method"].map(METHOD_NAMES)

    # A difference is "not meaningful" if it is smaller than the larger of the two stds.
    flags = []
    keyed = both.set_index(["method", "condition"])
    pairs = [((m1, c), (m2, c)) for c in CONDITIONS for m1, m2 in itertools.combinations(both["method"].unique(), 2)]
    pairs += [((m, c1), (m, c2)) for m in both["method"].unique() for c1, c2 in itertools.combinations(CONDITIONS, 2)]
    for a, b in pairs:
        for metric in ["test_acc", "test_macro_f1"]:
            diff = keyed.loc[a, f"{metric}_mean"] - keyed.loc[b, f"{metric}_mean"]
            max_std = max(keyed.loc[a, f"{metric}_std"], keyed.loc[b, f"{metric}_std"])
            flags.append({"a": f"{a[0]} / {a[1]}", "b": f"{b[0]} / {b[1]}", "metric": metric,
                          "diff": round(diff, 4), "max_std": round(max_std, 4),
                          "meaningful": abs(diff) >= max_std})
    return both, pd.DataFrame(flags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--fresh", action="store_true", help="ignore saved runs and start over")
    args = parser.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU only'}")

    df = load_data()
    data = load_splits(df)
    label_id = {family: i for i, family in enumerate(FAMILIES)}
    labels = {name: [label_id[f] for f in part["LLM_name"]] for name, part in data.items()}
    # Vocabulary from TRAIN prompts and masked outputs only.
    vocab = build_vocab(pd.concat([data["train"]["LLM_Input"], data["train"]["LLM_output_masked"]]))
    print(f"Vocab size: {len(vocab)} (train split, min_freq 2)")
    encoded = {c: {name: encode(part, c, vocab) for name, part in data.items()} for c in args.conditions}
    for c in args.conditions:
        lengths = [len(seq) for seq in encoded[c]["train"]]
        print(f"  {c}: median {np.median(lengths):.0f} tokens, {np.mean(np.array(lengths) == 512):.0%} hit the 512 cap")

    # Saved results from earlier (possibly interrupted) runs; those runs are skipped below.
    files = {name: ROOT / "results" / f"neural_{name}.csv" for name in ["runs", "history", "preds"]}
    saved = {name: (pd.read_csv(path).to_dict("records") if path.exists() and not args.fresh else [])
             for name, path in files.items()}
    runs, histories, preds = saved["runs"], saved["history"], saved["preds"]
    done = {(r["model"], r["condition"], r["seed"]) for r in runs}

    for model_name, condition, seed in itertools.product(args.models, args.conditions, args.seeds):
        if (model_name, condition, seed) in done:
            print(f"skip {model_name} / {condition} / seed {seed} (already in neural_runs.csv)")
            continue
        print(f"\n=== {model_name} / {condition} / seed {seed} ===")
        set_seeds(seed)  # before building the model, so the initial weights depend on the seed
        model = build_model(model_name, len(vocab), len(FAMILIES)).to(device)
        train_loader = make_loader(encoded[condition]["train"], labels["train"], shuffle=True, seed=seed)
        val_loader = make_loader(encoded[condition]["val"], labels["val"], shuffle=False, seed=seed)
        test_loader = make_loader(encoded[condition]["test"], labels["test"], shuffle=False, seed=seed)

        start = time.time()
        best_epoch, history = train_model(model, train_loader, val_loader, device, max_epochs=args.max_epochs)
        train_time = time.time() - start
        y_pred, y_true = predict(model, test_loader, device)

        run = {"model": model_name, "condition": condition, "seed": seed, "best_epoch": best_epoch,
               "epochs_run": len(history),
               "test_acc": accuracy_score(y_true, y_pred), "test_macro_f1": f1_score(y_true, y_pred, average="macro"),
               "params": sum(p.numel() for p in model.parameters()), "train_time_s": round(train_time, 1)}
        runs.append(run)
        histories += [{"model": model_name, "condition": condition, "seed": seed, **h} for h in history]
        preds += [{"model": model_name, "condition": condition, "seed": seed, "test_row": i,
                   "y_true": int(t), "y_pred": int(p)} for i, (t, p) in enumerate(zip(y_true, y_pred))]
        print(f"  best epoch {best_epoch}, test acc {run['test_acc']:.3f}, test macro-F1 {run['test_macro_f1']:.3f}, "
              f"{train_time:.0f}s")

        # Save after every run, so a crash never loses finished runs.
        for name, rows in [("runs", runs), ("history", histories), ("preds", preds)]:
            pd.DataFrame(rows).to_csv(files[name], index=False)

    # Figures and the length analysis are rebuilt from the saved files, so resumed runs are included.
    runs, histories, preds = pd.DataFrame(runs), pd.DataFrame(histories), pd.DataFrame(preds)
    length_rows = []
    for (model_name, condition, seed), run_preds in preds.groupby(["model", "condition", "seed"]):
        run_preds = run_preds.sort_values("test_row")
        y_true, y_pred = run_preds["y_true"].to_numpy(), run_preds["y_pred"].to_numpy()
        if condition == "output_only":
            length_rows += accuracy_by_length(data["test"], y_true, y_pred, model_name, seed)
        if seed == PLOT_SEED:
            is_run = ((histories["model"] == model_name) & (histories["condition"] == condition)
                      & (histories["seed"] == seed))
            best_epoch = runs.set_index(["model", "condition", "seed"]).loc[(model_name, condition, seed), "best_epoch"]
            plot_curves(histories[is_run], model_name, condition, best_epoch,
                        ROOT / "figures" / f"curves_{model_name}_{condition}.png")
            if condition == "output_only":
                plot_confusion(y_true, y_pred, model_name, ROOT / "figures" / f"cm_{model_name}_output.png")
    if length_rows:
        pd.DataFrame(length_rows).to_csv(ROOT / "results" / "acc_by_length.csv", index=False)

    summary = runs.groupby(["model", "condition"], sort=False).agg(
        test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
        test_macro_f1_mean=("test_macro_f1", "mean"), test_macro_f1_std=("test_macro_f1", "std"),
        best_epoch_mean=("best_epoch", "mean"), params=("params", "first"), train_time_s_mean=("train_time_s", "mean"),
    ).reset_index().round(4)
    summary.to_csv(ROOT / "results" / "neural_summary.csv", index=False)
    print("\nNeural summary (saved to results/neural_summary.csv):")
    print(summary.to_string(index=False))

    if length_rows:
        by_length = pd.DataFrame(length_rows).groupby(["model", "quartile"], observed=True)["acc"].agg(["mean", "std"])
        print("\nOutput-only test accuracy by output-length quartile (saved to results/acc_by_length.csv):")
        print(by_length.round(3).to_string())

    # The full comparison only makes sense after the full grid.
    if set(args.models) == set(MODELS) and set(args.conditions) == set(CONDITIONS):
        comparison, flags = comparison_tables(summary)
        comparison.round(4).to_csv(ROOT / "results" / "comparison.csv", index=False)
        flags.to_csv(ROOT / "results" / "comparison_flags.csv", index=False)
        print("\nSaved results/comparison.csv and results/comparison_flags.csv")
        print(flags[~flags["meaningful"]].to_string(index=False))


if __name__ == "__main__":
    main()

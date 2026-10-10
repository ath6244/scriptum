"""Statistical checks for the RQ1/RQ2 comparison table (CPU only).

Requires dataset.csv, splits.json, results/baselines_runs.csv and results/neural_preds.csv.

Usage (from the project root):
    python scripts/stats_rq1_rq2.py

Steps and outputs:
    1. Per-row test predictions for every method (baselines are refit with their saved best C;
       neural predictions come from results/neural_preds.csv)
         -> results/preds/baseline_preds.csv, results/preds/neural_preds.csv
    2. Grouped bootstrap 95% CIs (2000 resamples of test prompts)  -> results/comparison_ci.csv
    3. McNemar tests on the same test rows (seed 448)              -> results/mcnemar.csv
    4. Exact binomial test of input-only accuracy vs p = 0.20      -> results/binomial_input_only.csv
    5. Output-length quartiles: family mix + accuracy per model    -> results/length_quartiles.csv
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of scripts/)
sys.path.insert(0, str(ROOT))  # so "from src... / from scripts... import" works when run as a script
from scripts.baselines import get_text, make_features, make_model  # noqa: E402
from src.data import CONDITIONS, FAMILIES, load_data, load_splits  # noqa: E402
from src.stats import acc_and_macro_f1, binomial_vs_chance, bootstrap, ci, confusion, mcnemar  # noqa: E402

K = len(FAMILIES)
CHANCE = 1 / K
BASELINES = ["majority", "length_lr", "tfidf_lr"]
METHODS = BASELINES + ["textcnn", "bilstm"]
NAMES = {"majority": "Majority", "length_lr": "Length-only LR", "tfidf_lr": "TF-IDF + LR",
         "textcnn": "TextCNN", "bilstm": "BiLSTM"}
PRED_DIR = ROOT / "results" / "preds"
BASELINE_SEED = 0  # baselines are deterministic, so one seed's predictions stand for all three


def baseline_predictions(data):
    """Refit each baseline with the C it chose on val (saved in baselines_runs.csv) and predict test."""
    path = PRED_DIR / "baseline_preds.csv"
    if path.exists():
        return pd.read_csv(path)
    runs = pd.read_csv(ROOT / "results" / "baselines_runs.csv")
    rows = []
    for baseline in BASELINES:
        for condition in CONDITIONS:
            run = runs[(runs["baseline"] == baseline) & (runs["condition"] == condition)
                       & (runs["seed"] == BASELINE_SEED)].iloc[0]
            random.seed(BASELINE_SEED)
            np.random.seed(BASELINE_SEED)
            texts = [get_text(data[name], condition) for name in ["train", "val", "test"]]
            f_train, _, f_test = make_features(baseline, *texts)
            C = None if pd.isna(run["best_C"]) else run["best_C"]
            model = make_model(baseline, C, BASELINE_SEED).fit(f_train, data["train"]["LLM_name"])
            y_pred = model.predict(f_test)
            acc = (y_pred == data["test"]["LLM_name"]).mean()
            # The refit must reproduce the saved result exactly, otherwise something changed.
            assert abs(acc - run["test_acc"]) < 1e-9, f"{baseline}/{condition}: {acc} != {run['test_acc']}"
            print(f"  {baseline:10s} {condition:13s} C={C}  test acc {acc:.4f} (matches baselines_runs.csv)")
            rows += [{"method": baseline, "condition": condition, "seed": BASELINE_SEED, "test_row": i,
                      "y_true": t, "y_pred": p}
                     for i, (t, p) in enumerate(zip(data["test"]["LLM_name"], y_pred))]
    preds = pd.DataFrame(rows)
    preds.to_csv(path, index=False)
    return preds


def neural_predictions():
    preds = pd.read_csv(ROOT / "results" / "neural_preds.csv").rename(columns={"model": "method"})
    preds["y_true"] = preds["y_true"].map(lambda i: FAMILIES[i])
    preds["y_pred"] = preds["y_pred"].map(lambda i: FAMILIES[i])
    preds.to_csv(PRED_DIR / "neural_preds.csv", index=False)
    return preds


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    data = load_splits(df)
    test = data["test"]
    label_id = {family: i for i, family in enumerate(FAMILIES)}
    y_true = test["LLM_name"].map(label_id).to_numpy()

    # --- 1. Per-row predictions ------------------------------------------------
    print("1. Per-row test predictions")
    preds = pd.concat([baseline_predictions(data), neural_predictions()], ignore_index=True)

    def get_preds(method, condition, seed=None):
        """List of integer prediction arrays (one per seed), in test-row order."""
        rows = preds[(preds["method"] == method) & (preds["condition"] == condition)]
        if seed is not None and method in ("textcnn", "bilstm"):
            rows = rows[rows["seed"] == seed]
        out = []
        for _, run in rows.groupby("seed"):
            run = run.sort_values("test_row")
            assert (run["y_true"].map(label_id).to_numpy() == y_true).all(), "prediction rows out of order"
            out.append(run["y_pred"].map(label_id).to_numpy())
        return out

    # --- 2. Bootstrap CIs ------------------------------------------------------
    print("\n2. Grouped bootstrap CIs (2000 resamples of test prompts)")
    groups = test["prompt_id"]
    ci_rows = []
    for method in METHODS:
        for condition in CONDITIONS:
            seed_preds = get_preds(method, condition)
            point = np.mean([acc_and_macro_f1(confusion(y_true, p, K)) for p in seed_preds], axis=0)
            samples = bootstrap(y_true, seed_preds, groups, K)
            (acc_lo, acc_hi), (f1_lo, f1_hi) = ci(samples[:, 0]), ci(samples[:, 1])
            ci_rows.append({"method": NAMES[method], "condition": condition, "n_seeds": len(seed_preds),
                            "acc": point[0], "acc_ci_low": acc_lo, "acc_ci_high": acc_hi,
                            "macro_f1": point[1], "macro_f1_ci_low": f1_lo, "macro_f1_ci_high": f1_hi})
    ci_table = pd.DataFrame(ci_rows).round(4)
    ci_table.to_csv(ROOT / "results" / "comparison_ci.csv", index=False)
    print(ci_table.to_string(index=False))

    # --- 3. McNemar --------------------------------------------------------------
    print("\n3. McNemar tests (seed 448; baselines are deterministic)")
    comparisons = [
        (("tfidf_lr", "output_only"), ("textcnn", "output_only")),
        (("textcnn", "output_only"), ("bilstm", "output_only")),
        (("textcnn", "input_output"), ("textcnn", "output_only")),
        (("bilstm", "input_output"), ("bilstm", "output_only")),
    ]
    mc_rows = []
    for a, b in comparisons:
        pred_a, pred_b = get_preds(*a, seed=448)[0], get_preds(*b, seed=448)[0]
        result = mcnemar(y_true, pred_a, pred_b)
        mc_rows.append({"a": f"{NAMES[a[0]]} / {a[1]}", "b": f"{NAMES[b[0]]} / {b[1]}",
                        "acc_a": (pred_a == y_true).mean(), "acc_b": (pred_b == y_true).mean(), **result})
    mc_table = pd.DataFrame(mc_rows)
    mc_table["significant_0.05"] = mc_table["p_value"] < 0.05
    mc_table.round(6).to_csv(ROOT / "results" / "mcnemar.csv", index=False)
    print(mc_table.round(4).to_string(index=False))

    # --- 4. Binomial test of input-only vs chance ----------------------------------
    print(f"\n4. Exact binomial test, input-only accuracy vs p = {CHANCE:.2f}")
    bin_rows = []
    for method in METHODS:
        for i, p in enumerate(get_preds(method, "input_only")):
            seed = BASELINE_SEED if method in BASELINES else [448, 449, 450][i]
            bin_rows.append({"method": NAMES[method], "seed": seed,
                             **binomial_vs_chance(int((p == y_true).sum()), len(y_true), CHANCE)})
    bin_table = pd.DataFrame(bin_rows)
    bin_table["above_chance_0.05"] = (bin_table["p_value"] < 0.05) & (bin_table["acc"] > CHANCE)
    bin_table.round(6).to_csv(ROOT / "results" / "binomial_input_only.csv", index=False)
    print(bin_table.round(4).to_string(index=False))

    # --- 5. Output-length quartiles ---------------------------------------------------
    print("\n5. Output-length quartiles (test set, words in LLM_output)")
    words = test["LLM_output"].str.split().str.len()
    quartile = pd.qcut(words, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    q_rows = []
    for q in quartile.cat.categories:
        inside = (quartile == q).to_numpy()
        row = {"quartile": q, "words_min": words[inside].min(), "words_max": words[inside].max(), "n": inside.sum()}
        for family in FAMILIES:  # family mix of this quartile
            row[f"share_{family}"] = (test["LLM_name"][inside] == family).mean()
        for method in ["length_lr", "tfidf_lr", "textcnn", "bilstm"]:
            seed_preds = get_preds(method, "output_only")
            # Overall per-family recall of this model (all quartiles together).
            overall_recall = np.mean([np.diag(confusion(y_true, p, K)) / np.bincount(y_true, minlength=K)
                                      for p in seed_preds], axis=0)
            accs, balanced = [], []
            for p in seed_preds:
                cm = confusion(y_true[inside], p[inside], K)
                accs.append(np.trace(cm) / cm.sum())
                balanced.append(np.mean(np.diag(cm) / cm.sum(axis=1).clip(min=1)))  # mean per-family recall
            row[f"{method}_acc"] = np.mean(accs)
            # Balanced accuracy weights every family equally, so it is not affected by the quartile's family mix.
            row[f"{method}_balanced_acc"] = np.mean(balanced)
            # What accuracy would be in this quartile if only the family mix changed (per-family recall fixed).
            row[f"{method}_acc_expected_from_mix"] = sum(row[f"share_{f}"] * overall_recall[i]
                                                         for i, f in enumerate(FAMILIES))
        q_rows.append(row)
    q_table = pd.DataFrame(q_rows).round(4)
    q_table.to_csv(ROOT / "results" / "length_quartiles.csv", index=False)
    print(q_table.T.to_string(header=False))


if __name__ == "__main__":
    main()

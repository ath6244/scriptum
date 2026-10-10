"""Near-duplicate prompt check: do almost-identical prompts end up in different splits?

prompt_id hashes the exact prompt text, so "Hi!" and "hi" get different ids and can land in
different splits. Here prompts are normalized (lowercase, punctuation and whitespace removed) and
re-hashed. A normalized group that spans more than one split is a possible leak.

If more than 2% of test rows are affected, every method's test accuracy is recomputed without
those rows (no retraining needed; uses results/preds/ from scripts/stats_rq1_rq2.py).

Usage (from the project root):
    python scripts/check_near_duplicates.py

Outputs:
    results/near_duplicates.csv           summary numbers
    results/near_duplicate_groups.csv     every normalized group that spans more than one split
    results/near_duplicates_reeval.csv    accuracy with and without the affected test rows (if > 2%)
"""

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of scripts/)
sys.path.insert(0, str(ROOT))  # so "from src... import" works when run as a script
from src.data import load_data, load_splits  # noqa: E402

THRESHOLD = 0.02  # re-evaluate if more than 2% of test rows are affected
NON_ALNUM = re.compile(r"[\W_]+")  # everything that is not a letter or digit (including spaces)


def normalized_id(text):
    return hashlib.sha1(NON_ALNUM.sub("", text.lower()).encode("utf-8")).hexdigest()[:12]


def main():
    data = load_splits(load_data())
    df = pd.concat([part.assign(split=name) for name, part in data.items()], ignore_index=True)
    df["norm_id"] = df["LLM_Input"].map(normalized_id)

    splits_per_group = df.groupby("norm_id")["split"].nunique()
    leaky_groups = splits_per_group[splits_per_group > 1].index
    test = df[df["split"] == "test"]
    affected = test["norm_id"].isin(leaky_groups)

    summary = {
        "exact_prompt_ids": df["prompt_id"].nunique(),
        "normalized_groups": df["norm_id"].nunique(),
        "groups_spanning_splits": len(leaky_groups),
        "test_rows": len(test),
        "test_rows_affected": int(affected.sum()),
        "share_test_rows_affected": affected.mean(),
    }
    pd.DataFrame([summary]).round(4).to_csv(ROOT / "results" / "near_duplicates.csv", index=False)
    for key, value in summary.items():
        print(f"{key:28s} {value:.4f}" if isinstance(value, float) else f"{key:28s} {value}")

    leaky_rows = df[df["norm_id"].isin(leaky_groups)]
    groups = (leaky_rows.groupby("norm_id").agg(splits=("split", lambda s: ",".join(sorted(set(s)))),
                                                n_rows=("split", "size"))
              .sort_values("n_rows", ascending=False))
    groups.to_csv(ROOT / "results" / "near_duplicate_groups.csv")  # no prompt text: this file is published
    print("\nLargest groups spanning splits (prompt text shown on screen only):")
    for norm_id, group in groups.head(10).iterrows():
        variants = " || ".join(sorted(set(leaky_rows.loc[leaky_rows["norm_id"] == norm_id, "LLM_Input"])))
        print(f"  {group['splits']:16s} {group['n_rows']:3d} rows  {variants[:120]!r}")

    if summary["share_test_rows_affected"] > THRESHOLD:
        print(f"\nMore than {THRESHOLD:.0%} of test rows affected: recomputing accuracy without them")
        keep = ~affected.to_numpy()
        preds = pd.concat([pd.read_csv(ROOT / "results" / "preds" / "baseline_preds.csv"),
                           pd.read_csv(ROOT / "results" / "preds" / "neural_preds.csv")])
        rows = []
        for (method, condition, seed), run in preds.groupby(["method", "condition", "seed"]):
            correct = (run.sort_values("test_row")["y_true"] == run.sort_values("test_row")["y_pred"]).to_numpy()
            rows.append({"method": method, "condition": condition, "seed": seed,
                         "acc_all": correct.mean(), "acc_without_affected": correct[keep].mean()})
        reeval = pd.DataFrame(rows).groupby(["method", "condition"])[["acc_all", "acc_without_affected"]].mean()
        reeval["difference"] = reeval["acc_without_affected"] - reeval["acc_all"]
        reeval.round(4).to_csv(ROOT / "results" / "near_duplicates_reeval.csv")
        print(reeval.round(4).to_string())
    else:
        print(f"\nAt most {THRESHOLD:.0%} of test rows affected: no re-evaluation needed")


if __name__ == "__main__":
    main()

"""Masking report, top model_version per family, and the grouped 70/15/15 splits. Seed 448.

Usage (from the project root, after scripts/prepare_arena.py):
    python scripts/make_splits.py

Outputs:
    splits.json                      {"train": [prompt_ids], "val": [...], "test": [...]}
    results/masking_stats.csv        per family: share of outputs changed by masking
    results/masking_name_hits.csv    per family x name: how many outputs contain that name
    results/top_model_version.csv    most common model_version in each family and its share
    results/split_counts.csv         rows per split by LLM_name and by domain
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of scripts/)
sys.path.insert(0, str(ROOT))  # so "from src.data import ..." works when run as a script
from src.data import MODEL_NAMES, NAME_REGEX, SPLIT_SEED, load_data, make_splits  # noqa: E402


def main():
    random.seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['prompt_id'].nunique()} prompts")

    # --- Masking report -------------------------------------------------------
    df["changed"] = df["LLM_output"] != df["LLM_output_masked"]
    masking = df.groupby("LLM_name")["changed"].agg(outputs="size", changed="sum", share_changed="mean")
    masking.round(4).to_csv(results / "masking_stats.csv")
    print("\nOutputs changed by masking (saved to results/masking_stats.csv):")
    print(masking.round(3).to_string())

    # Which names caused the changes? Generic words like "meta" or "o3" can hit ordinary text.
    found = df["LLM_output"].str.findall(NAME_REGEX).map(lambda hits: {h.lower() for h in hits})
    hits = pd.DataFrame({name: found.map(lambda names, n=name.lower(): n in names) for name in MODEL_NAMES})
    hits = hits.groupby(df["LLM_name"]).sum().T
    hits.to_csv(results / "masking_name_hits.csv")
    print("\nOutputs containing each name (saved to results/masking_name_hits.csv):")
    print(hits.to_string())

    # --- Top model_version per family ----------------------------------------
    composition = pd.read_csv(results / "model_composition.csv")
    composition["share"] = composition["rows"] / composition.groupby("LLM_name")["rows"].transform("sum")
    top = composition.sort_values("rows", ascending=False).groupby("LLM_name").head(1).sort_values("LLM_name")
    top.round(4).to_csv(results / "top_model_version.csv", index=False)
    print("\nTop model_version per family (saved to results/top_model_version.csv):")
    print(top.round(3).to_string(index=False))

    # --- Grouped, domain-stratified splits -----------------------------------
    splits = make_splits(df)  # asserts that no prompt_id is in two splits
    with open(ROOT / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    split_of = {pid: name for name, ids in splits.items() for pid in ids}
    df["split"] = df["prompt_id"].map(split_of)

    counts = pd.concat([
        pd.crosstab(df["LLM_name"], df["split"]).assign(group="LLM_name"),
        pd.crosstab(df["domain"], df["split"]).assign(group="domain"),
    ])[["group", "train", "val", "test"]]
    counts.index.name = "value"
    counts.to_csv(results / "split_counts.csv")
    print(f"\nSaved splits.json. Prompts per split: { {k: len(v) for k, v in splits.items()} }")
    print("Rows per split (saved to results/split_counts.csv):")
    print(counts.to_string())


if __name__ == "__main__":
    main()

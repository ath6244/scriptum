"""Build dataset.csv from LMArena battles (lmarena-ai/arena-human-preference-140k). Seed 448.

Usage (from the project root):
    python scripts/prepare_arena.py
    python scripts/prepare_arena.py --per-family 1000     # smaller set

Each Arena battle is one user prompt answered by two anonymous models (side a and side b).
We turn every battle into two rows, one per side, so each row is one (input, model, output).

Output columns (in this order):
    LLM_name, LLM_Input, LLM_output, model_version, prompt_id, domain, source_id
  - LLM_name      model family (the class label), e.g. claude, openai, gemini
  - model_version exact Arena model name, e.g. claude-3-7-sonnet-20250219
  - prompt_id     short sha1 of LLM_Input, so identical prompts share an id (use it to group splits)
  - source_id     Arena battle id (the two sides of a battle share it)

After sampling, outputs that langdetect does not call English are dropped, and every
family is cut down to the smallest family's count so the classes stay balanced.

Also writes results/model_composition.csv, results/data_stats.csv, results/language_filter.csv,
and results/prepare_arena_log.txt (a copy of everything printed).
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # langdetect is random by default; this makes it give the same answer every run

SEED = 448
ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of scripts/)
REPO = "lmarena-ai/arena-human-preference-140k"
# Only these columns are read from the parquet files; the rest of the ~1.6 GB is skipped.
COLUMNS = ["id", "model_a", "model_b", "evaluation_order", "language", "is_code",
           "category_tag", "conversation_a", "conversation_b"]
N_FAMILIES = 5
MIN_OUTPUT_WORDS = 5
LANGID_CHARS = 2000  # langdetect only needs a few sentences; long outputs are cut for speed
# Fenced code blocks (```...```) are removed before language ID: code is not English prose and
# would confuse the detector. An unclosed fence runs to the end of the text.
CODE_BLOCK = re.compile(r"```.*?(```|$)", flags=re.DOTALL)
MIN_CELL = 50  # warn when a family x domain cell has fewer rows than this
OUT_COLUMNS = ["LLM_name", "LLM_Input", "LLM_output", "model_version", "prompt_id", "domain", "source_id"]

# (prefix, family). Checked in order; a model that matches no prefix is dropped.
# "gemini" does not match "gemma", so Google's open Gemma models are dropped on purpose.
FAMILY_PREFIXES = [
    ("claude", "claude"),
    ("gpt", "openai"), ("chatgpt", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
    ("gemini", "gemini"),
    ("qwen", "qwen"), ("qwq", "qwen"),
    ("deepseek", "deepseek"),
    ("llama", "llama"),
    ("mistral", "mistral"),
]


class Tee:
    """Send everything printed to the screen and to a log file."""

    def __init__(self, path):
        self.screen = sys.stdout
        self.file = open(path, "w", encoding="utf-8")

    def write(self, text):
        self.screen.write(text)
        self.file.write(text)

    def flush(self):
        self.screen.flush()
        self.file.flush()


def model_family(model_name):
    name = model_name.lower()
    for prefix, family in FAMILY_PREFIXES:
        if name.startswith(prefix):
            return family
    return None


def turn_text(turn):
    """Join the text items of one turn. Returns None if the turn has any non-text item (e.g. an image)."""
    parts = []
    for item in turn["content"]:
        if item["type"] != "text":
            return None
        parts.append(item["text"] or "")
    return "\n".join(parts).strip()


def single_turn(conversation):
    """Return (user_text, assistant_text) if the conversation is exactly one user + one assistant turn."""
    if len(conversation) != 2:
        return None
    user, assistant = conversation
    if user["role"] != "user" or assistant["role"] != "assistant":
        return None
    user_text, assistant_text = turn_text(user), turn_text(assistant)
    if user_text is None or assistant_text is None:
        return None
    return user_text, assistant_text


def get_domain(is_code, category_tag):
    tags = category_tag or {}
    if is_code:
        return "coding"
    if (tags.get("math_v0.1") or {}).get("math"):
        return "math"
    if (tags.get("creative_writing_v0.1") or {}).get("creative_writing"):
        return "creative_writing"
    return "general"


def load_battles():
    """Download the parquet shards and return one row per (battle, side) after the battle-level filters."""
    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    rows = []
    counts = {"read": 0, "en_first_turn": 0, "single_turn_text_only": 0}
    for file in files:
        path = hf_hub_download(REPO, file, repo_type="dataset")
        counts["read"] += pq.ParquetFile(path).metadata.num_rows
        # Filter on language and evaluation_order while reading, so we never hold the other rows.
        table = pq.read_table(path, columns=COLUMNS,
                              filters=[("language", "==", "en"), ("evaluation_order", "==", 1)])
        counts["en_first_turn"] += table.num_rows
        for battle in table.to_pylist():
            side_a = single_turn(battle["conversation_a"])
            side_b = single_turn(battle["conversation_b"])
            if side_a is None or side_b is None:
                continue
            counts["single_turn_text_only"] += 1
            domain = get_domain(battle["is_code"], battle["category_tag"])
            for side, (prompt, output), model in [("a", side_a, battle["model_a"]),
                                                  ("b", side_b, battle["model_b"])]:
                rows.append({"source_id": battle["id"], "side": side, "LLM_Input": prompt,
                             "LLM_output": output, "model_version": model, "domain": domain})
        print(f"  {file}: {table.num_rows} English first-turn battles")
    print(f"Battles read: {counts['read']}, English + evaluation_order 1: {counts['en_first_turn']}, "
          f"single-turn text-only on both sides: {counts['single_turn_text_only']}")
    return pd.DataFrame(rows)


def balanced_sample(df, families, per_family, rng):
    """Take per_family rows from each family, taking both sides of cross-family battles first."""
    df = df[df["LLM_name"].isin(families)]
    taken = {family: 0 for family in families}
    chosen = []  # row indices into df

    # 1) Battles where both sides survived cleaning and come from two different kept families.
    sides = df.groupby("source_id")
    pairs = [group for _, group in sides if len(group) == 2 and group["LLM_name"].nunique() == 2]
    for i in rng.permutation(len(pairs)):
        fam_1, fam_2 = pairs[i]["LLM_name"]
        if taken[fam_1] < per_family and taken[fam_2] < per_family:
            chosen.extend(pairs[i].index)
            taken[fam_1] += 1
            taken[fam_2] += 1
    n_paired = len(chosen)

    # 2) Fill each family's remaining quota with single rows not already chosen.
    rest = df.drop(index=chosen)
    for family in families:
        pool = rest[rest["LLM_name"] == family]
        need = per_family - taken[family]
        chosen.extend(pool.index[rng.permutation(len(pool))[:need]])
    print(f"Rows from paired cross-family battles: {n_paired} ({n_paired // 2} battles); "
          f"single-side rows: {len(chosen) - n_paired}")
    return df.loc[chosen]


def output_language(text):
    """Language code of the prose in an output, or "code" if there is (almost) no prose outside code blocks."""
    prose = CODE_BLOCK.sub(" ", text).strip()[:LANGID_CHARS]
    if len(prose.split()) < MIN_OUTPUT_WORDS:
        return "code"
    try:
        return detect(prose)
    except LangDetectException:  # raised when the text has no letters at all
        return "unknown"


def downsample_keep_pairs(df, rng):
    """Cut every family to the smallest family's count. Single rows are dropped before paired rows."""
    n = df["LLM_name"].value_counts().min()
    df = df.assign(paired=df["source_id"].map(df["source_id"].value_counts()) == 2)
    kept = []
    for family, rows in df.groupby("LLM_name"):
        rows = rows.iloc[rng.permutation(len(rows))]  # random order within the family...
        rows = rows.sort_values("paired", ascending=False, kind="stable")  # ...but paired rows first
        kept.append(rows.head(n))
    return pd.concat(kept).drop(columns="paired")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=2000, help="cap on rows per family")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    sys.stdout = Tee(results / "prepare_arena_log.txt")

    df = load_battles()
    print(f"Rows after unpivot (2 per battle): {len(df)}")

    # Map model names to families and show the full table so the rules can be checked by eye.
    df["LLM_name"] = df["model_version"].map(model_family)
    table = (df.groupby("model_version", dropna=False)["LLM_name"].first().fillna("(dropped)")
             .reset_index().sort_values(["LLM_name", "model_version"]))
    table["rows"] = table["model_version"].map(df["model_version"].value_counts())
    print("\nmodel_version -> LLM_name:")
    print(table.to_string(index=False))
    df = df[df["LLM_name"].notna()]
    print(f"\nRows with a mapped family: {len(df)}")

    # Clean before counting, so the counts below are rows we can actually use.
    words = df["LLM_output"].str.split().str.len()
    df = df[(df["LLM_Input"] != "") & (words >= MIN_OUTPUT_WORDS)]
    print(f"Rows after dropping empty inputs and outputs under {MIN_OUTPUT_WORDS} words: {len(df)}")

    usable = df["LLM_name"].value_counts()
    print("\nUsable rows per LLM_name:")
    print(usable.to_string())
    families = sorted(usable.index[:N_FAMILIES])
    per_family = min(args.per_family, usable[families].min())
    print(f"\nKeeping {families}, {per_family} rows each")

    sample = balanced_sample(df, families, per_family, rng)

    # Language filter on the outputs. language == "en" above only describes the prompt, and some
    # answers are in another language (e.g. translation requests).
    print("\nRunning language ID on outputs (langdetect, seed 0)...")
    sample["output_lang"] = sample["LLM_output"].map(output_language)
    is_english = sample["output_lang"].isin(["en", "code"])
    print("Detected output languages:", sample["output_lang"].value_counts().head(10).to_dict())
    lang_table = pd.DataFrame({
        "sampled": sample.groupby("LLM_name").size(),
        "code_only_kept": sample[sample["output_lang"] == "code"].groupby("LLM_name").size(),
        "dropped_non_english": sample[~is_english].groupby("LLM_name").size(),
    }).fillna(0).astype(int)
    sample = sample[is_english]
    lang_table["after_language_filter"] = sample.groupby("LLM_name").size()
    sample = downsample_keep_pairs(sample, rng)
    lang_table["final"] = sample.groupby("LLM_name").size()
    lang_table.to_csv(results / "language_filter.csv")
    print("Language filter per LLM_name (saved to results/language_filter.csv):")
    print(lang_table.to_string())
    print(f"Downsampled every family to {lang_table['final'].min()} rows (seed {SEED}); "
          f"battles with both sides kept: {(sample['source_id'].value_counts() == 2).sum()}")

    sample["prompt_id"] = [hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] for text in sample["LLM_Input"]]
    sample = sample.sort_values(["source_id", "side"])[OUT_COLUMNS].reset_index(drop=True)
    assert sample["LLM_name"].value_counts().nunique() == 1, "families are not balanced"
    sample.to_csv(ROOT / "dataset.csv", index=False)
    print(f"\nSaved {len(sample)} rows to dataset.csv")
    n_families = sample.groupby("prompt_id")["LLM_name"].nunique()
    print(f"Unique prompts: {len(n_families)}, answered by 2+ families: {(n_families >= 2).sum()}")

    crosstab = pd.crosstab(sample["LLM_name"], sample["domain"])
    print("\nLLM_name x domain:")
    print(crosstab.to_string())
    small = crosstab.stack()
    small = small[small < MIN_CELL]
    for (family, domain), n in small.items():
        print(f"  WARNING: {family} x {domain} has only {n} rows (< {MIN_CELL})")

    composition = (sample.groupby(["LLM_name", "model_version"]).size().rename("rows").reset_index()
                   .sort_values(["LLM_name", "rows"], ascending=[True, False]))
    composition.to_csv(results / "model_composition.csv", index=False)

    sample["words"] = sample["LLM_output"].str.split().str.len()
    stats = sample.groupby("LLM_name").agg(rows=("words", "size"), mean_words=("words", "mean"),
                                           median_words=("words", "median"))
    stats = stats.join(crosstab.add_prefix("n_")).round(1).reset_index()
    stats.to_csv(results / "data_stats.csv", index=False)
    print("\nSaved results/model_composition.csv and results/data_stats.csv")
    print(stats.to_string(index=False))

    # Example rows go to the screen only: the log is published, and these rows are not PII-screened.
    log = sys.stdout
    log.file.write("\n[Example rows are printed to the screen only, not saved to this log.]\n")
    log.file.close()
    sys.stdout = log.screen
    print("\nTwo random examples per LLM_name:")
    for family in families:
        rows = sample[sample["LLM_name"] == family]
        for i in rng.choice(len(rows), size=2, replace=False):
            row = rows.iloc[i]
            print(f"\n--- {family} | {row['model_version']} | {row['domain']} | prompt {row['prompt_id']}")
            print(f"INPUT:  {row['LLM_Input'][:300]!r}")
            print(f"OUTPUT: {row['LLM_output'][:500]!r}")


if __name__ == "__main__":
    main()

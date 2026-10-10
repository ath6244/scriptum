"""Load dataset.csv, add the masked output column, and make / load the grouped splits.

Every experiment should get its data from here:
    from src.data import load_data, load_splits
    df = load_data()                      # has LLM_output_masked
    data = load_splits(df)                # {"train": df, "val": df, "test": df}

Also the word-level tokenizer and vocabulary used by the neural models.
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of src/)
FAMILIES = ["claude", "gemini", "llama", "openai", "qwen"]  # fixed label order for tables and plots
SPLIT_SEED = 448

# Model and company names that would give the answer away. Replaced with [MODEL] in LLM_output_masked.
MODEL_NAMES = [
    "Claude", "Anthropic",
    "GPT", "ChatGPT", "OpenAI", "o3", "o4",
    "Gemini", "Bard", "Google", "DeepMind",
    "Qwen", "Alibaba", "Tongyi",
    "Llama", "Meta",
]
# \b = word boundary, so "Claude" matches but "Claudette" does not, and "GPT-4o" becomes "[MODEL]-4o".
NAME_REGEX = re.compile(r"\b(" + "|".join(MODEL_NAMES) + r")\b", flags=re.IGNORECASE)


def mask_names(text):
    return NAME_REGEX.sub("[MODEL]", text)


def load_data(path=ROOT / "dataset.csv"):
    """Read dataset.csv and add LLM_output_masked. Experiments use the masked column by default."""
    df = pd.read_csv(path, dtype={"prompt_id": str, "source_id": str})
    df["LLM_Input"] = df["LLM_Input"].fillna("").astype(str)
    df["LLM_output"] = df["LLM_output"].fillna("").astype(str)
    df["LLM_output_masked"] = df["LLM_output"].map(mask_names)
    return df


def make_splits(df, seed=SPLIT_SEED, val_frac=0.15, test_frac=0.15):
    """Split prompt_ids (not rows) 70/15/15, separately within each domain."""
    rng = np.random.default_rng(seed)

    # One domain per prompt. A few prompts got different domain tags in different battles;
    # use the most common one (ties broken alphabetically).
    prompt_domain = (df.groupby("prompt_id")["domain"]
                     .agg(lambda s: s.value_counts().sort_index().idxmax()))

    splits = {"train": [], "val": [], "test": []}
    for domain in sorted(prompt_domain.unique()):
        ids = sorted(prompt_domain.index[prompt_domain == domain])
        ids = list(rng.permutation(ids))
        n_val = round(len(ids) * val_frac)
        n_test = round(len(ids) * test_frac)
        splits["val"] += ids[:n_val]
        splits["test"] += ids[n_val:n_val + n_test]
        splits["train"] += ids[n_val + n_test:]

    splits = {name: sorted(str(i) for i in ids) for name, ids in splits.items()}
    check_splits(splits, df)
    return splits


def check_splits(splits, df):
    train, val, test = (set(splits[name]) for name in ["train", "val", "test"])
    assert not (train & val) and not (train & test) and not (val & test), \
        "a prompt_id appears in more than one split"
    assert train | val | test == set(df["prompt_id"]), "some prompt_id is in no split"


def load_splits(df, path=ROOT / "splits.json"):
    """Return {"train": rows, "val": rows, "test": rows} using the prompt_ids in splits.json."""
    with open(path) as f:
        splits = json.load(f)
    check_splits(splits, df)
    return {name: df[df["prompt_id"].isin(ids)].reset_index(drop=True) for name, ids in splits.items()}


# ---------------------------------------------------------------------------
# Tokenizer and vocabulary for the neural models
# ---------------------------------------------------------------------------
CONDITIONS = ["output_only", "input_only", "input_output"]
MAX_LEN = 512        # tokens per example
MAX_INPUT_LEN = 128  # in input_output, the prompt gets at most this many tokens; the output gets the rest
PAD, UNK, SEP = "<pad>", "<unk>", "[SEP]"  # ids 0, 1, 2

# A token is "[model]" (the mask), a run of letters/digits/underscores, or ONE punctuation character.
# So markdown like "## Title" -> "#", "#", "title" and "**bold**" -> "*", "*", "bold", "*", "*":
# formatting symbols stay visible to the model as their own tokens.
TOKEN_REGEX = re.compile(r"\[model\]|\w+|[^\w\s]")


def tokenize(text):
    return TOKEN_REGEX.findall(text.lower())


def build_vocab(texts, min_freq=2):
    """Map token -> id. Call this with TRAIN texts only, so val/test words never leak into the vocab."""
    counts = Counter(token for text in texts for token in tokenize(text))
    # sorted() so the same data always gives the same ids
    words = sorted(token for token, count in counts.items() if count >= min_freq)
    return {token: i for i, token in enumerate([PAD, UNK, SEP] + words)}


def encode(df, condition, vocab):
    """Turn each row into a list of token ids for one input condition. Long texts keep their beginning."""
    unk = vocab[UNK]

    def ids(text):
        return [vocab.get(token, unk) for token in tokenize(text)]

    sequences = []
    for prompt, output in zip(df["LLM_Input"], df["LLM_output_masked"]):
        if condition == "output_only":
            seq = ids(output)[:MAX_LEN]
        elif condition == "input_only":
            seq = ids(prompt)[:MAX_LEN]
        elif condition == "input_output":
            seq = ids(prompt)[:MAX_INPUT_LEN] + [vocab[SEP]]
            seq = seq + ids(output)[:MAX_LEN - len(seq)]
        else:
            raise ValueError(condition)
        sequences.append(seq or [unk])  # never return an empty sequence
    return sequences

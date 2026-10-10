"""Tests for [MODEL] name masking and prompt-grouped splits (src/data.py). No dataset.csv needed."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import MODEL_NAMES, check_splits, make_splits, mask_names

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
def test_there_are_16_masked_names():
    assert len(MODEL_NAMES) == 16
    assert len({name.lower() for name in MODEL_NAMES}) == 16  # no duplicates


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_name_is_masked(name):
    assert mask_names(f"Hello, I am {name}.") == "Hello, I am [MODEL]."


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_masking_ignores_case(name):
    text = f"{name.lower()} / {name.upper()} / {name.title()}"
    assert mask_names(text) == "[MODEL] / [MODEL] / [MODEL]"


def test_masking_replaces_every_occurrence():
    assert mask_names("Claude, Claude and Claude") == "[MODEL], [MODEL] and [MODEL]"


def test_masking_matches_whole_words_only():
    # Words that merely contain a name must stay unchanged.
    for text in ["Claudette", "metadata", "Googled", "Geminids", "llamas", "co3", "go4", "bardic"]:
        assert mask_names(text) == text


def test_masking_keeps_version_suffixes():
    assert mask_names("GPT-4o and o3-mini") == "[MODEL]-4o and [MODEL]-mini"
    assert mask_names("I'm Meta AI.") == "I'm [MODEL] AI."


def test_text_without_names_is_unchanged():
    text = "A stack is LIFO; a queue is FIFO."
    assert mask_names(text) == text


# ---------------------------------------------------------------------------
# Prompt-grouped splits
# ---------------------------------------------------------------------------
def synthetic_rows(n_prompts=300, seed=0):
    """Rows with repeated prompt_ids (2-4 answers per prompt) across four domains."""
    rng = np.random.default_rng(seed)
    domains = ["coding", "math", "creative_writing", "general"]
    rows = []
    for i in range(n_prompts):
        domain = domains[i % len(domains)]
        for _ in range(rng.integers(2, 5)):
            rows.append({"prompt_id": f"p{i:04d}", "domain": domain})
    return pd.DataFrame(rows)


def test_splits_never_share_a_prompt_id():
    splits = make_splits(synthetic_rows())
    train, val, test = (set(splits[name]) for name in ["train", "val", "test"])
    assert not train & val and not train & test and not val & test


def test_every_row_lands_in_exactly_one_split():
    df = synthetic_rows()
    splits = make_splits(df)
    split_of = {pid: name for name, ids in splits.items() for pid in ids}
    assert df["prompt_id"].map(split_of).notna().all()
    assert sum(len(ids) for ids in splits.values()) == df["prompt_id"].nunique()


def test_split_sizes_are_70_15_15_within_each_domain():
    df = synthetic_rows(n_prompts=400)
    splits = make_splits(df)
    prompts = df.drop_duplicates("prompt_id").set_index("prompt_id")["domain"]
    for domain in prompts.unique():
        n = (prompts == domain).sum()
        for name, share in [("val", 0.15), ("test", 0.15)]:
            n_split = prompts.loc[splits[name]].eq(domain).sum()
            assert abs(n_split - round(n * share)) <= 1


def test_splits_are_deterministic_for_a_seed():
    df = synthetic_rows()
    assert make_splits(df, seed=448) == make_splits(df, seed=448)
    assert make_splits(df, seed=448) != make_splits(df, seed=1)


def test_check_splits_rejects_overlap():
    df = synthetic_rows(n_prompts=8)
    ids = sorted(df["prompt_id"].unique())
    bad = {"train": ids[:6], "val": ids[5:7], "test": ids[7:]}  # ids[5] is in train and val
    with pytest.raises(AssertionError):
        check_splits(bad, df)


def test_committed_splits_json_has_no_overlap():
    splits = json.loads((ROOT / "splits.json").read_text())
    train, val, test = (set(splits[name]) for name in ["train", "val", "test"])
    assert train and val and test
    assert not train & val and not train & test and not val & test

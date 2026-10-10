"""Statistics helpers: fast accuracy / macro-F1, grouped bootstrap, McNemar, binomial test.

Labels are integers 0..K-1 (index into FAMILIES).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

N_BOOT = 2000
BOOT_SEED = 448


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Build a k x k confusion matrix (rows = true class, columns = predicted class).

    Args:
        y_true: true labels, integers 0..k-1.
        y_pred: predicted labels, integers 0..k-1, same length as y_true.
        k: number of classes.
        weights: optional count per row; a weight of 2 counts that row twice (used by the bootstrap).

    Returns:
        Array of shape (k, k). Entry [i, j] is the (weighted) number of rows with true class i
        predicted as class j.
    """
    return np.bincount(y_true * k + y_pred, weights=weights, minlength=k * k).reshape(k, k)


def acc_and_macro_f1(cm: np.ndarray) -> tuple[float, float]:
    """Accuracy and macro-F1 from a confusion matrix. Same as sklearn with zero_division=0.

    Args:
        cm: k x k confusion matrix from `confusion`.

    Returns:
        (accuracy, macro_f1). A class that never appears and is never predicted gets F1 = 0.
    """
    tp = np.diag(cm)
    acc = tp.sum() / cm.sum()
    # F1 = 2TP / (2TP + FP + FN) = 2TP / (true count + predicted count)
    denominator = cm.sum(axis=1) + cm.sum(axis=0)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(tp)), where=denominator > 0)
    return acc, f1.mean()


def bootstrap(y_true: np.ndarray, preds_by_seed: Sequence[np.ndarray], groups: Sequence, k: int,
              n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """Grouped bootstrap of (accuracy, macro-F1), averaged over seeds.

    Each resample draws prompt_ids (groups) with replacement, and every row of a drawn prompt
    comes along, so rows that share a prompt are never split up. The same `seed` gives the same
    resamples for every method, so two methods' arrays can be subtracted to get a paired bootstrap
    of their difference.

    Args:
        y_true: true labels of the test rows, integers 0..k-1.
        preds_by_seed: one prediction array per training seed (each aligned with y_true); the
            score of a resample is the mean over seeds.
        groups: group id of each test row (the prompt_id), same length as y_true.
        k: number of classes.
        n_boot: number of resamples.
        seed: random seed of the resampling.

    Returns:
        Array of shape (n_boot, 2): column 0 is accuracy, column 1 is macro-F1.
    """
    codes, _ = pd.factorize(pd.Series(groups))
    n_groups = codes.max() + 1
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, 2))
    for b in range(n_boot):
        times_drawn = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        weights = times_drawn[codes]  # how often each row appears in this resample
        scores = [acc_and_macro_f1(confusion(y_true, y_pred, k, weights)) for y_pred in preds_by_seed]
        out[b] = np.mean(scores, axis=0)
    return out


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile confidence interval of bootstrap samples.

    Args:
        samples: 1-D array of bootstrap values (e.g. one column of `bootstrap`).
        level: confidence level; 0.95 gives the 2.5th and 97.5th percentiles.

    Returns:
        (low, high).
    """
    tail = (1 - level) / 2 * 100
    return np.percentile(samples, tail), np.percentile(samples, 100 - tail)


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test: do A and B make different numbers of errors on the same rows?

    Only rows where exactly one model is right matter. Under "no difference", each of those rows
    is a fair coin flip between A and B, so the p-value is an exact two-sided binomial test.

    Args:
        y_true: true labels.
        pred_a: predictions of model A on the same rows.
        pred_b: predictions of model B on the same rows.

    Returns:
        {"only_a_right": rows only A got right, "only_b_right": rows only B got right,
         "p_value": two-sided exact p-value (1.0 if the models never disagree in correctness)}.
    """
    a_right, b_right = pred_a == y_true, pred_b == y_true
    only_a = int((a_right & ~b_right).sum())  # A right, B wrong
    only_b = int((~a_right & b_right).sum())  # A wrong, B right
    # Under "no difference", each disagreement is a coin flip between A and B.
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b > 0 else 1.0
    return {"only_a_right": only_a, "only_b_right": only_b, "p_value": p}


def binomial_vs_chance(n_correct: int, n: int, p0: float) -> dict[str, float]:
    """Exact two-sided binomial test of accuracy against chance p0, with a Clopper-Pearson 95% CI.

    Args:
        n_correct: number of correct predictions.
        n: number of predictions.
        p0: chance accuracy (0.2 for five balanced classes).

    Returns:
        {"n_correct", "n", "acc": n_correct / n, "ci_low", "ci_high" (95% Clopper-Pearson), "p_value"}.
    """
    result = binomtest(n_correct, n, p0)
    low, high = result.proportion_ci(0.95)
    return {"n_correct": n_correct, "n": n, "acc": n_correct / n, "ci_low": low, "ci_high": high,
            "p_value": result.pvalue}

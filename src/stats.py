"""Statistics helpers: fast accuracy / macro-F1, grouped bootstrap, McNemar, binomial test.

Labels are integers 0..K-1 (index into FAMILIES).
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest

N_BOOT = 2000
BOOT_SEED = 448


def confusion(y_true, y_pred, k, weights=None):
    """k x k confusion matrix (rows = true, cols = predicted); weights count a row several times."""
    return np.bincount(y_true * k + y_pred, weights=weights, minlength=k * k).reshape(k, k)


def acc_and_macro_f1(cm):
    """Accuracy and macro-F1 from a confusion matrix. Same as sklearn with zero_division=0."""
    tp = np.diag(cm)
    acc = tp.sum() / cm.sum()
    # F1 = 2TP / (2TP + FP + FN) = 2TP / (true count + predicted count)
    denominator = cm.sum(axis=1) + cm.sum(axis=0)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(tp)), where=denominator > 0)
    return acc, f1.mean()


def bootstrap(y_true, preds_by_seed, groups, k, n_boot=N_BOOT, seed=BOOT_SEED):
    """Grouped bootstrap of (accuracy, macro-F1), averaged over seeds.

    Each resample draws prompt_ids (groups) with replacement, and every row of a drawn prompt
    comes along, so rows that share a prompt are never split up. Returns an array (n_boot, 2).
    The same `seed` gives the same resamples for every method, so two methods' arrays can be
    subtracted to get a paired bootstrap of their difference.
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


def ci(samples, level=0.95):
    tail = (1 - level) / 2 * 100
    return np.percentile(samples, tail), np.percentile(samples, 100 - tail)


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test: do A and B make different numbers of errors on the same rows?"""
    a_right, b_right = pred_a == y_true, pred_b == y_true
    only_a = int((a_right & ~b_right).sum())  # A right, B wrong
    only_b = int((~a_right & b_right).sum())  # A wrong, B right
    # Under "no difference", each disagreement is a coin flip between A and B.
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b > 0 else 1.0
    return {"only_a_right": only_a, "only_b_right": only_b, "p_value": p}


def binomial_vs_chance(n_correct, n, p0):
    """Exact two-sided binomial test of accuracy against chance p0, with a Clopper-Pearson 95% CI."""
    result = binomtest(n_correct, n, p0)
    low, high = result.proportion_ci(0.95)
    return {"n_correct": n_correct, "n": n, "acc": n_correct / n, "ci_low": low, "ci_high": high,
            "p_value": result.pvalue}

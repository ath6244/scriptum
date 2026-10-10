"""Tests for the metrics and statistical tests in src/stats.py."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.stats import acc_and_macro_f1, binomial_vs_chance, bootstrap, ci, confusion, mcnemar

# A tiny example computed by hand (3 classes, 5 rows):
#   true: 0 0 1 1 2
#   pred: 0 1 1 1 0
# confusion (rows = true): [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
# accuracy = 3/5 = 0.6
# F1_0 = 2*1/(2+2) = 0.5, F1_1 = 2*2/(2+3) = 0.8, F1_2 = 0 (never predicted) -> macro = 1.3/3
Y_TRUE = np.array([0, 0, 1, 1, 2])
Y_PRED = np.array([0, 1, 1, 1, 0])


def test_confusion_matrix_by_hand():
    expected = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 0]])
    assert (confusion(Y_TRUE, Y_PRED, 3) == expected).all()


def test_accuracy_and_macro_f1_by_hand():
    acc, macro_f1 = acc_and_macro_f1(confusion(Y_TRUE, Y_PRED, 3))
    assert acc == pytest.approx(0.6)
    assert macro_f1 == pytest.approx(1.3 / 3)


def test_perfect_and_all_wrong_predictions():
    y = np.array([0, 1, 2, 3, 4])
    assert acc_and_macro_f1(confusion(y, y, 5)) == pytest.approx((1.0, 1.0))
    assert acc_and_macro_f1(confusion(y, (y + 1) % 5, 5)) == pytest.approx((0.0, 0.0))


def test_matches_sklearn_on_random_labels():
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.integers(0, 5, 500), rng.integers(0, 5, 500)
    acc, macro_f1 = acc_and_macro_f1(confusion(y_true, y_pred, 5))
    assert acc == pytest.approx(accuracy_score(y_true, y_pred))
    assert macro_f1 == pytest.approx(f1_score(y_true, y_pred, average="macro", labels=range(5), zero_division=0))


def test_weights_count_rows_several_times():
    weights = np.array([2, 1, 1, 1, 3])
    repeated = np.repeat(np.arange(5), weights)
    assert (confusion(Y_TRUE, Y_PRED, 3, weights) == confusion(Y_TRUE[repeated], Y_PRED[repeated], 3)).all()


# ---------------------------------------------------------------------------
# Grouped bootstrap
# ---------------------------------------------------------------------------
def make_test_set(n_prompts=200, seed=0):
    """Two rows per prompt; predictions are right about 70% of the time."""
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"p{i}" for i in range(n_prompts)], 2)
    y_true = rng.integers(0, 5, len(groups))
    y_pred = np.where(rng.random(len(groups)) < 0.7, y_true, rng.integers(0, 5, len(groups)))
    return y_true, y_pred, groups


def test_bootstrap_ci_contains_the_point_estimate():
    y_true, y_pred, groups = make_test_set()
    point = acc_and_macro_f1(confusion(y_true, y_pred, 5))
    samples = bootstrap(y_true, [y_pred], groups, 5, n_boot=500)
    assert samples.shape == (500, 2)
    for column, value in enumerate(point):
        low, high = ci(samples[:, column])
        assert low <= value <= high
        assert high - low > 0  # resampling really varies the score


def test_bootstrap_is_deterministic_and_paired():
    y_true, y_pred, groups = make_test_set()
    a = bootstrap(y_true, [y_pred], groups, 5, n_boot=200, seed=448)
    b = bootstrap(y_true, [y_pred], groups, 5, n_boot=200, seed=448)
    assert (a == b).all()  # same seed -> same resamples, so differences between methods are paired
    assert not (a == bootstrap(y_true, [y_pred], groups, 5, n_boot=200, seed=1)).all()


def test_bootstrap_averages_over_seeds():
    y_true, y_pred, groups = make_test_set()
    perfect = bootstrap(y_true, [y_true], groups, 5, n_boot=50)
    mixed = bootstrap(y_true, [y_true, y_pred], groups, 5, n_boot=50)
    alone = bootstrap(y_true, [y_pred], groups, 5, n_boot=50)
    assert mixed == pytest.approx((perfect + alone) / 2)


def test_ci_percentiles():
    low, high = ci(np.arange(1001))  # 0..1000
    assert low == pytest.approx(25) and high == pytest.approx(975)


# ---------------------------------------------------------------------------
# McNemar and binomial tests
# ---------------------------------------------------------------------------
def test_mcnemar_counts_disagreements():
    y = np.array([0, 0, 0, 0, 0, 0])
    a = np.array([0, 0, 0, 1, 1, 0])  # right on rows 0, 1, 2, 5
    b = np.array([0, 1, 1, 0, 1, 1])  # right on rows 0, 3
    result = mcnemar(y, a, b)
    assert result["only_a_right"] == 3 and result["only_b_right"] == 1
    assert 0 < result["p_value"] <= 1


def test_mcnemar_identical_models_give_p_1():
    y = np.array([0, 1, 2, 3])
    assert mcnemar(y, y, y)["p_value"] == 1.0


def test_binomial_at_chance_is_not_significant():
    result = binomial_vs_chance(20, 100, 0.2)
    assert result["acc"] == pytest.approx(0.2)
    assert result["p_value"] > 0.9
    assert result["ci_low"] < 0.2 < result["ci_high"]


def test_binomial_far_above_chance_is_significant():
    result = binomial_vs_chance(60, 100, 0.2)
    assert result["p_value"] < 1e-10
    assert result["ci_low"] > 0.2

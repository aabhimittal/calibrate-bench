import numpy as np
import pytest

from cbench.metrics import aurc, bootstrap_ci, brier, ece, risk_coverage

# Toy case, binary. conf = max prob, pred = argmax.
#   i  probs        label pred conf correct
#   0  (.90, .10)   0     0    .90  1
#   1  (.80, .20)   0     0    .80  1
#   2  (.70, .30)   1     0    .70  0
#   3  (.40, .60)   1     1    .60  1
#   4  (.45, .55)   0     1    .55  0
#   5  (.35, .65)   1     1    .65  1
PROBS = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6], [0.45, 0.55], [0.35, 0.65]])
LABELS = np.array([0, 0, 1, 1, 0, 1])


def test_ece_hand_computed():
    # 2 equal-mass bins by ascending conf:
    #   {.55, .60, .65}: conf 0.60, acc 2/3 -> gap 1/15
    #   {.70, .80, .90}: conf 0.80, acc 2/3 -> gap 2/15
    # ECE = 1/2 * 1/15 + 1/2 * 2/15 = 0.1
    assert ece(PROBS, LABELS, n_bins=2) == pytest.approx(0.1)


def test_ece_one_bin_is_global_gap_and_more_bins_than_items_is_per_item():
    # 1 bin: |mean acc - mean conf| = |4/6 - 4.2/6| = 0.2/6
    assert ece(PROBS, LABELS, n_bins=1) == pytest.approx(0.2 / 6)
    # 1 item per bin (extra bins empty): mean |correct - conf|
    # = (.1 + .2 + .7 + .4 + .55 + .35) / 6 = 2.3/6
    assert ece(PROBS, LABELS, n_bins=50) == pytest.approx(2.3 / 6)


def test_ece_perfect_calibration_is_zero():
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])
    # last two: conf .5, pred 0, labels 0 and 1 -> acc .5 in that bin
    assert ece(probs, [0, 1, 0, 1], n_bins=2) == pytest.approx(0.0)


def test_brier_hand_computed():
    # per item sum_k (p_k - y_k)^2: .02, .08, .98, .32, .605, .245 -> sum 2.25, mean .375
    assert brier(PROBS, LABELS) == pytest.approx(0.375)


def test_brier_multiclass():
    # (.5,.3,.2) label 0: .25+.09+.04 = .38 ; (0,0,1) label 2: 0 -> mean .19
    assert brier([[0.5, 0.3, 0.2], [0, 0, 1]], [0, 2]) == pytest.approx(0.19)


def test_aurc_hand_computed():
    # Descending conf: .90 .80 .70 .65 .60 .55 ; errors 0 0 1 0 0 1
    # risk@k = cum errors / k: 0, 0, 1/3, 1/4, 1/5, 2/6
    # AURC = mean = (1/3 + 1/4 + 1/5 + 1/3) / 6 = 67/360
    coverage, risk = risk_coverage(PROBS, LABELS)
    np.testing.assert_allclose(coverage, np.arange(1, 7) / 6)
    np.testing.assert_allclose(risk, [0, 0, 1 / 3, 1 / 4, 1 / 5, 1 / 3])
    assert aurc(PROBS, LABELS) == pytest.approx(67 / 360)


def test_aurc_ties_do_not_depend_on_input_order():
    # Two tie groups of two. Accepting the .9 group: 1 error / 2 = .5; accepting all: 2/4 = .5.
    # AURC = .5 * .5 + .5 * .5 = .5. An order-dependent version gives .583 or .333.
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.6, 0.4]])
    labels = np.array([1, 0, 0, 1])
    assert aurc(probs, labels) == pytest.approx(0.5)
    assert aurc(probs[::-1], labels[::-1]) == pytest.approx(0.5)
    coverage, risk = risk_coverage(probs, labels)
    np.testing.assert_allclose(coverage, [0.5, 1.0])


def test_bootstrap_ci_brackets_point_and_is_seeded():
    rng = np.random.default_rng(1)
    p1 = rng.uniform(0, 1, 400)
    probs = np.c_[1 - p1, p1]
    labels = (rng.uniform(0, 1, 400) < p1).astype(int)
    point, lo, hi = bootstrap_ci(brier, probs, labels, n_resamples=200)
    assert lo <= point <= hi and hi - lo > 0
    assert bootstrap_ci(brier, probs, labels, n_resamples=200) == (point, lo, hi)


def test_bootstrap_ci_degenerate_has_zero_width():
    probs = np.tile([1.0, 0.0], (10, 1))
    assert bootstrap_ci(ece, probs, np.zeros(10, int), n_resamples=50) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "probs, labels",
    [
        ([[0.5, 0.6]], [0]),  # doesn't sum to 1
        ([[1.2, -0.2]], [0]),  # negative
        ([[np.nan, 1.0]], [0]),  # non-finite
        ([[0.5, 0.5]], [2]),  # label out of range
        ([[0.5, 0.5]], [0.0]),  # float label
        ([0.5, 0.5], [0]),  # 1-D probs
        (np.empty((0, 2)), np.empty(0, int)),  # empty
    ],
)
def test_rejects_malformed_predictions(probs, labels):
    with pytest.raises(ValueError):
        brier(probs, labels)

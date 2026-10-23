"""Calibration and selective-prediction metrics (M2).

All functions take probs of shape (n, K) and integer labels of shape (n,).
Confidence is the max probability and the prediction is its argmax (ties -> lowest index).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _check(probs, labels):
    """Trust boundary: predictions arrive from submitted parquet files."""
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels)
    if probs.ndim != 2 or len(probs) == 0 or labels.shape != (len(probs),):
        raise ValueError(f"need probs (n, K) and labels (n,), got {probs.shape} and {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() >= probs.shape[1]:
        raise ValueError("labels must be integers in [0, K)")
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=1e-6):
        raise ValueError("each row of probs must be a finite, non-negative distribution summing to 1")
    return probs, labels


def _confidence(probs, labels):
    probs, labels = _check(probs, labels)
    return probs.max(1), (probs.argmax(1) == labels).astype(float)


def ece(probs, labels, n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins: sum_m |B_m|/n * |acc(B_m) - conf(B_m)|."""
    conf, correct = _confidence(probs, labels)
    # shortcut: equal confidences straddling a bin edge are split by input order; fine unless ties are massive.
    bins = np.array_split(np.argsort(conf, kind="stable"), n_bins)
    return float(sum(len(b) / len(conf) * abs(correct[b].mean() - conf[b].mean()) for b in bins if len(b)))


def brier(probs, labels) -> float:
    """Multiclass Brier score: mean over examples of sum_k (p_k - y_k)^2. Range [0, 2]."""
    probs, labels = _check(probs, labels)
    return float(((probs - np.eye(probs.shape[1])[labels]) ** 2).sum(1).mean())


def risk_coverage(probs, labels) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and selective risk (error rate of accepted items) at each distinct confidence threshold.

    Tied confidences are accepted or rejected together, so the curve doesn't depend on input order.
    """
    conf, correct = _confidence(probs, labels)
    order = np.argsort(-conf, kind="stable")
    c, err = conf[order], 1 - correct[order]
    ends = np.flatnonzero(np.r_[c[1:] != c[:-1], True])  # last index of each tie group
    accepted = ends + 1
    return accepted / len(c), np.cumsum(err)[ends] / accepted


def aurc(probs, labels) -> float:
    """Area under the risk-coverage curve (step integral, lower is better)."""
    coverage, risk = risk_coverage(probs, labels)
    return float(np.sum(np.diff(coverage, prepend=0) * risk))


def bootstrap_ci(
    metric: Callable, probs, labels, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) via the percentile bootstrap over examples."""
    probs, labels = _check(probs, labels)
    idx = np.random.default_rng(seed).integers(0, len(labels), size=(n_resamples, len(labels)))
    stats = [metric(probs[i], labels[i]) for i in idx]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return metric(probs, labels), float(lo), float(hi)

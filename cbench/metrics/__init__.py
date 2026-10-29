"""Calibration and selective-prediction metrics (M2).

All functions take probs of shape (n, K) and labels that are either integers of shape (n,)
or soft targets of shape (n, K) (e.g. the share of raters picking each option).
Confidence is the max probability and the prediction is its argmax (ties -> lowest index).
"Correct" is the target mass on the prediction: 0/1 for hard labels, rater agreement for soft ones.
Soft-target scores have a non-zero floor (rater disagreement); never rank them against hard-target scores.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _is_distribution(a):
    return np.isfinite(a).all() and (a >= 0).all() and np.allclose(a.sum(1), 1, atol=1e-6)


def _check(probs, labels):
    """Trust boundary: predictions arrive from submitted parquet files. Returns (probs, targets (n, K))."""
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels)
    if probs.ndim != 2 or len(probs) == 0 or labels.shape not in ((len(probs),), probs.shape):
        raise ValueError(f"need probs (n, K) and labels (n,) or (n, K), got {probs.shape} and {labels.shape}")
    if not _is_distribution(probs):
        raise ValueError("each row of probs must be a finite, non-negative distribution summing to 1")
    if labels.ndim == 2:
        if not _is_distribution(labels.astype(float)):
            raise ValueError("each row of soft labels must be a finite, non-negative distribution summing to 1")
        return probs, labels.astype(float)
    if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() >= probs.shape[1]:
        raise ValueError("labels must be integers in [0, K)")
    return probs, np.eye(probs.shape[1])[labels]


def _confidence(probs, labels):
    probs, targets = _check(probs, labels)
    return probs.max(1), targets[np.arange(len(probs)), probs.argmax(1)]


def reliability(probs, labels, n_bins: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability diagram data over equal-mass bins: mean confidence, accuracy and size of each non-empty bin.

    Tied confidences always share a bin (the bin of their group's first rank). Splitting them by input order
    would bin arbitrary subsets: a constant predictor would get a non-zero ECE from sampling noise alone.
    """
    conf, correct = _confidence(probs, labels)
    order = np.argsort(conf, kind="stable")
    first = np.searchsorted(conf[order], conf[order], side="left")  # rank where each item's tie group starts
    which = first * n_bins // len(conf)
    bins = [order[which == m] for m in range(n_bins) if (which == m).any()]
    return (np.array([conf[b].mean() for b in bins]), np.array([correct[b].mean() for b in bins]),
            np.array([len(b) for b in bins]))


def ece(probs, labels, n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins: sum_m |B_m|/n * |acc(B_m) - conf(B_m)|."""
    conf, acc, size = reliability(probs, labels, n_bins)
    return float((size / size.sum() * np.abs(acc - conf)).sum())


def brier(probs, labels) -> float:
    """Multiclass Brier score: mean over examples of sum_k (p_k - y_k)^2. Range [0, 2]."""
    probs, targets = _check(probs, labels)
    return float(((probs - targets) ** 2).sum(1).mean())


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


def flips(p, q) -> np.ndarray:
    """Per row: 1 if the predicted option differs between two distributions over the same options."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return (p.argmax(1) != q.argmax(1)).astype(float)


def kl(p, q, eps: float = 1e-6) -> np.ndarray:
    """Per row KL(p || q) in nats; both clipped at eps so a hard zero in q can't make it infinite."""
    p, q = (np.clip(np.asarray(a, dtype=float), eps, None) for a in (p, q))
    p, q = p / p.sum(1, keepdims=True), q / q.sum(1, keepdims=True)
    return (p * np.log(p / q)).sum(1)


def bootstrap_ci(
    metric: Callable, probs, labels, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) via the percentile bootstrap over examples."""
    probs, targets = _check(probs, labels)
    idx = np.random.default_rng(seed).integers(0, len(targets), size=(n_resamples, len(targets)))
    stats = [metric(probs[i], targets[i]) for i in idx]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return metric(probs, targets), float(lo), float(hi)

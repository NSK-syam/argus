"""Deterministic metrics for RUL regression, used by the trust gate.

Nothing in this module is LLM-derived — every number here is reproducible
from the predictions and ground truth alone, which is the point: model
promotion is decided by these functions, never by an LLM's opinion.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def nasa_score(y_true, y_pred) -> float:
    """The PHM08 / C-MAPSS asymmetric scoring function.

    Penalizes *late* predictions (predicting more RUL than actually
    remains — i.e. under-warning) far more heavily than early ones, which
    matches the real cost asymmetry of a maintenance system: a missed
    failure is worse than an unnecessary inspection.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    early = d < 0
    late = ~early
    score = np.empty_like(d)
    score[early] = np.exp(-d[early] / 13.0) - 1.0
    score[late] = np.exp(d[late] / 10.0) - 1.0
    return float(score.sum())


def warning_f1(y_true, y_pred, threshold: int = 30) -> float:
    """F1 for the binary "needs attention soon" call: RUL <= threshold."""
    y_true_bin = (np.asarray(y_true) <= threshold).astype(int)
    y_pred_bin = (np.asarray(y_pred) <= threshold).astype(int)
    if y_true_bin.sum() == 0 and y_pred_bin.sum() == 0:
        return 1.0
    return float(f1_score(y_true_bin, y_pred_bin, zero_division=0))


def full_report(y_true, y_pred, threshold: int = 30) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "f1_at_threshold": warning_f1(y_true, y_pred, threshold),
        "n": int(len(y_true)),
    }

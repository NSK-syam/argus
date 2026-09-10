"""Model training utilities: mean-RUL baseline, Random Forest, XGBoost,
grouped (by-engine) cross-validation, and split-conformal interval
calibration.

Validation is always grouped by ``unit_number`` — rows from the same
engine never appear in both the train and validation side of a fold.
Randomly splitting rows would let the model see near-duplicate cycles
from the same engine trajectory on both sides, which silently inflates
validation scores without meaning anything (a leakage failure mode that's
easy to miss and would make the trust gate meaningless).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - exercised only if xgboost is absent
    XGBRegressor = None

from .evaluate import full_report

N_SPLITS = 5
CONFORMAL_ALPHA = 0.10  # targets ~90% coverage; trust gate requires >=85%
RANDOM_STATE = 42


@dataclass
class AttemptResult:
    attempt_number: int
    feature_spec: dict
    model_family: str
    hyperparams: dict
    validation_metrics: dict
    test_metrics: dict
    conformal_q: float
    conformal_coverage: float
    n_features: int
    notes: list[str] = field(default_factory=list)


def mean_rul_baseline_predictions(train_rul: pd.Series, n: int) -> np.ndarray:
    """The dumbest defensible baseline: predict the training-set mean RUL
    for every row, regardless of sensor readings. Everything else must
    beat this by a meaningful margin (trust gate: >=15% RMSE improvement)."""
    return np.full(n, fill_value=float(train_rul.mean()))


def _build_model(model_family: str, hyperparams: dict):
    if model_family == "linear_regression":
        return LinearRegression(**hyperparams)
    if model_family == "random_forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **hyperparams)
    if model_family == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed")
        return XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror",
            **hyperparams,
        )
    raise ValueError(f"unsupported model_family: {model_family}")


def grouped_cross_val_predict(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_family: str,
    hyperparams: dict,
    n_splits: int = N_SPLITS,
) -> np.ndarray:
    """Out-of-fold predictions using GroupKFold by engine. These act both
    as the validation signal and as the calibration set for split-conformal
    intervals (a cross-conformal approximation — documented limitation,
    fine for a prototype, worth swapping for a dedicated calibration split
    once more engines are available)."""
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    for train_idx, val_idx in gkf.split(X, y, groups):
        model = _build_model(model_family, hyperparams)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = model.predict(X.iloc[val_idx])
    return oof


def fit_final_model(X: pd.DataFrame, y: pd.Series, model_family: str, hyperparams: dict):
    model = _build_model(model_family, hyperparams)
    model.fit(X, y)
    return model


def conformal_interval(residuals: np.ndarray, alpha: float = CONFORMAL_ALPHA) -> float:
    """Split-conformal half-width: the (1-alpha) quantile of |residual|
    from the calibration set. Interval = prediction +/- q."""
    abs_res = np.abs(residuals[~np.isnan(residuals)])
    return float(np.quantile(abs_res, 1 - alpha))


def coverage(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    within = np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= q
    return float(np.mean(within))


def run_attempt(
    attempt_number: int,
    feature_spec: dict,
    model_family: str,
    hyperparams: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    X_test_last_cycle: pd.DataFrame,
    y_test_true: pd.Series,
) -> tuple[AttemptResult, object]:
    oof_pred = grouped_cross_val_predict(X_train, y_train, groups_train, model_family, hyperparams)
    oof_pred = np.clip(oof_pred, 0, None)
    val_metrics = full_report(y_train, oof_pred)

    q = conformal_interval(oof_pred - y_train.to_numpy())

    model = fit_final_model(X_train, y_train, model_family, hyperparams)
    test_pred = np.clip(model.predict(X_test_last_cycle), 0, None)
    test_metrics = full_report(y_test_true, test_pred)
    cov = coverage(y_test_true.to_numpy(), test_pred, q)

    result = AttemptResult(
        attempt_number=attempt_number,
        feature_spec=feature_spec,
        model_family=model_family,
        hyperparams=hyperparams,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        conformal_q=q,
        conformal_coverage=cov,
        n_features=X_train.shape[1],
    )
    return result, model


def bounded_random_search(
    model_family: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    max_trials: int = 6,
    seed: int = RANDOM_STATE,
) -> tuple[dict, float]:
    """A small bounded hyperparameter search (stand-in for the Optuna
    search named in the plan — same contract: propose params, evaluate by
    grouped CV RMSE, keep the best; swapping in optuna.create_study() here
    is a mechanical follow-up, not an architecture change).

    Capped at ``max_trials`` (<=8 per the plan's compute budget).
    """
    rng = np.random.RandomState(seed)
    if model_family == "random_forest":
        space = lambda: dict(
            n_estimators=int(rng.choice([150, 250, 350])),
            max_depth=int(rng.choice([6, 8, 10, 12])),
            min_samples_leaf=int(rng.choice([1, 2, 4])),
        )
    elif model_family == "xgboost":
        space = lambda: dict(
            n_estimators=int(rng.choice([150, 250, 350])),
            max_depth=int(rng.choice([3, 4, 5, 6])),
            learning_rate=float(rng.choice([0.03, 0.05, 0.08, 0.1])),
            subsample=float(rng.choice([0.7, 0.85, 1.0])),
        )
    else:
        raise ValueError(model_family)

    best_params, best_rmse = None, float("inf")
    for _ in range(max_trials):
        params = space()
        oof = grouped_cross_val_predict(X_train, y_train, groups_train, model_family, params)
        oof = np.clip(oof, 0, None)
        from .evaluate import rmse as rmse_fn

        score = rmse_fn(y_train, oof)
        if score < best_rmse:
            best_rmse, best_params = score, params
    return best_params, best_rmse

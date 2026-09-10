"""SHAP explanations, converted into plain-language sentences.

Per the build plan: Claude converts *verified* SHAP values into
maintenance language without inventing causes. That means SHAP does the
actual attribution (real numbers, computed deterministically from the
model), and any natural-language step downstream is only ever describing
numbers that were actually computed -- never generating an explanation
from scratch. The plain-language function here has no LLM in it at all;
it's a template over real SHAP output, which is deliberately the "verified
... without inventing" half of that requirement. A live-Claude polish pass
over this text is a legitimate later enhancement, but the numbers must
never depend on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def build_explainer(model, background: pd.DataFrame, model_family: str):
    """Tree models get the fast, exact TreeExplainer; linear regression
    gets shap.Explainer's linear path. Both are deterministic given the
    same model and background sample."""
    if model_family in ("random_forest", "xgboost"):
        return shap.TreeExplainer(model)
    return shap.Explainer(model, background)


def global_importance(explainer, background: pd.DataFrame, top_n: int = 8) -> list[dict]:
    """Mean |SHAP value| per feature across the background sample -- the
    same "global feature importance" the theme brief asks for, computed
    from real attributions rather than e.g. raw model.feature_importances_
    (which for a Random Forest doesn't account for correlated features the
    way SHAP does)."""
    shap_values = explainer(background)
    values = shap_values.values
    if values.ndim == 3:  # some explainers return (n, features, outputs)
        values = values[:, :, 0]
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    cols = list(background.columns)
    return [{"feature": cols[i], "mean_abs_shap": float(mean_abs[i])} for i in order]


def explain_prediction(explainer, row: pd.DataFrame, top_n: int = 3) -> dict:
    """Per-prediction explanation for a single row (1-row DataFrame with
    the model's exact feature columns). Returns the raw SHAP contributions
    (signed, real numbers) plus a plain-language sentence built only from
    those numbers."""
    shap_values = explainer(row)
    values = shap_values.values[0]
    if values.ndim == 2:
        values = values[:, 0]
    base_value = shap_values.base_values[0]
    base_value = float(base_value if np.isscalar(base_value) else base_value[0])

    cols = list(row.columns)
    contributions = sorted(
        zip(cols, values.tolist()), key=lambda kv: abs(kv[1]), reverse=True
    )[:top_n]

    return {
        "base_value": base_value,
        "contributions": [{"feature": f, "shap_value": float(v)} for f, v in contributions],
        "narration": _narrate(contributions),
    }


def _narrate(contributions: list[tuple[str, float]]) -> str:
    if not contributions:
        return "No dominant contributing sensors identified."
    parts = []
    for feature, value in contributions:
        direction = "pushing predicted RUL down (higher risk)" if value < 0 else "pushing predicted RUL up (lower risk)"
        parts.append(f"{feature} ({direction}, contribution {value:+.1f} cycles)")
    return "Top contributing signals: " + "; ".join(parts) + "."

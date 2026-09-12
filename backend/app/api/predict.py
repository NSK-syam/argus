from __future__ import annotations

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import models
from ..db.session import get_db
from ..ml.pipeline import explain

router = APIRouter(prefix="/api/v1", tags=["predict"])


class PredictRequest(BaseModel):
    model_version_id: str
    features: dict[str, float]


@router.post("/predict")
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    mv = db.get(models.ModelVersion, req.model_version_id)
    if mv is None:
        raise HTTPException(404, "model version not found")
    if mv.stage != "production":
        # Found in external code review: nothing previously stopped a shadow
        # (unpromoted, possibly trust-gate-failing) model from serving real
        # predictions -- only the frontend happened to always pass a
        # promoted id. Promotion is the trust gate's enforcement point; an
        # API that skips it isn't actually gated.
        raise HTTPException(403, "model version is not in production -- promote it first")

    missing = set(mv.feature_columns_json) - set(req.features.keys())
    if missing:
        raise HTTPException(422, f"missing required features: {sorted(missing)}")

    model = joblib.load(mv.artifact_path)
    row = pd.DataFrame([{c: req.features[c] for c in mv.feature_columns_json}])
    pred = float(max(0.0, model.predict(row)[0]))
    q = mv.conformal_q
    warning = pred <= 30

    explanation = None
    try:
        # a single ad hoc prediction has no stored background sample here,
        # so the request row doubles as its own background -- fine for a
        # tree explainer's exact SHAP values on one row.
        explainer = explain.build_explainer(model, row, model_family=mv.model_family)
        explanation = explain.explain_prediction(explainer, row)
    except Exception:
        explanation = None

    return {
        "predicted_rul": pred,
        "interval": [max(0.0, pred - q), pred + q],
        "warning": warning,
        "warning_threshold": 30,
        "model_stage": mv.stage,
        "explanation": explanation,
    }

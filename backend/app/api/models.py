from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import models
from ..db.session import get_db

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("/{model_version_id}")
def get_model_version(model_version_id: str, db: Session = Depends(get_db)):
    mv = db.get(models.ModelVersion, model_version_id)
    if mv is None:
        raise HTTPException(404, "model version not found")
    return _model_out(mv)


@router.post("/{model_version_id}/promote")
def promote_model(model_version_id: str, db: Session = Depends(get_db)):
    """One-click human-confirmed promotion. This is the ONLY way a model
    reaches 'production' -- passing the trust gate makes it eligible, it
    never auto-promotes itself. A failing model is refused outright."""
    mv = db.get(models.ModelVersion, model_version_id)
    if mv is None:
        raise HTTPException(404, "model version not found")
    if not mv.trust_gate_passed:
        raise HTTPException(
            409,
            "this model failed the trust gate and is not promotion_eligible; "
            "production exposure always requires a promotion_eligible model",
        )

    if mv.stage == "production":
        # Idempotent: re-promoting an already-production model is a no-op,
        # not a second Deployment row. Found in external code review: the
        # previous version created a fresh audit row and re-set the same
        # field on every repeated click.
        return _model_out(mv)

    # The frontend (runs/[runId]/page.tsx) assumes at most one production
    # model at a time (`.find(mv => mv.stage === "production")`), so
    # promotion is global, not scoped to this model's run: demote every
    # other currently-production ModelVersion in the same transaction.
    # Found in external code review: without this, two promotions from two
    # different runs could both read "production" simultaneously, which is
    # both an inconsistent state and silently wrong for any caller assuming
    # a single production model.
    other_production = (
        db.query(models.ModelVersion)
        .filter(models.ModelVersion.stage == "production", models.ModelVersion.id != mv.id)
        .all()
    )
    now = datetime.now(timezone.utc)
    for other in other_production:
        other.stage = "shadow"
        db.add(
            models.Deployment(
                model_version_id=other.id,
                stage="shadow",
                promoted_by="auto_demoted_on_new_promotion",
                audit_json={"demoted_at": now.isoformat(), "demoted_in_favor_of": mv.id},
            )
        )

    mv.stage = "production"
    deployment = models.Deployment(
        model_version_id=mv.id,
        stage="production",
        promoted_by="human_confirmed",
        audit_json={
            "promoted_at": now.isoformat(),
            "trust_gate_passed": mv.trust_gate_passed,
            "conformal_q": mv.conformal_q,
        },
    )
    db.add(deployment)
    db.commit()
    db.refresh(mv)
    return _model_out(mv)


def _model_out(mv: models.ModelVersion) -> dict:
    return {
        "id": mv.id,
        "run_id": mv.run_id,
        "attempt_number": mv.attempt_number,
        "model_family": mv.model_family,
        "feature_columns": mv.feature_columns_json,
        "conformal_q": mv.conformal_q,
        "shap_summary": mv.shap_summary_json,
        "trust_gate_passed": mv.trust_gate_passed,
        "stage": mv.stage,
        "created_at": mv.created_at.isoformat(),
    }

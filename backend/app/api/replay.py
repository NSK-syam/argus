from __future__ import annotations

import asyncio
import json

import joblib
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..core.config import settings
from ..db import models
from ..db.session import SessionLocal
from ..ml.data import cmapss
from ..ml.pipeline import explain as explain_mod

router = APIRouter(prefix="/api/v1/replay", tags=["replay"])


@router.get("/engines")
def list_replay_engines():
    """Which held-out FD001 test engines are available to replay."""
    test_df, _true_rul = cmapss.load_fd001_test(settings.data_dir)
    engines = sorted(int(u) for u in test_df["unit_number"].unique().tolist())
    return {"engines": engines, "count": len(engines)}


@router.get("/{model_version_id}/{engine_id}/events")
async def replay_engine(model_version_id: str, engine_id: int, speed: float = 8.0):
    """SSE stream: replay one real held-out FD001 test engine's full
    trajectory, cycle by cycle, through the given model version. Each
    event carries predicted RUL, the conformal interval, the warning
    flag, and a plain-language SHAP explanation of that cycle's
    prediction -- the "watch risk climb and get explained in real time"
    demo moment from the build plan, driven entirely by real held-out
    data, not a synthetic stream.

    ``speed`` is cycles-per-second (default 8 -- a ~30-cycle warning
    window plays in a few seconds, fast enough for a live demo).
    """
    db = SessionLocal()
    mv = db.get(models.ModelVersion, model_version_id)
    if mv is None:
        db.close()
        raise HTTPException(404, "model version not found")
    attempt = (
        db.query(models.Attempt)
        .filter(models.Attempt.run_id == mv.run_id, models.Attempt.attempt_number == mv.attempt_number)
        .first()
    )
    feature_columns = list(mv.feature_columns_json)
    model_family = mv.model_family
    conformal_q = mv.conformal_q
    artifact_path = mv.artifact_path
    feature_spec = (attempt.plan_json or {}).get("feature_spec", {}) if attempt else {}
    db.close()

    test_df, true_rul = cmapss.load_fd001_test(settings.data_dir)
    if engine_id not in set(int(u) for u in test_df["unit_number"].unique()):
        raise HTTPException(404, f"engine {engine_id} not in the FD001 test set")

    async def event_stream():
        model = joblib.load(artifact_path)

        windows = tuple(feature_spec.get("windows", ()))
        lags = tuple(feature_spec.get("lags", ()))
        base_sensor_cols = sorted(
            {c.split("_roll_")[0].split("_lag_")[0] for c in feature_columns if c.startswith("sensor_")}
        )

        engineered = cmapss.engineer_features(test_df, base_sensor_cols, windows=windows, lags=lags)
        engine_rows = engineered[engineered["unit_number"] == engine_id].sort_values("time_cycles")
        last_observed_cycle = int(engine_rows["time_cycles"].max())
        true_rul_at_last_cycle = float(true_rul.loc[engine_id]) if engine_id in true_rul.index else None

        explainer = None
        try:
            bg = engine_rows[feature_columns].sample(n=min(20, len(engine_rows)), random_state=42)
            explainer = explain_mod.build_explainer(model, bg, model_family)
        except Exception:
            explainer = None

        for _, cycle_row in engine_rows.iterrows():
            X = cycle_row[feature_columns].to_frame().T
            pred = float(max(0.0, model.predict(X)[0]))
            warning = pred <= 30

            narration = None
            if explainer is not None:
                try:
                    narration = explain_mod.explain_prediction(explainer, X)["narration"]
                except Exception:
                    narration = None

            true_rul_at_cycle = (
                true_rul_at_last_cycle + (last_observed_cycle - int(cycle_row["time_cycles"]))
                if true_rul_at_last_cycle is not None
                else None
            )

            yield _sse(
                "cycle",
                {
                    "engine_id": engine_id,
                    "time_cycles": int(cycle_row["time_cycles"]),
                    "predicted_rul": pred,
                    "interval": [max(0.0, pred - conformal_q), pred + conformal_q],
                    "true_rul_estimate": true_rul_at_cycle,
                    "warning": warning,
                    "narration": narration,
                },
            )
            await asyncio.sleep(1.0 / max(speed, 0.1))

        yield _sse("done", {"engine_id": engine_id, "true_rul_at_last_cycle": true_rul_at_last_cycle})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"

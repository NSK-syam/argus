from pathlib import Path

from app.ml.data.baseline import AttemptResult
from app.ml.pipeline.mlflow_logging import log_attempt
from app.ml.pipeline.orchestrator import LoggedAttempt
from app.ml.pipeline.trust_gate import GateCheck, TrustGateResult


def _fake_logged_attempt() -> LoggedAttempt:
    result = AttemptResult(
        attempt_number=1,
        feature_spec={"windows": (), "lags": ()},
        model_family="random_forest",
        hyperparams={"n_estimators": 100, "max_depth": 6},
        validation_metrics={"rmse": 20.0, "mae": 15.0, "nasa_score": 500.0, "f1_at_threshold": 0.7, "n": 100},
        test_metrics={"rmse": 18.0, "mae": 13.0, "nasa_score": 450.0, "f1_at_threshold": 0.75, "n": 100},
        conformal_q=10.0,
        conformal_coverage=0.9,
        n_features=10,
        feature_columns=[f"f{i}" for i in range(10)],
        background_sample=[],
    )
    gate = TrustGateResult(
        passed=True,
        checks=[GateCheck(name="beats_baseline_by_15pct", passed=True, detail="ok")],
    )
    return LoggedAttempt(
        result=result,
        gate=gate,
        revision_action="initial_plan",
        revision_rationale="test",
        plan_source="deterministic_fallback",
    )


def test_log_attempt_returns_a_run_id(tmp_path: Path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow_run_id = log_attempt(tracking_uri, run_id="test-run-1", logged=_fake_logged_attempt())
    assert mlflow_run_id is not None
    assert isinstance(mlflow_run_id, str)


def test_log_attempt_never_raises_even_on_bad_tracking_uri():
    # an invalid tracking URI must degrade to None, not crash the caller
    # -- an MLflow outage must never take down a real pipeline run
    result = log_attempt("not-a-valid-mlflow-uri://nope", run_id="x", logged=_fake_logged_attempt())
    assert result is None

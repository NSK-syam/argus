"""MLflow logging for the reflection loop: every attempt becomes a run,
not just the winner -- this is what "preserve every attempt in MLflow and
the decision journal" (build plan, agentic workflow step 6) means in
practice, and it's what makes the retry visible in MLflow's UI, not just
in the app's own attempt timeline.

Deliberately NOT imported by app/ml/pipeline/orchestrator.py -- the ML
core stays free of any tracking-system dependency and its tests stay fast
and hermetic. This module is called from the run service's on_attempt
hook instead, same pattern as DB persistence and the demo bundle.
"""

from __future__ import annotations

import logging

import mlflow

from ..pipeline.orchestrator import LoggedAttempt

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "argus-fd001-rul"


def _ensure_experiment(tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_attempt(tracking_uri: str, run_id: str, logged: LoggedAttempt) -> str | None:
    """Log one attempt as an MLflow run. Returns the MLflow run id, or
    None if logging failed -- this must never block or fail the pipeline
    run itself; an MLflow outage is not a reason to lose a real result."""
    try:
        _ensure_experiment(tracking_uri)
        r = logged.result
        with mlflow.start_run(
            run_name=f"{run_id}-attempt-{r.attempt_number}"
        ) as mlflow_run:
            mlflow.set_tags(
                {
                    "argus_run_id": run_id,
                    "attempt_number": r.attempt_number,
                    "plan_source": logged.plan_source,
                    "revision_action": logged.revision_action,
                    "trust_gate_passed": logged.gate.passed,
                }
            )
            mlflow.log_param("model_family", r.model_family)
            for k, v in r.hyperparams.items():
                mlflow.log_param(f"hp_{k}", v)
            mlflow.log_param("n_features", r.n_features)
            mlflow.log_param("windows", str(r.feature_spec.get("windows")))
            mlflow.log_param("lags", str(r.feature_spec.get("lags")))

            for k, v in r.validation_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in r.test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            mlflow.log_metric("conformal_coverage", r.conformal_coverage)
            mlflow.log_metric("conformal_q", r.conformal_q)

            for check in logged.gate.checks:
                mlflow.log_metric(f"gate_pass_{check.name}", 1.0 if check.passed else 0.0)

            return mlflow_run.info.run_id
    except Exception as exc:  # noqa: BLE001 - tracking failures must never break a real result
        logger.warning("MLflow logging failed for attempt %d: %s", logged.result.attempt_number, exc)
        return None

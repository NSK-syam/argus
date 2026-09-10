"""The Planner -> Trainer -> Critic reflection loop.

This is the single most important module in the prototype: the concrete,
runnable proof that the pipeline can notice a weak result and change its
own approach, rather than running once and reporting whatever it got.

Two planners exist behind one interface:

  - ``_next_plan``: a **deterministic fallback sequence**. This came first
    and is fully covered by tests independent of any LLM behavior (see
    ``tests/test_orchestrator.py``).
  - ``claude_planner``: a **live Claude adapter** (schema-constrained tool
    use, evidence-only prompts) that proposes the same shape of plan/
    revision. It is tried first when enabled; on ANY failure -- missing
    API key, network error, malformed response, a plan that fails
    validation -- it raises ``ClaudePlannerError`` and this module falls
    back to the deterministic sequence and logs why. That fallback is a
    hard requirement from the build plan, not an edge case: a rate limit
    or an outage must never block the demo.

Allowed revision actions (matches the plan's allowlist, enforced in
``plan_schema.RevisionAction``):
  - "change_feature_windows" / "add_rolling_lag_features...": richer or
    different feature engineering
  - "remove_unstable_features": drop features from the sensor subset
  - "switch_model_family": try a different model family
  - "tune_search_space": bounded hyperparameter search on the current model
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..data import cmapss
from ..data.baseline import (
    AttemptResult,
    bounded_random_search,
    bounded_search_from_space,
    mean_rul_baseline_predictions,
    run_attempt,
)
from ..data.evaluate import full_report
from .plan_schema import FeatureSpec, ModelFamily, PipelinePlan, SearchSpace
from .trust_gate import TrustGateResult, evaluate_trust_gate

logger = logging.getLogger(__name__)

MAX_REVISIONS = 2  # -> 3 total attempts, per the plan


@dataclass
class LoggedAttempt:
    result: AttemptResult
    gate: TrustGateResult
    revision_action: str
    revision_rationale: str
    plan_source: str = "deterministic_fallback"  # "claude" or "deterministic_fallback"
    fallback_reason: str | None = None  # set only when a live Claude call was tried and failed


@dataclass
class PipelineRunResult:
    baseline_test_rmse: float
    attempts: list[LoggedAttempt] = field(default_factory=list)
    promoted: bool = False
    promoted_attempt: int | None = None
    stopped_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "baseline_test_rmse": self.baseline_test_rmse,
            "promoted": self.promoted,
            "promoted_attempt": self.promoted_attempt,
            "stopped_reason": self.stopped_reason,
            "attempts": [
                {
                    "attempt_number": a.result.attempt_number,
                    "feature_spec": a.result.feature_spec,
                    "model_family": a.result.model_family,
                    "hyperparams": a.result.hyperparams,
                    "n_features": a.result.n_features,
                    "validation_metrics": a.result.validation_metrics,
                    "test_metrics": a.result.test_metrics,
                    "conformal_coverage": a.result.conformal_coverage,
                    "revision_action": a.revision_action,
                    "revision_rationale": a.revision_rationale,
                    "plan_source": a.plan_source,
                    "fallback_reason": a.fallback_reason,
                    "trust_gate": a.gate.as_evidence(),
                }
                for a in self.attempts
            ],
        }


#: A short, classically-informative subset of C-MAPSS sensors (see e.g.
#: Saxena & Goebel 2008 and most published FD001 baselines) — deliberately
#: NOT the full usable sensor set, because attempt 1 is meant to be the
#: cheapest plan a time-pressed engineer would try first, not the best
#: plan we're capable of.
_QUICK_SENSOR_SUBSET = ["sensor_2", "sensor_4", "sensor_11", "sensor_15"]


def _next_plan(attempt_number: int, sensor_cols: list[str]):
    """Deterministic fallback planner. Returns (feature_spec, model_family,
    hyperparams, action, rationale).

    Calibrated against the real FD001 data (see docs/day1_status.md): a
    linear model on a handful of sensors genuinely fails the trust gate's
    22-cycle test-RMSE bar (~23.4 cycles), and a Random Forest on the full
    sensor set genuinely passes it (~18.4 cycles) — a real revision
    responding to a real failure, not a scripted demo.
    """
    if attempt_number == 1:
        return (
            {"windows": (), "lags": (), "sensor_cols": _QUICK_SENSOR_SUBSET},
            "linear_regression",
            {},
            "initial_plan",
            "Start with the cheapest plausible plan: a linear model on the handful of "
            "sensors classically associated with degradation in this engine family, no "
            "feature engineering yet. If it clears the bar, there's no reason to spend "
            "more compute; if not, we know exactly what to add.",
        )
    if attempt_number == 2:
        return (
            {"windows": (), "lags": (), "sensor_cols": sensor_cols},
            "random_forest",
            {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 2},
            "switch_model_family",
            "Attempt 1 failed the trust gate on test RMSE. A linear model can't capture "
            "the non-linear relationship between sensor readings and remaining life, and "
            "a 4-sensor subset may be discarding useful signal — switch to a Random "
            "Forest and use the full set of non-constant sensors.",
        )
    return (
        {"windows": cmapss.ROLLING_WINDOWS, "lags": cmapss.LAG_STEPS, "sensor_cols": sensor_cols},
        "xgboost",
        None,  # filled in by a bounded search at call time
        "tune_search_space",
        "Attempt 2 still failed the trust gate. Add rolling mean/std and lag features so "
        "the model sees degradation trajectory, not just a snapshot, and run a bounded "
        "hyperparameter search over gradient boosting rather than guessing parameters.",
    )


def _plan_obj_to_internal(plan: PipelinePlan) -> tuple[dict, str, SearchSpace]:
    feature_spec = {
        "windows": tuple(plan.feature_spec.windows),
        "lags": tuple(plan.feature_spec.lags),
        "sensor_cols": list(plan.feature_spec.sensor_subset),
    }
    return feature_spec, plan.model_family.value, plan.search_space


def _internal_to_plan_obj(
    feature_spec: dict, model_family: str, hyperparams: dict | None, rationale: str
) -> PipelinePlan:
    """Build a structured PipelinePlan from a deterministic (or already-run
    Claude) attempt, so the NEXT attempt can always hand Claude a proper
    prior_plan for a revision call — regardless of whether the last
    attempt's plan came from Claude or the fallback."""
    hp = hyperparams or {}
    fs = FeatureSpec(
        windows=list(feature_spec.get("windows", ())),
        lags=list(feature_spec.get("lags", ())),
        sensor_subset=list(feature_spec.get("sensor_cols", [])),
    )
    ss = SearchSpace(
        n_estimators_choices=[hp.get("n_estimators", 200)],
        max_depth_choices=[hp.get("max_depth", 8)],
        learning_rate_choices=[hp["learning_rate"]] if "learning_rate" in hp else None,
        max_trials=1,
    )
    return PipelinePlan(
        task_type="regression",
        target_column="RUL",
        asset_column="unit_number",
        cycle_column="time_cycles",
        feature_spec=fs,
        model_family=ModelFamily(model_family),
        search_space=ss,
        validation_strategy="group_kfold_by_engine",
        rationale=rationale[:600],
    )


def _resolve_plan(
    attempt_number: int,
    sensor_cols: list[str],
    profile_summary: dict,
    prior_plan_obj: PipelinePlan | None,
    prior_evidence: dict | None,
    use_claude: bool,
):
    """Try Claude (if enabled), fall back to the deterministic sequence on
    any failure. Returns (feature_spec, model_family, hyperparams_or_None,
    search_space_or_None, action, rationale, source, fallback_reason).
    """
    fallback_reason: str | None = None

    if use_claude:
        try:
            from .claude_planner import propose_initial_plan, propose_revision

            if attempt_number == 1:
                plan_obj = propose_initial_plan(profile_summary, sensor_cols)
                action, rationale = "initial_plan", plan_obj.rationale
            else:
                assert prior_plan_obj is not None and prior_evidence is not None
                revision = propose_revision(prior_plan_obj, prior_evidence, sensor_cols)
                plan_obj = revision.updated_plan
                action, rationale = revision.action.value, revision.rationale

            feature_spec, model_family, search_space = _plan_obj_to_internal(plan_obj)
            return feature_spec, model_family, None, search_space, action, rationale, "claude", None, plan_obj

        except Exception as exc:  # noqa: BLE001 - ClaudePlannerError or an import failure, both fall back
            fallback_reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Claude planner unavailable on attempt %d, falling back: %s", attempt_number, fallback_reason
            )

    feature_spec, model_family, hyperparams, action, rationale = _next_plan(attempt_number, sensor_cols)
    plan_obj = _internal_to_plan_obj(feature_spec, model_family, hyperparams, rationale)
    return (
        feature_spec,
        model_family,
        hyperparams,
        None,
        action,
        rationale,
        "deterministic_fallback",
        fallback_reason,
        plan_obj,
    )


def run_reflection_loop(
    data_dir: Path,
    max_revisions: int = MAX_REVISIONS,
    use_claude: bool | None = None,
    on_attempt=None,
) -> PipelineRunResult:
    """Run the full loop. ``use_claude=None`` (default) auto-detects: try
    Claude only if ANTHROPIC_API_KEY is set in the environment. Pass
    ``use_claude=False`` to force the deterministic path (used by the fast
    test suite so results are reproducible without any network access).

    ``on_attempt``, if given, is called as ``on_attempt(logged_attempt,
    fitted_model)`` right after each attempt is appended to the result --
    this is what lets a caller (the FastAPI run service) persist progress
    and save a model artifact incrementally instead of only getting a
    result once the whole loop finishes, without the ML core itself
    knowing anything about databases, SSE, or the filesystem.
    """
    if use_claude is None:
        use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))

    train_df = cmapss.load_fd001_train(data_dir)
    test_df, true_rul = cmapss.load_fd001_test(data_dir)
    test_last = cmapss.last_cycle_per_engine(test_df)
    y_test_true = true_rul.loc[test_last["unit_number"]].reset_index(drop=True)

    profile = cmapss.profile_dataset(train_df)
    sensor_cols = cmapss.usable_sensor_columns(profile)
    profile_summary = {
        "n_engines": profile.n_engines,
        "n_rows": profile.n_rows,
        "constant_sensors": profile.constant_sensors,
        "near_constant_sensors": profile.near_constant_sensors,
        "engine_life_stats": profile.engine_life_stats,
        "leakage_risk_notes": profile.leakage_risk_notes,
    }

    baseline_pred = mean_rul_baseline_predictions(train_df["RUL"], len(y_test_true))
    baseline_test_rmse = full_report(y_test_true, baseline_pred)["rmse"]

    result = PipelineRunResult(baseline_test_rmse=baseline_test_rmse)
    prior_plan_obj: PipelinePlan | None = None
    prior_evidence: dict | None = None

    for attempt_number in range(1, max_revisions + 2):
        (
            feature_spec,
            model_family,
            hyperparams,
            search_space,
            action,
            rationale,
            source,
            fallback_reason,
            plan_obj_used,
        ) = _resolve_plan(attempt_number, sensor_cols, profile_summary, prior_plan_obj, prior_evidence, use_claude)

        attempt_sensor_cols = feature_spec["sensor_cols"]
        engineered_train = cmapss.engineer_features(
            train_df, attempt_sensor_cols, windows=feature_spec["windows"], lags=feature_spec["lags"]
        )
        engineered_test_last = cmapss.engineer_features(
            test_df, attempt_sensor_cols, windows=feature_spec["windows"], lags=feature_spec["lags"]
        )
        engineered_test_last = cmapss.last_cycle_per_engine(engineered_test_last)

        feat_cols = cmapss.feature_columns(engineered_train, attempt_sensor_cols)
        if model_family == "linear_regression" and not feature_spec["windows"] and not feature_spec["lags"]:
            # a plain linear model on just its sensor subset, no op settings
            # (op settings carry ~no signal in FD001's single operating
            # condition and would just add noise / near-collinearity)
            feat_cols = attempt_sensor_cols
        no_leakage = not ({"unit_number", "time_cycles"} & set(feat_cols))

        X_train = engineered_train[feat_cols]
        y_train = engineered_train["RUL"]
        groups_train = engineered_train["unit_number"]
        X_test_last = engineered_test_last[feat_cols]

        if hyperparams is None:
            if search_space is not None:
                hyperparams, _ = bounded_search_from_space(
                    model_family, search_space, X_train, y_train, groups_train
                )
            else:
                hyperparams, _ = bounded_random_search(
                    model_family, X_train, y_train, groups_train, max_trials=6
                )

        attempt_result, fitted_model = run_attempt(
            attempt_number=attempt_number,
            feature_spec={k: v for k, v in feature_spec.items() if k != "sensor_cols"},
            model_family=model_family,
            hyperparams=hyperparams,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            X_test_last_cycle=X_test_last,
            y_test_true=y_test_true,
        )

        gate = evaluate_trust_gate(
            no_leakage=no_leakage,
            baseline_test_rmse=baseline_test_rmse,
            candidate_test_rmse=attempt_result.test_metrics["rmse"],
            candidate_val_rmse=attempt_result.validation_metrics["rmse"],
            conformal_coverage=attempt_result.conformal_coverage,
        )

        logged = LoggedAttempt(
            result=attempt_result,
            gate=gate,
            revision_action=action,
            revision_rationale=rationale,
            plan_source=source,
            fallback_reason=fallback_reason,
        )
        result.attempts.append(logged)
        if on_attempt is not None:
            on_attempt(logged, fitted_model)

        # keep a structured plan object for the next Claude revision call,
        # regardless of whether THIS attempt's plan came from Claude or
        # the deterministic fallback
        prior_plan_obj = _internal_to_plan_obj(feature_spec, model_family, hyperparams, rationale)
        prior_evidence = gate.as_evidence()

        if gate.passed:
            result.promoted = True
            result.promoted_attempt = attempt_number
            result.stopped_reason = "trust_gate_passed"
            return result

        if attempt_number == max_revisions + 1:
            result.stopped_reason = "max_revisions_exhausted"

    return result

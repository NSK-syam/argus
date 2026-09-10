"""Tests for the live Claude adapter's plumbing, using a mocked Anthropic
client -- no network access, no API key required. What's under test here
is NOT "does Claude give a good answer" (that needs a real key and isn't
reproducible in CI); it's the fail-closed contract: a valid tool response
parses into a real PipelinePlan/Revision, and every kind of bad response
(no tool call, invalid schema, unsafe plan) raises ClaudePlannerError so
the orchestrator's fallback actually triggers.
"""

from types import SimpleNamespace

import pytest

from app.ml.pipeline.claude_planner import (
    ClaudePlannerError,
    _PLAN_TOOL_NAME,
    _REVISION_TOOL_NAME,
    propose_initial_plan,
    propose_revision,
)
from app.ml.pipeline.plan_schema import FeatureSpec, ModelFamily, PipelinePlan, SearchSpace

USABLE_SENSORS = ["sensor_2", "sensor_4", "sensor_7", "sensor_11", "sensor_15"]

VALID_PLAN_INPUT = {
    "task_type": "regression",
    "target_column": "RUL",
    "asset_column": "unit_number",
    "cycle_column": "time_cycles",
    "feature_spec": {"windows": [5, 10], "lags": [1], "sensor_subset": ["sensor_2", "sensor_4"]},
    "model_family": "random_forest",
    "search_space": {
        "n_estimators_choices": [150, 250],
        "max_depth_choices": [6, 8],
        "learning_rate_choices": None,
        "max_trials": 4,
    },
    "validation_strategy": "group_kfold_by_engine",
    "rationale": "A reasonable plan for testing.",
}


def _fake_message_with_tool_use(tool_name: str, tool_input: dict):
    block = SimpleNamespace(type="tool_use", name=tool_name, input=tool_input)
    return SimpleNamespace(content=[block])


def _fake_client(response_message=None, raise_exc: Exception | None = None):
    client = SimpleNamespace()

    def create(**kwargs):
        if raise_exc is not None:
            raise raise_exc
        return response_message

    client.messages = SimpleNamespace(create=create)
    return client


def test_propose_initial_plan_parses_a_valid_tool_response():
    client = _fake_client(_fake_message_with_tool_use(_PLAN_TOOL_NAME, VALID_PLAN_INPUT))
    plan = propose_initial_plan({"n_engines": 100}, USABLE_SENSORS, client=client)
    assert isinstance(plan, PipelinePlan)
    assert plan.model_family == ModelFamily.random_forest
    assert plan.feature_spec.sensor_subset == ["sensor_2", "sensor_4"]


def test_propose_initial_plan_raises_when_no_tool_call_made():
    text_block = SimpleNamespace(type="text", text="I'd rather just chat about this.")
    client = _fake_client(SimpleNamespace(content=[text_block]))
    with pytest.raises(ClaudePlannerError):
        propose_initial_plan({"n_engines": 100}, USABLE_SENSORS, client=client)


def test_propose_initial_plan_raises_on_schema_violation():
    bad_input = dict(VALID_PLAN_INPUT)
    bad_input["target_column"] = "totally_wrong_column"  # violates the fixed-value validator
    client = _fake_client(_fake_message_with_tool_use(_PLAN_TOOL_NAME, bad_input))
    with pytest.raises(ClaudePlannerError):
        propose_initial_plan({"n_engines": 100}, USABLE_SENSORS, client=client)


def test_propose_initial_plan_raises_on_unsafe_sensor_subset():
    bad_input = dict(VALID_PLAN_INPUT)
    bad_input["feature_spec"] = dict(VALID_PLAN_INPUT["feature_spec"])
    bad_input["feature_spec"]["sensor_subset"] = ["sensor_99"]  # not in the usable set
    client = _fake_client(_fake_message_with_tool_use(_PLAN_TOOL_NAME, bad_input))
    with pytest.raises(ClaudePlannerError):
        propose_initial_plan({"n_engines": 100}, USABLE_SENSORS, client=client)


def test_propose_initial_plan_raises_on_api_failure():
    client = _fake_client(raise_exc=RuntimeError("connection reset"))
    with pytest.raises(ClaudePlannerError):
        propose_initial_plan({"n_engines": 100}, USABLE_SENSORS, client=client)


def test_propose_revision_parses_a_valid_tool_response():
    prior_plan = PipelinePlan.model_validate(VALID_PLAN_INPUT)
    revision_input = {
        "action": "switch_model_family",
        "updated_plan": {**VALID_PLAN_INPUT, "model_family": "xgboost",
                          "search_space": {**VALID_PLAN_INPUT["search_space"],
                                            "learning_rate_choices": [0.05, 0.1]}},
        "rationale": "Random Forest underfit; try gradient boosting.",
    }
    client = _fake_client(_fake_message_with_tool_use(_REVISION_TOOL_NAME, revision_input))
    revision = propose_revision(prior_plan, {"passed": False}, USABLE_SENSORS, client=client)
    assert revision.action.value == "switch_model_family"
    assert revision.updated_plan.model_family == ModelFamily.xgboost


def test_propose_revision_raises_when_xgboost_has_no_learning_rate_choices():
    prior_plan = PipelinePlan.model_validate(VALID_PLAN_INPUT)
    revision_input = {
        "action": "switch_model_family",
        "updated_plan": {**VALID_PLAN_INPUT, "model_family": "xgboost"},  # no learning_rate_choices
        "rationale": "Switch to boosting.",
    }
    client = _fake_client(_fake_message_with_tool_use(_REVISION_TOOL_NAME, revision_input))
    with pytest.raises(ClaudePlannerError):
        propose_revision(prior_plan, {"passed": False}, USABLE_SENSORS, client=client)


def test_missing_api_key_raises_without_any_network_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ClaudePlannerError):
        propose_initial_plan({"n_engines": 100}, USABLE_SENSORS)  # no client injected -> real _client() path

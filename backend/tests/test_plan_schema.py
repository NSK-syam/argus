"""Regression tests for two real findings from an actual (non-mocked) live
Claude API run against the real Anthropic API (see docs/day1_status.md):

1. Claude's genuine revision rationale ran past the schema's 600-character
   cap, and the client-side Pydantic validation rejected the whole
   revision over prose length alone -- silently falling back to the
   deterministic planner exactly when the live-planner differentiator
   matters most. Fix: truncate an overlong rationale instead of rejecting.

2. A genuine Claude-proposed search_space used hyperparameter *values*
   (not just a count of choices) far outside what's sane for this
   dataset's size, turning one "bounded, <=8 trial" search into a run
   that was still training after nearly 30 minutes on this dataset. Fix:
   bound each individual value, not just how many choices are offered.
"""

import pytest
from pydantic import ValidationError

from app.ml.pipeline.plan_schema import PipelinePlan, Revision, SearchSpace

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


def test_overlong_plan_rationale_is_truncated_not_rejected():
    long_rationale = "x" * 900
    plan = PipelinePlan.model_validate({**VALID_PLAN_INPUT, "rationale": long_rationale})
    assert len(plan.rationale) == 600
    assert plan.rationale.endswith("...")


def test_overlong_revision_rationale_is_truncated_not_rejected():
    revision = Revision.model_validate(
        {
            "action": "tune_search_space",
            "updated_plan": VALID_PLAN_INPUT,
            "rationale": "y" * 900,
        }
    )
    assert len(revision.rationale) == 600
    assert revision.rationale.endswith("...")


def test_rationale_within_limit_is_left_untouched():
    plan = PipelinePlan.model_validate(VALID_PLAN_INPUT)
    assert plan.rationale == "A reasonable plan for testing."


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("n_estimators_choices", [5000]),
        ("n_estimators_choices", [10]),
        ("max_depth_choices", [100]),
        ("max_depth_choices", [1]),
        ("learning_rate_choices", [2.5]),
    ],
)
def test_search_space_rejects_out_of_bounds_hyperparameter_values(field, bad_value):
    """The real bug: only the *count* of choices was ever bounded
    (max_length=4) -- nothing stopped a value like n_estimators=5000 or
    max_depth=100 from being proposed and actually trained on, which is
    what turned one live revision into a run that ran for ~30 minutes
    instead of finishing in the demo's time budget."""
    base = {
        "n_estimators_choices": [150],
        "max_depth_choices": [6],
        "learning_rate_choices": [0.05],
    }
    with pytest.raises(ValidationError):
        SearchSpace.model_validate({**base, field: bad_value, "max_trials": 4})


def test_search_space_accepts_values_within_bounds():
    space = SearchSpace.model_validate(
        {
            "n_estimators_choices": [25, 400],
            "max_depth_choices": [2, 15],
            "learning_rate_choices": [0.01, 0.3],
            "max_trials": 8,
        }
    )
    assert space.n_estimators_choices == [25, 400]
    assert space.max_depth_choices == [2, 15]

"""The structured contract between the orchestrator and any Planner
(deterministic fallback today, live Claude adapter as of this module).

Everything Claude is allowed to propose is expressed as a Pydantic model
and enforced as a JSON schema on the API call itself (forced tool-use), so
"malformed Claude output" mostly can't happen at the wire level — what's
left to validate is domain-level sanity (e.g. cycle counts, dedup), which
`validate_plan_is_safe` below checks explicitly.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

# Keep this in sync with app.ml.data.cmapss.SENSOR_COLS
_ALL_SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

_RATIONALE_MAX_LENGTH = 600


def _truncate_rationale(v: object) -> object:
    """Trim an overlong rationale instead of rejecting the whole plan over
    prose length. Found via a real (non-mocked) Claude API call: the model
    is told to "keep rationale concise" but nothing stops it from running a
    little over the schema's max_length on a genuine response, and rationale
    text is narrative explanation, not a safety-relevant field -- unlike
    task_type/target_column/validation_strategy above, there's no reason a
    live plan that is otherwise valid should fall back to the deterministic
    planner just because its explanation ran long."""
    if isinstance(v, str) and len(v) > _RATIONALE_MAX_LENGTH:
        return v[: _RATIONALE_MAX_LENGTH - 3].rstrip() + "..."
    return v


class ModelFamily(str, Enum):
    linear_regression = "linear_regression"
    random_forest = "random_forest"
    xgboost = "xgboost"


class RevisionAction(str, Enum):
    """The allowlist from the build plan: 'change feature windows, remove
    unstable features, switch model family, or revise the bounded search
    space.' Nothing outside this enum is a valid revision — Claude cannot,
    for example, propose changing the trust-gate thresholds or the
    validation strategy."""

    change_feature_windows = "change_feature_windows"
    remove_unstable_features = "remove_unstable_features"
    switch_model_family = "switch_model_family"
    tune_search_space = "tune_search_space"


class FeatureSpec(BaseModel):
    windows: list[int] = Field(
        default_factory=list, description="rolling-window sizes in cycles, e.g. [5, 10, 20]"
    )
    lags: list[int] = Field(default_factory=list, description="lag steps in cycles, e.g. [1, 5]")
    sensor_subset: list[str] = Field(
        description="which usable sensor columns to use; must be a non-empty subset of the "
        "profiled usable sensors passed in the prompt"
    )

    @field_validator("windows", "lags")
    @classmethod
    def _bounded_and_positive(cls, v: list[int]) -> list[int]:
        if any(x <= 0 for x in v):
            raise ValueError("windows/lags must be positive")
        if len(v) > 5:
            raise ValueError("at most 5 windows/lags — this is a bounded search, not a free-for-all")
        return v


#: Numeric bounds on each individual hyperparameter *value* Claude may
#: propose. Found via a real (non-mocked) Claude API call: max_trials was
#: already capped at 8, but nothing bounded how expensive a single trial
#: could be -- only the *count* of choices was capped (max_length=4), not
#: their *magnitude*. A live revision proposed a search space with values
#: well outside the deterministic fallback's own range (n_estimators up to
#: 350, max_depth up to 12) and turned one "bounded, <=8 trial" search into
#: a run that was still going after nearly 30 minutes on this dataset --
#: a real, reproducible way the plan's own compute budget could be blown
#: through, not a hypothetical. These ranges give real headroom over what
#: the deterministic fallback uses while still ruling out pathological
#: values.
_N_ESTIMATORS_BOUNDS = (25, 400)
_MAX_DEPTH_BOUNDS = (2, 15)
_LEARNING_RATE_BOUNDS = (0.01, 0.3)


class SearchSpace(BaseModel):
    n_estimators_choices: list[Annotated[int, Field(ge=_N_ESTIMATORS_BOUNDS[0], le=_N_ESTIMATORS_BOUNDS[1])]] = (
        Field(min_length=1, max_length=4)
    )
    max_depth_choices: list[Annotated[int, Field(ge=_MAX_DEPTH_BOUNDS[0], le=_MAX_DEPTH_BOUNDS[1])]] = Field(
        min_length=1, max_length=4
    )
    learning_rate_choices: (
        list[Annotated[float, Field(ge=_LEARNING_RATE_BOUNDS[0], le=_LEARNING_RATE_BOUNDS[1])]] | None
    ) = Field(default=None, description="only used when model_family == xgboost")
    max_trials: int = Field(ge=1, le=8, description="bounded per the plan's compute budget")


class PipelinePlan(BaseModel):
    task_type: str = Field(description="must be 'regression' for RUL prediction")
    target_column: str = Field(description="must be 'RUL'")
    asset_column: str = Field(description="must be 'unit_number'")
    cycle_column: str = Field(description="must be 'time_cycles'")
    feature_spec: FeatureSpec
    model_family: ModelFamily
    search_space: SearchSpace
    validation_strategy: str = Field(description="must be 'group_kfold_by_engine'")
    rationale: str = Field(max_length=_RATIONALE_MAX_LENGTH)

    @field_validator("rationale", mode="before")
    @classmethod
    def _rationale_truncated(cls, v: object) -> object:
        return _truncate_rationale(v)

    @field_validator("task_type")
    @classmethod
    def _task_type_fixed(cls, v: str) -> str:
        if v != "regression":
            raise ValueError("only 'regression' is supported in this prototype")
        return v

    @field_validator("target_column")
    @classmethod
    def _target_fixed(cls, v: str) -> str:
        if v != "RUL":
            raise ValueError("target_column must be 'RUL' — this is a fixed schema, not a proposal")
        return v

    @field_validator("validation_strategy")
    @classmethod
    def _validation_fixed(cls, v: str) -> str:
        if v != "group_kfold_by_engine":
            raise ValueError(
                "validation_strategy must be 'group_kfold_by_engine' — row-random splits are "
                "a leakage risk on this dataset and are never a valid proposal"
            )
        return v


class Revision(BaseModel):
    action: RevisionAction
    updated_plan: PipelinePlan
    rationale: str = Field(max_length=_RATIONALE_MAX_LENGTH)

    @field_validator("rationale", mode="before")
    @classmethod
    def _rationale_truncated(cls, v: object) -> object:
        return _truncate_rationale(v)


def validate_plan_is_safe(plan: PipelinePlan, usable_sensor_cols: list[str]) -> list[str]:
    """Domain-level checks beyond what the JSON schema alone can express.
    Returns a list of problems; empty list = safe to execute."""
    problems: list[str] = []

    unknown = set(plan.feature_spec.sensor_subset) - set(usable_sensor_cols)
    if unknown:
        problems.append(f"sensor_subset references sensors not in the usable profiled set: {sorted(unknown)}")
    if not plan.feature_spec.sensor_subset:
        problems.append("sensor_subset must not be empty")

    if plan.model_family == ModelFamily.xgboost and not plan.search_space.learning_rate_choices:
        problems.append("xgboost requires at least one learning_rate choice in the search space")

    if plan.search_space.max_trials * len(plan.feature_spec.windows or [1]) > 32:
        problems.append("search space too large for the compute budget (max_trials x windows)")

    return problems

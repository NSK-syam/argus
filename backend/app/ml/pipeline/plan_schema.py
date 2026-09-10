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

from pydantic import BaseModel, Field, field_validator

# Keep this in sync with app.ml.data.cmapss.SENSOR_COLS
_ALL_SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


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


class SearchSpace(BaseModel):
    n_estimators_choices: list[int] = Field(min_length=1, max_length=4)
    max_depth_choices: list[int] = Field(min_length=1, max_length=4)
    learning_rate_choices: list[float] | None = Field(
        default=None, description="only used when model_family == xgboost"
    )
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
    rationale: str = Field(max_length=600)

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
    rationale: str = Field(max_length=600)


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

"""Live Claude adapter for the Planner/Critic step.

Fail-closed by design: every failure mode (no API key, network error,
timeout, malformed tool response, a plan that doesn't validate, a plan
that fails domain-level safety checks) raises ``ClaudePlannerError`` and
nothing else. Callers (``orchestrator.py``) must catch it and fall back to
the deterministic planner -- that fallback is a hard requirement from the
build plan, not an edge case, so this module never tries to be clever
about partial recovery. It either returns a fully valid, fully validated
plan, or it raises.

Claude sees ONLY:
  - a deterministic dataset profile (schema, missingness, constant-sensor
    list, engine-life stats) -- never raw sensor rows
  - on retries, the trust gate's structured evidence (metrics + which
    checks passed/failed) -- never chain-of-thought, never a data sample

The response is forced through Anthropic's tool-use mechanism with the
Pydantic-generated JSON schema as the tool's input_schema, so "Claude
returned something that isn't a plan" is rare by construction; the
Pydantic validation afterward is the second line of defense, and
``validate_plan_is_safe`` is the third (domain rules a JSON schema alone
can't express, e.g. "sensor_subset must be a subset of usable sensors").
"""

from __future__ import annotations

import json
import logging
import os

from pydantic import ValidationError

from .plan_schema import PipelinePlan, Revision, validate_plan_is_safe

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"

_PLAN_TOOL_NAME = "propose_pipeline_plan"
_REVISION_TOOL_NAME = "propose_revision"

SYSTEM_PROMPT = (
    "You are the Planner/Critic component of Argus, an agentic predictive-maintenance "
    "ML pipeline for NASA C-MAPSS turbofan sensor data. You never see raw sensor rows -- "
    "only a deterministic data profile and, on retries, structured trust-gate evidence "
    "(metrics and which specific checks failed). Propose a modeling plan (or, on a retry, "
    "exactly ONE revision from an explicit allowlist) by calling the provided tool. Always "
    "call the tool -- never respond with plain text. Keep rationale concise and grounded "
    "only in the evidence given; do not invent sensor semantics you were not told."
)


class ClaudePlannerError(Exception):
    """Any failure in the live planning path. Callers must catch this and
    fall back to the deterministic planner."""


def _client():
    import anthropic  # lazy import: the rest of the module still imports cleanly without it

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ClaudePlannerError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def _extract_tool_input(message, tool_name: str) -> dict:
    for block in getattr(message, "content", []):
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            return block.input
    raise ClaudePlannerError(f"Claude did not call the expected tool '{tool_name}'")


def propose_initial_plan(
    profile_summary: dict,
    usable_sensor_cols: list[str],
    model: str = DEFAULT_MODEL,
    client=None,
) -> PipelinePlan:
    """Attempt-1 planning call. ``client`` is injectable for testing; in
    production it's built from ANTHROPIC_API_KEY."""
    client = client or _client()

    tool = {
        "name": _PLAN_TOOL_NAME,
        "description": "Propose an initial modeling plan for RUL prediction on FD001.",
        "input_schema": PipelinePlan.model_json_schema(),
    }
    user_content = (
        "Dataset profile (deterministic, no raw rows):\n"
        f"{json.dumps(profile_summary, indent=2, default=str)}\n\n"
        f"Usable (non-constant) sensor columns: {usable_sensor_cols}\n\n"
        "Propose the CHEAPEST plausible first plan -- simple model, small sensor subset, "
        "no feature engineering yet -- by calling the tool. If it's good enough there's no "
        "reason to have spent more compute; if not, the trust gate will say exactly why and "
        "you'll get a chance to revise."
    )

    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": _PLAN_TOOL_NAME},
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception as exc:  # noqa: BLE001 - any API failure must degrade, never raise past this module
        raise ClaudePlannerError(f"Claude API call failed: {exc}") from exc

    raw = _extract_tool_input(message, _PLAN_TOOL_NAME)
    try:
        plan = PipelinePlan.model_validate(raw)
    except ValidationError as exc:
        raise ClaudePlannerError(f"Claude returned an invalid plan: {exc}") from exc

    problems = validate_plan_is_safe(plan, usable_sensor_cols)
    if problems:
        raise ClaudePlannerError(f"Claude's plan failed safety validation: {problems}")

    return plan


def propose_revision(
    prior_plan: PipelinePlan,
    gate_evidence: dict,
    usable_sensor_cols: list[str],
    model: str = DEFAULT_MODEL,
    client=None,
) -> Revision:
    """Retry planning call: given the prior plan and the trust gate's
    structured failure evidence, propose exactly ONE allowlisted revision.
    Same fail-closed contract as ``propose_initial_plan``."""
    client = client or _client()

    tool = {
        "name": _REVISION_TOOL_NAME,
        "description": (
            "Propose exactly one revision to the prior plan, in response to trust-gate "
            "failure evidence. The action must be one of the allowlisted revision types."
        ),
        "input_schema": Revision.model_json_schema(),
    }
    user_content = (
        "Prior plan:\n"
        f"{json.dumps(prior_plan.model_dump(mode='json'), indent=2, default=str)}\n\n"
        "Trust gate evidence from that plan's result (this is ALL the information you have "
        "about why it failed -- no raw data, no chain-of-thought, just these numbers):\n"
        f"{json.dumps(gate_evidence, indent=2, default=str)}\n\n"
        f"Usable (non-constant) sensor columns: {usable_sensor_cols}\n\n"
        "Propose ONE revision by calling the tool. Pick the single change most likely to "
        "address the specific failed check(s) above -- don't change everything at once."
    )

    try:
        message = client.messages.create(
            model=model,
            max_tokens=1536,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": _REVISION_TOOL_NAME},
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception as exc:  # noqa: BLE001
        raise ClaudePlannerError(f"Claude API call failed: {exc}") from exc

    raw = _extract_tool_input(message, _REVISION_TOOL_NAME)
    try:
        revision = Revision.model_validate(raw)
    except ValidationError as exc:
        raise ClaudePlannerError(f"Claude returned an invalid revision: {exc}") from exc

    problems = validate_plan_is_safe(revision.updated_plan, usable_sensor_cols)
    if problems:
        raise ClaudePlannerError(f"Claude's revision failed safety validation: {problems}")

    return revision

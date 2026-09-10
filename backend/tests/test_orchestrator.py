"""Integration test for the Planner -> Trainer -> Critic reflection loop
against the real FD001 dataset.

This is the direct test for the plan's acceptance criterion: "Verify an
intentionally weak first attempt produces a permitted revision and a
second logged attempt." It's marked slow (trains real models) and skips
gracefully if the dataset hasn't been downloaded into data/cmapss.
"""

from pathlib import Path

import pytest

from app.ml.pipeline.orchestrator import MAX_REVISIONS, run_reflection_loop

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cmapss"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "train_FD001.txt").exists(),
    reason="FD001 data not present at data/cmapss (see README for the download step)",
)


@pytest.mark.slow
def test_reflection_loop_retries_after_a_genuinely_weak_first_attempt():
    result = run_reflection_loop(DATA_DIR)

    # the deliberately cheap first attempt (linear regression on 4
    # sensors) must fail the trust gate on real FD001 data
    assert len(result.attempts) >= 2, "expected at least one retry, got a single attempt"
    first, second = result.attempts[0], result.attempts[1]
    assert first.gate.passed is False, "attempt 1 was expected to fail the trust gate"
    assert first.revision_action == "initial_plan"
    assert second.revision_action != "initial_plan", "attempt 2 must be a genuine revision, not a repeat"

    # and the revision must actually be logged with the evidence that
    # motivated it (this is what would be handed to a live Claude call)
    assert first.gate.as_evidence()["passed"] is False

    # the loop must eventually reach a promotable model on this dataset
    assert result.promoted is True
    assert result.promoted_attempt == second.result.attempt_number


@pytest.mark.slow
def test_reflection_loop_never_exceeds_max_revisions():
    result = run_reflection_loop(DATA_DIR)
    assert len(result.attempts) <= MAX_REVISIONS + 1

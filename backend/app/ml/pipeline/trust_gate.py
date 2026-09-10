"""The deterministic promotion gate.

This is the part of the plan that explicitly does NOT trust an LLM's
judgment: whether a model is good enough to promote is decided entirely
by reproducible arithmetic on held-out metrics. An LLM (Planner/Critic)
only ever sees the *output* of this gate — pass/fail plus which specific
condition(s) failed — and proposes a revision from an allowlisted set of
actions. It never overrides the gate itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_IMPROVEMENT_OVER_BASELINE = 0.15  # 15% RMSE improvement vs. mean-RUL baseline
MAX_TEST_RMSE = 22.0  # cycles; FD001 official test set
MAX_VAL_TO_TEST_DEGRADATION = 0.30  # relative
MIN_CONFORMAL_COVERAGE = 0.85


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class TrustGateResult:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)

    def failures(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]

    def as_evidence(self) -> dict:
        """Structured, LLM-facing evidence — metrics and pass/fail flags
        only. No raw data rows, no chain-of-thought, just the numbers a
        revision decision should be based on."""
        return {
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }


def evaluate_trust_gate(
    *,
    no_leakage: bool,
    baseline_test_rmse: float,
    candidate_test_rmse: float,
    candidate_val_rmse: float,
    conformal_coverage: float,
) -> TrustGateResult:
    checks: list[GateCheck] = []

    checks.append(
        GateCheck(
            name="no_leakage_or_schema_blockers",
            passed=no_leakage,
            detail="no unresolved leakage/schema blockers" if no_leakage else "leakage or schema blocker present",
        )
    )

    improvement = 1.0 - (candidate_test_rmse / baseline_test_rmse) if baseline_test_rmse > 0 else 0.0
    checks.append(
        GateCheck(
            name="beats_baseline_by_15pct",
            passed=improvement >= MIN_IMPROVEMENT_OVER_BASELINE,
            detail=f"{improvement:.1%} RMSE improvement over baseline (need >= {MIN_IMPROVEMENT_OVER_BASELINE:.0%})",
        )
    )

    checks.append(
        GateCheck(
            name="test_rmse_under_22_cycles",
            passed=candidate_test_rmse <= MAX_TEST_RMSE,
            detail=f"test RMSE = {candidate_test_rmse:.2f} cycles (need <= {MAX_TEST_RMSE})",
        )
    )

    degradation = (
        (candidate_test_rmse - candidate_val_rmse) / candidate_val_rmse
        if candidate_val_rmse > 0
        else float("inf")
    )
    checks.append(
        GateCheck(
            name="val_to_test_degradation_under_30pct",
            passed=degradation <= MAX_VAL_TO_TEST_DEGRADATION,
            detail=f"val->test degradation = {degradation:.1%} (need <= {MAX_VAL_TO_TEST_DEGRADATION:.0%})",
        )
    )

    checks.append(
        GateCheck(
            name="conformal_coverage_at_least_85pct",
            passed=conformal_coverage >= MIN_CONFORMAL_COVERAGE,
            detail=f"conformal coverage = {conformal_coverage:.1%} (need >= {MIN_CONFORMAL_COVERAGE:.0%})",
        )
    )

    return TrustGateResult(passed=all(c.passed for c in checks), checks=checks)

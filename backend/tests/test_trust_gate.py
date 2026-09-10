from app.ml.pipeline.trust_gate import evaluate_trust_gate

GOOD = dict(
    no_leakage=True,
    baseline_test_rmse=40.0,
    candidate_test_rmse=18.0,  # >15% better than baseline, under 22
    candidate_val_rmse=17.5,  # small val->test degradation
    conformal_coverage=0.90,
)


def test_gate_passes_when_everything_is_good():
    result = evaluate_trust_gate(**GOOD)
    assert result.passed is True
    assert result.failures() == []


def test_gate_fails_on_leakage_flag_alone():
    params = dict(GOOD, no_leakage=False)
    result = evaluate_trust_gate(**params)
    assert result.passed is False
    names = {c.name for c in result.failures()}
    assert "no_leakage_or_schema_blockers" in names


def test_gate_fails_when_improvement_under_15_percent():
    # candidate barely better than baseline: (40-36)/40 = 10% improvement
    params = dict(GOOD, candidate_test_rmse=36.0, candidate_val_rmse=35.0)
    result = evaluate_trust_gate(**params)
    assert result.passed is False
    names = {c.name for c in result.failures()}
    assert "beats_baseline_by_15pct" in names


def test_gate_fails_when_test_rmse_over_22():
    params = dict(GOOD, candidate_test_rmse=23.0, candidate_val_rmse=22.0)
    result = evaluate_trust_gate(**params)
    assert result.passed is False
    names = {c.name for c in result.failures()}
    assert "test_rmse_under_22_cycles" in names


def test_gate_fails_on_val_to_test_degradation():
    # val looked great (10) but test is much worse (18) -> 80% degradation
    params = dict(GOOD, candidate_val_rmse=10.0, candidate_test_rmse=18.0)
    result = evaluate_trust_gate(**params)
    assert result.passed is False
    names = {c.name for c in result.failures()}
    assert "val_to_test_degradation_under_30pct" in names


def test_gate_fails_on_low_conformal_coverage():
    params = dict(GOOD, conformal_coverage=0.70)
    result = evaluate_trust_gate(**params)
    assert result.passed is False
    names = {c.name for c in result.failures()}
    assert "conformal_coverage_at_least_85pct" in names


def test_gate_evidence_is_structured_and_json_safe():
    import json

    result = evaluate_trust_gate(**GOOD)
    evidence = result.as_evidence()
    assert evidence["passed"] is True
    assert isinstance(evidence["checks"], list)
    assert all({"name", "passed", "detail"} <= set(c.keys()) for c in evidence["checks"])
    json.dumps(evidence)  # must be safe to hand to an LLM call as-is

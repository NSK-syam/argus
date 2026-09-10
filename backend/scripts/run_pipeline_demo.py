#!/usr/bin/env python3
"""Run the Planner->Trainer->Critic reflection loop end-to-end against the
real NASA C-MAPSS FD001 dataset and print a readable summary.

This is the concrete artifact for the plan's Day-1/Day-2 deliverables:
  - a working FD001 loader + baseline
  - proof that the pipeline retries at least once when the trust gate fails

Usage:
    python backend/scripts/run_pipeline_demo.py [--data-dir data/cmapss]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.pipeline.orchestrator import run_reflection_loop  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--json-out", default=None, help="optional path to dump full JSON results")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "cmapss"

    print(f"Loading FD001 from {data_dir} ...")
    t0 = time.time()
    result = run_reflection_loop(data_dir)
    elapsed = time.time() - t0

    print()
    print("=" * 72)
    print(f"Mean-RUL baseline test RMSE: {result.baseline_test_rmse:.2f} cycles")
    print("=" * 72)

    for logged in result.attempts:
        r = logged.result
        print()
        print(f"--- Attempt {r.attempt_number}: {logged.revision_action} ---")
        print(f"  rationale: {logged.revision_rationale}")
        print(f"  model_family: {r.model_family}  hyperparams: {r.hyperparams}")
        print(f"  feature_spec: {r.feature_spec}  (n_features={r.n_features})")
        print(
            f"  validation RMSE: {r.validation_metrics['rmse']:.2f}  "
            f"test RMSE: {r.test_metrics['rmse']:.2f}  "
            f"NASA score: {r.test_metrics['nasa_score']:.1f}  "
            f"F1@30: {r.test_metrics['f1_at_threshold']:.2f}"
        )
        print(f"  conformal coverage on test set: {r.conformal_coverage:.1%}")
        print(f"  trust gate: {'PASSED' if logged.gate.passed else 'FAILED'}")
        for check in logged.gate.checks:
            mark = "OK  " if check.passed else "FAIL"
            print(f"    [{mark}] {check.name}: {check.detail}")

    print()
    print("=" * 72)
    if result.promoted:
        print(f"RESULT: promotion_eligible after attempt {result.promoted_attempt} "
              f"({len(result.attempts)} attempt(s) run, {sum(1 for a in result.attempts) - 1} retry(ies))")
    else:
        print(f"RESULT: no attempt reached promotion_eligible "
              f"({len(result.attempts)} attempt(s) run). Reason: {result.stopped_reason}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 72)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nFull results written to {args.json_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the precomputed demo bundle: run the reflection loop once against
real FD001 data and save EVERY attempt's model + metadata to
app/demo_bundle/, so the API can seed a complete, already-finished run at
startup (app/services/run_service.py:seed_demo_run) -- letting a judge see
the full retry-evidence -> promote -> replay flow without waiting for live
training.

Usage:
    python3 backend/scripts/build_demo_artifact.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.ml.pipeline.demo_bundle import save_bundle  # noqa: E402
from app.ml.pipeline.orchestrator import run_reflection_loop  # noqa: E402


def main() -> None:
    captured = []

    def on_attempt(logged, model):
        captured.append((logged, model))

    print(f"Running the reflection loop against {settings.data_dir} ...")
    result = run_reflection_loop(settings.data_dir, use_claude=False, on_attempt=on_attempt)

    if not result.promoted:
        print("WARNING: no attempt was promotion_eligible; saving all attempts anyway so the "
              "demo bundle exists, but none of them will pass the trust gate if re-seeded.")

    save_bundle(bundle_dir=settings.demo_bundle_dir, attempts=captured, run_result=result)

    print(f"Saved demo bundle ({len(captured)} attempt(s)) to {settings.demo_bundle_dir}")
    for logged, _model in captured:
        r = logged.result
        print(
            f"  attempt {r.attempt_number} ({r.model_family}): test RMSE {r.test_metrics['rmse']:.2f}, "
            f"gate_passed={logged.gate.passed}"
        )
    print(f"promoted_attempt={result.promoted_attempt} stopped_reason={result.stopped_reason}")


if __name__ == "__main__":
    main()

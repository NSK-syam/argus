#!/usr/bin/env python3
"""Build the precomputed demo bundle: run the reflection loop once against
real FD001 data and save the winning model + metadata to
app/demo_bundle/, so the API can serve a known-good result even if live
training fails or takes too long during a demo.

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
    captured = {}

    def on_attempt(logged, model):
        # keep the LAST attempt's model in hand at every step; if the run
        # promotes, that's the winner. If it doesn't (shouldn't happen on
        # this dataset, but just in case), we still have the last attempt
        # as a reasonable fallback rather than nothing.
        captured["logged"] = logged
        captured["model"] = model

    print(f"Running the reflection loop against {settings.data_dir} ...")
    result = run_reflection_loop(settings.data_dir, use_claude=False, on_attempt=on_attempt)

    if not result.promoted:
        print("WARNING: no attempt was promotion_eligible; saving the last attempt's model anyway "
              "so the demo bundle exists, but it will NOT pass the trust gate if re-evaluated.")

    logged = captured["logged"]
    model = captured["model"]
    r = logged.result

    save_bundle(
        bundle_dir=settings.demo_bundle_dir,
        model=model,
        winning_attempt_number=r.attempt_number,
        feature_columns=r.feature_columns,
        background_sample=r.background_sample,
        conformal_q=r.conformal_q,
        model_family=r.model_family,
        run_result=result,
    )

    print(f"Saved demo bundle to {settings.demo_bundle_dir}")
    print(f"  attempt {r.attempt_number} ({r.model_family}), test RMSE {r.test_metrics['rmse']:.2f}, "
          f"promoted={result.promoted}")


if __name__ == "__main__":
    main()

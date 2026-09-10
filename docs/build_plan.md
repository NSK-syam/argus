# Argus — ABB Accelerator 2026 Build Plan

*Adopted verbatim as the canonical build plan (authored via Codex, refining the initial Claude draft in `docs/idea_phase_plan.md`). This repository's structure and today's implementation follow this document directly.*

## Summary

Build **Argus**, a solo-friendly agentic predictive-maintenance studio for ABB Theme 1. It will ingest industrial time-series data, create and revise an ML pipeline, explain predictions, and route models through a deterministic trust gate before deployment.

The demonstration will use **NASA C-MAPSS FD001** and replay a real held-out engine trajectory. Claude's draft supplies the competitive thesis; this plan narrows it into an achievable Next.js/FastAPI prototype.

Success means:

- Idea submitted by **September 13, 2026, noon Central**, ahead of the 4:59 PM cutoff shown on the [official hackathon page](https://www.hackerearth.com/community/challenges/hackathon/abb-accelerator-2026/).
- End-to-end prototype submitted by September 27.
- A judge can run the preloaded demonstration in under 90 seconds without uploading data or waiting for training.
- The agent visibly revises a failed pipeline at least once.
- Model promotion is governed by reproducible metrics, not an LLM decision.
- Deliverables include a live URL, repository, demo video, technical documentation, and optional deck.

## Product and Architecture

- Build a monorepo containing:
  - Next.js/TypeScript frontend with Tailwind, shadcn/ui, and Recharts.
  - FastAPI backend with Pydantic, SQLAlchemy, LangGraph, XGBoost, scikit-learn, Optuna, MLflow, and SHAP.
  - Docker Compose configuration for reproducible local execution.
- Provide five primary views:
  1. Goal and dataset selection.
  2. Data-quality profile and schema mapping.
  3. Live planner/trainer/critic attempt timeline.
  4. Model comparison, SHAP explanations, and trust-gate evidence.
  5. Deployment dashboard with held-out-engine replay.
- Use Claude Sonnet 5 (`claude-sonnet-5`) through the Anthropic API for structured pipeline proposals and plain-language explanations. Anthropic's current API supports schema-constrained outputs for this model ([official documentation](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)).
- Send Claude only schema summaries, metrics, and feature-importance values—not uploaded raw sensor rows. Display a structured decision journal rather than private chain-of-thought.
- Host the frontend on Vercel Hobby, the FastAPI service on Render's $7 `0.5c-512mb` plan, and metadata/artifacts on Supabase Free. Cap Anthropic usage at $15, keeping total operating cost below $25. The public backend runs one training job at a time and caps uploads at 10 MB/50,000 rows.
- Bundle a precomputed FD001 run and model artifact with the deployment so the main demo survives API, training, or hosting failures.

## Interfaces and Data Flow

- Persist these entities:
  - `Dataset`: schema, asset/cycle/target mappings, profile, storage reference.
  - `PipelineRun`: goal, task type, status, winning attempt, timestamps.
  - `Attempt`: structured plan, features, model family, parameters, metrics, gate failures.
  - `ModelVersion`: artifact, SHAP summary, conformal-calibration statistics, stage.
  - `Deployment`: model version, `shadow` or `production` stage, promotion audit record.
- Define a validated `PipelinePlan` containing task type, target, asset and cycle columns, preprocessing steps, feature windows, candidate model, search space, validation strategy, and rationale.
- Expose:
  - `POST /api/v1/datasets` for CSV upload or selection of the bundled FD001 dataset.
  - `POST /api/v1/runs` to start an AutoML run.
  - `GET /api/v1/runs/{id}` for state, attempts, metrics, and gate evidence.
  - `GET /api/v1/runs/{id}/events` as Server-Sent Events for progress.
  - `POST /api/v1/models/{id}/promote` for one-click human-confirmed promotion.
  - `POST /api/v1/predict` for production inference.
  - `POST /api/v1/replay/{engine_id}` and an SSE feed for the live engine trajectory.
- Persist job state before execution. A restart marks interrupted jobs failed with a retry option rather than silently losing them.

## Agentic ML Workflow

1. Profile data deterministically: schema, missingness, cardinality, constant sensors, asset timelines, leakage risks, and target availability.
2. Ask Claude for a schema-valid pipeline proposal restricted to an allowlist of supported transformations and models.
3. For FD001:
   - Derive RUL from engine maximum cycle and cap training targets at 125 cycles.
   - Remove constant sensors.
   - Generate raw values, rolling mean/std for 5/10/20 cycles, and 1/5-cycle lags.
   - Compare a mean-RUL baseline, Random Forest, and XGBoost.
   - Run at most eight Optuna trials across candidate models.
   - Validate by engine, never by randomly splitting rows; retain the official FD001 test set for final evaluation.
   - Report RMSE, MAE, NASA asymmetric score, and F1 for the `RUL ≤ 30` warning threshold.
4. Apply a deterministic trust gate:
   - No leakage or unresolved schema blockers.
   - At least 15% RMSE improvement over the baseline.
   - FD001 test RMSE no worse than 22 cycles.
   - Validation-to-test RMSE degradation no greater than 30%.
   - Split-conformal interval coverage at least 85%.
5. When a gate fails, pass only the failure evidence to Claude. Claude may select one allowed revision: change feature windows, remove unstable features, switch model family, or revise the bounded search space.
6. Allow two revisions, for three total attempts. Preserve every attempt in MLflow and the decision journal.
7. Passing models become `promotion_eligible`; failing models remain `shadow`. Production exposure always requires the one-click promotion action.
8. Generate global and per-prediction SHAP explanations. Claude converts verified SHAP values into concise maintenance language without inventing causes.
9. Replay a real held-out FD001 engine at one cycle per second, showing predicted RUL, interval, warning state, and top contributing sensors.
10. Defer drift detection until the core flow passes acceptance tests. If time remains, add a read-only PSI/KS alert using replayed shifted data; do not add automatic retraining.

## Delivery Schedule

- **Sep 10:** finalize idea copy, architecture diagram, repository skeleton, FD001 loader, and baseline.
- **Sep 11:** implement one deterministic train/evaluate/revise loop and capture evidence that it retries.
- **Sep 12:** prepare screenshots, a 30-45 second proof clip, originality/license notes, and submission copy.
- **Sep 13:** submit by noon Central and verify every saved field before the cutoff.
- **Sep 14-17:** finish ML pipeline, trust gate, structured Claude adapter, and precomputed demo artifact.
- **Sep 18-20:** build the Next.js workflow and SSE attempt timeline.
- **Sep 21-23:** add SHAP explanations, model promotion, inference endpoint, and engine replay.
- **Sep 24-25:** deploy, add Docker fallback, tests, documentation, and attribution.
- **Sep 26:** record the final 90-second demo and freeze the candidate release.
- **Sep 27:** perform clean-machine verification and submit by noon Central.

Cut in this order if schedule slips: drift monitoring, generic uploads, visual polish beyond the five core views. Never cut the working reflection loop, trust evidence, explanations, preloaded demo, or reproducible setup.

## Test Plan

- Unit-test RUL labeling, feature-window boundaries, engine-group splitting, leakage detection, metric calculations, plan-schema validation, retry limits, and every trust-gate condition.
- Verify an intentionally weak first attempt produces a permitted revision and a second logged attempt.
- Verify malformed Claude output, rate limits, and API outages fall back to a deterministic default plan and do not block the preloaded demo.
- Integration-test upload → profile → run → SSE updates → attempts → SHAP → eligibility → promotion → inference.
- Confirm failed/borderline models cannot reach production and all promotions create an audit record.
- Run end-to-end tests for the bundled demo, held-out-engine replay, API restart recovery, upload limits, and server-only secret handling.
- On a clean machine, verify Docker startup, README instructions, live URL, repository links, and the complete judge flow in under 90 seconds.

## Assumptions

- Working name remains **Argus**.
- The participant is eligible, solo, and can contribute 20-30 focused hours weekly.
- The project is an advisory prototype; it never controls physical equipment or makes autonomous maintenance decisions.
- C-MAPSS FD001 is the sole required dataset; generic uploads are retained only if the core demo is stable.
- Open-source dependencies and the NASA dataset receive explicit attribution.
- Uploaded data expires after 24 hours, and the interface warns users not to upload confidential plant data.
- The Claude plan is treated as strategic source material; its Streamlit frontend, synthetic bearing stream, unrestricted LLM critic, and separate deployment microservice are intentionally replaced.

---

## Implementation notes (added after Day 1 build)

- Today's Sep 10 checklist items are done and exceed scope: rather than just
  a "baseline," a full Planner→Trainer→Critic reflection loop is implemented
  and tested against real FD001 data, including a genuine (not staged)
  trust-gate failure and revision — see `docs/day1_status.md`.
- One calibration finding worth preserving: a full-featured first attempt
  (all sensors, sensible defaults) clears the trust gate immediately with no
  retry. The implemented `_next_plan()` in `orchestrator.py` deliberately
  starts attempt 1 with a minimal, genuinely defensible plan (linear
  regression on 4 classically-informative sensors) so the "visibly revises a
  failed pipeline" success criterion is met honestly rather than staged.

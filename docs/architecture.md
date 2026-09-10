# Argus — Architecture

Status: everything in the diagram below is implemented and tested against
the real NASA C-MAPSS FD001 dataset — the ML core, the FastAPI service
layer, SHAP explainability, MLflow logging, and the Next.js frontend, all
verified end to end (including a headless-browser run of the actual UI
against the actual running backend). See `docs/day1_status.md` for the
day-by-day log, including the honest, non-staged finding about how the
trust-gate thresholds were calibrated.

```
                     ┌─────────────────────────────────────────────┐
                     │              Next.js Frontend                 │
                     │  1) Goal & dataset selection                  │
                     │  2) Data-quality profile (part of the same view)│
                     │  3) Live planner/trainer/critic timeline (SSE)│
                     │  4) Model comparison, SHAP, trust-gate proof  │
                     │  5) Deployment dashboard + held-out replay    │
                     └───────────────────────┬─────────────────────┘
                                              │ REST + SSE
                     ┌───────────────────────▼─────────────────────┐
                     │           FastAPI Application Layer           │
                     │  /datasets  /runs  /runs/{id}/events (SSE)    │
                     │  /models/{id}/promote  /predict  /replay/{id} │
                     └───────────────────────┬─────────────────────┘
                                              │
          ┌───────────────────────────────────────────────────────────┐
          │      Reflection-loop Orchestrator (Planner→Trainer→Critic)  │
          │      backend/app/ml/pipeline/orchestrator.py                │
          │                                                             │
          │  Planner: deterministic fallback sequence, OR a live         │
          │  Claude adapter (schema-constrained tool use) tried first    │
          │  when ANTHROPIC_API_KEY is set — any failure (no key,        │
          │  network, malformed/unsafe response) falls back to the       │
          │  deterministic sequence automatically, logged with why        │
          │       ↓                                                      │
          │  Trainer: bounded model search (linear / random forest /     │
          │  xgboost), grouped cross-validation, split-conformal          │
          │  calibration                                                 │
          │       ↓                                                      │
          │  Critic = deterministic Trust Gate (trust_gate.py)           │
          │       ──(fail: revise plan, max 2 retries)──┐                │
          │       ↓                                       │              │
          │  Explainer (SHAP, global + per-prediction)  <──┘              │
          │       ↓                                                      │
          │  Deploy Router: human-confirmed promotion, refused (409)     │
          │  outright for any model that failed the gate                 │
          └───────────────────────────┬───────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                               │
┌───────▼────────┐          ┌──────────▼─────────┐          ┌──────────▼─────────┐
│  MLflow Tracking │          │  SQLAlchemy / DB     │          │  Held-out engine     │
│  every attempt    │          │  (Dataset,           │          │  replay: streams a   │
│  logged as its     │          │   PipelineRun,        │          │  real FD001 test     │
│  own run, never     │          │   Attempt,             │          │  engine cycle by     │
│  blocking a result   │          │   ModelVersion,        │          │  cycle through the   │
│                     │          │   Deployment)          │          │  promoted model       │
└───────────────────┘          └──────────────────────┘          └──────────────────────┘
```

Drift detection (post-deployment monitoring for distribution shift) remains
explicitly deferred — it's Tier 2 in the build plan and only worth adding
if time remains after the core flow and frontend are solid, which they now
are.

## Why the deterministic fallback came first, and why that decision paid off

The build plan requires that a malformed Claude response, a rate limit, or
an API outage fall back to a deterministic default plan and never block the
pipeline. Building that deterministic path *first* — rather than the live
Claude adapter — meant:

1. The reflection loop, trust gate, and metrics were provably correct
   (pytest, `backend/tests/`) independent of any LLM behavior, before any
   LLM was involved at all.
2. The live Claude Planner/Critic (`claude_planner.py`) is a genuine
   drop-in: same inputs (structured trust-gate evidence, never raw data),
   same output contract (a `PipelinePlan`-shaped revision from an
   allowlist), no architecture change to `orchestrator.py` beyond a
   try/except around which planner answered.
3. It means the whole system stays honest under judging conditions: if a
   judge runs this with no `ANTHROPIC_API_KEY` at all, they get the exact
   same reflection loop, the exact same honest retry, and the exact same
   trust-gate enforcement — the "agentic" claim doesn't evaporate the
   moment the LLM is unavailable.

## Data model (`backend/app/db/models.py`, SQLAlchemy 2.0)

- `Dataset` — schema, profile (engine counts, constant-sensor detection,
  leakage-risk notes), storage reference, upload expiry
- `PipelineRun` — goal, status, winning attempt, stopped reason, timestamps
- `Attempt` — structured plan, model family, hyperparameters, validation +
  test metrics, conformal coverage, gate evidence, gate_passed, revision
  rationale, plan_source (`deterministic_fallback` vs. Claude-proposed),
  fallback_reason, MLflow run id
- `ModelVersion` — artifact path, feature columns, conformal quantile, SHAP
  global-importance summary, trust_gate_passed, stage (shadow/production)
- `Deployment` — model version, stage, promoted_by, audit record (always
  `human_confirmed` — there is no auto-promotion path)

## Trust gate (`backend/app/ml/pipeline/trust_gate.py`)

| Condition | Threshold |
|---|---|
| No leakage / unresolved schema blockers | boolean |
| Improvement over mean-RUL baseline | ≥ 15% |
| FD001 official test RMSE | ≤ 22 cycles |
| Validation → test RMSE degradation | ≤ 30% relative |
| Split-conformal interval coverage | ≥ 85% |

All five are arithmetic on held-out metrics — no LLM is ever in this
decision path, by design (see the module's own docstring), and the FastAPI
promotion endpoint enforces it server-side (HTTP 409 on failure) regardless
of what any client asks for.

## API surface (`backend/app/api/*.py`)

`POST /api/v1/datasets` (bundled FD001) and `/datasets/upload` (generic
CSV, size/row-capped, profiling only — not yet wired to the training loop);
`POST /api/v1/runs`, `GET /api/v1/runs/{id}`, `GET /api/v1/runs/{id}/events`
(SSE), `POST /api/v1/runs/{id}/retry`; `POST /api/v1/models/{id}/promote`,
`GET /api/v1/models/{id}`; `POST /api/v1/predict`; `GET
/api/v1/replay/engines`, `GET /api/v1/replay/{model}/{engine}/events` (SSE).

## Deployment

`docker-compose.yml` brings up Postgres, a dedicated-Dockerfile MLflow
server (`mlflow/Dockerfile` — dependencies baked in at build time, not
installed at container start), the FastAPI backend, and the Next.js
frontend, each with its own Dockerfile. Verified against a real Docker
daemon (build + run + cross-container connectivity, including the
backend-to-mlflow path by its actual Docker-network hostname).

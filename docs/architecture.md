# Argus — Architecture

Status: the **ML core** (data loader, feature engineering, trust gate, and the
Planner→Trainer→Critic reflection loop) described below is implemented and
tested against the real NASA C-MAPSS FD001 dataset. The FastAPI service layer,
LangGraph wiring, MLflow/Postgres persistence, and the Next.js frontend are
scaffolded (Dockerfiles, compose, dependency list) but not yet built — that's
the Sep 14-23 portion of the build plan. See `docs/day1_status.md` for exactly
what runs today.

```
                     ┌─────────────────────────────────────────────┐
                     │              Next.js Frontend                 │
                     │  1) Goal & dataset selection                  │
                     │  2) Data-quality profile & schema mapping     │
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
          │         LangGraph Orchestrator (Planner→Trainer→Critic)     │
          │                                                             │
          │  IMPLEMENTED TODAY as a deterministic fallback sequence     │
          │  (backend/app/ml/pipeline/orchestrator.py) — same contract  │
          │  a live Claude adapter will plug into: propose a            │
          │  PipelinePlan, train, evaluate, and on trust-gate failure    │
          │  receive ONLY structured evidence (metrics + which checks    │
          │  failed) and select one allowlisted revision.                │
          │                                                             │
          │  Planner → Trainer (Optuna-style bounded search)            │
          │       → Critic = deterministic Trust Gate (trust_gate.py)   │
          │           ──(fail: revise, max 2 retries)──┐                │
          │       → Explainer (SHAP — not yet wired)  <──┘              │
          │       → Deploy Router (confidence-based — not yet wired)    │
          │       → Drift Monitor (deferred, Tier 2 in the build plan)  │
          └───────────────────────────┬───────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                               │
┌───────▼────────┐          ┌──────────▼─────────┐          ┌──────────▼─────────┐
│  MLflow Tracking │          │  PostgreSQL          │          │  Model Registry /   │
│  (compose stub,   │          │  (compose stub,      │          │  Deployed Inference  │
│   not yet wired   │          │   schema not yet      │          │  Service (not yet    │
│   into pipeline)  │          │   implemented)        │          │  implemented)         │
└───────────────────┘          └──────────────────────┘          └──────────────────────┘
```

## Why the deterministic fallback came first

The build plan requires that a malformed Claude response, a rate limit, or
an API outage fall back to a deterministic default plan and never block the
demo. Building that deterministic path *first* — rather than the live Claude
adapter — means:

1. The reflection loop, trust gate, and metrics are provably correct
   (pytest, `backend/tests/`) independent of any LLM behavior.
2. The live Claude Planner/Critic is a drop-in replacement for
   `_next_plan()` in `orchestrator.py` later: same inputs (structured trust-
   gate evidence), same output contract (a `PipelinePlan`-shaped revision
   from the same allowlist), no architecture change.
3. If Sep 13/14 arrives before the Claude adapter is wired in, there is
   already a real, working, tested pipeline to submit — not a plan.

## Data model (per the build plan; not yet implemented as SQLAlchemy models)

- `Dataset` — schema, asset/cycle/target mappings, profile, storage reference
- `PipelineRun` — goal, task type, status, winning attempt, timestamps
- `Attempt` — structured plan, features, model family, params, metrics, gate
  failures (this maps directly onto `LoggedAttempt` in `orchestrator.py`
  today, just not persisted yet)
- `ModelVersion` — artifact, SHAP summary, conformal-calibration stats, stage
- `Deployment` — model version, shadow/production stage, promotion audit
  record

## Trust gate (implemented, `backend/app/ml/pipeline/trust_gate.py`)

| Condition | Threshold |
|---|---|
| No leakage / unresolved schema blockers | boolean |
| Improvement over mean-RUL baseline | ≥ 15% |
| FD001 official test RMSE | ≤ 22 cycles |
| Validation → test RMSE degradation | ≤ 30% relative |
| Split-conformal interval coverage | ≥ 85% |

All five are arithmetic on held-out metrics — no LLM is in this decision
path, by design (see `trust_gate.py` module docstring).

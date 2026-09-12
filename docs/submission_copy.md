# Argus — ABB Accelerator 2026 Idea Phase Submission

Theme 1: Agentic Predictive Maintenance Studio — AutoML + MLOps Copilot for
Industrial Equipment. Ready to paste into the HackerEarth submission form;
sections below map to its fields.

---

## Project name

**Argus**

## One-line summary

An agentic AutoML copilot for predictive maintenance that plans, trains,
and critiques its own modeling pipeline — visibly revising a genuinely
failed attempt rather than shipping whatever it trains first — and refuses
to promote any model that doesn't pass a fixed, deterministic trust gate.

## Project summary (~1 page)

Industrial teams sit on huge volumes of sensor data, but turning it into a
trustworthy failure-prediction model normally needs a data scientist: clean
the data, choose an approach, engineer features, validate honestly, and
then keep watching the model as it drifts. Most maintenance teams don't
have that person on call for every asset class — so the data goes unused
and maintenance stays reactive.

Argus is a small, opinionated AutoML/MLOps copilot built around one idea:
an agent is only worth the name if it can notice its own pipeline is weak
and change course — not just run a fixed script and call the output final.

**What's actually built, not just planned:** the full loop runs today
against real data, not a mock. A **Planner** proposes a modeling plan (task
framing, feature spec, model family, search space); a **Trainer** executes
it with leakage-safe, per-engine feature engineering and grouped
cross-validation; a **Critic** scores the result against a fixed,
**deterministic** five-check trust gate — no unresolved leakage, ≥15%
RMSE improvement over a mean-RUL baseline, test RMSE ≤22 cycles, ≤30%
validation-to-test degradation, and ≥85% split-conformal interval coverage
— and if the model fails, the Critic hands back structured evidence (which
check failed and by how much) and the Planner revises: a different model
family, a different feature set. This is bounded (max two revisions) and
logged as its own MLflow run at every step.

**Methodology disclosure, stated plainly:** today the trust gate's RMSE and
coverage checks are computed against the official FD001 test set on *every*
attempt, and that same test-set score is what decides whether to retry or
promote. Reused this way across iterative decisions, it is evaluation-driven
model selection, not a single held-out check — so the specific RMSE numbers
below describe what this reflection loop actually does today, not a
leakage-free final evaluation. (This is distinct from the "leakage-safe,
per-engine feature engineering and grouped cross-validation" claim above,
which is about avoiding intra-engine leakage in feature computation and
remains accurate on its own terms.) Before the prototype-phase deadline we
are re-splitting the training engines into disjoint dev/calibration/gate-
validation partitions so every iterative decision uses only held-in data,
and touching the official test set exactly once, after the winning
configuration is fixed.

Run on the real NASA C-MAPSS FD001 turbofan dataset, this isn't staged: a
genuinely reasonable first attempt (linear regression on the four sensors
classical literature flags as most informative) fails the gate at 23.4
cycles RMSE; the Critic's revision (Random Forest, full sensor set) passes
at 18.4 cycles. Anyone can rerun it and get the same numbers — the repo
includes both a one-command CLI demo and a full web app that reproduces
this live. (As above: these are the current gate's own numbers, computed
against the official test set on every attempt, not a one-shot held-out
evaluation yet.)

Two features push this past a "profile → train → explain, once" pipeline
with a chatbot skin:

- **A live LLM planner with a fail-closed contract, not a fixed script.**
  Argus can hand planning to Claude (schema-constrained tool use — the
  model can only emit a validated, domain-checked pipeline plan, never
  free-form code) instead of the deterministic fallback sequence. If the
  API key is missing, the network fails, or the model proposes something
  unsafe, Argus **falls back to the deterministic planner automatically**
  and logs why — the agentic layer can degrade, but the pipeline never
  breaks and never silently does something unsafe.
- **Deterministic trust gate, not LLM-judged promotion.** No model — human
  or AI — decides "good enough" by vibes. Promotion to production is a
  human-confirmed action, and it is **refused outright (HTTP 409)** for any
  model that failed the five-check gate, regardless of who asks. The line
  between "an agent proposes" and "a fixed rule decides" is deliberate.

A judge doesn't have to wait through a live training run to see any of
this: the app seeds a **preloaded demo run** at startup from a bundled,
precomputed result — opening the app shows a one-click "already trained —
no waiting" path straight to the honest retry, promotion, and replay,
verified end to end at 7 seconds total. Starting a fresh run against live
training is also right there for anyone who wants to watch it happen from
scratch.

Once promoted, a model can be exercised two ways: **ad hoc prediction**
against arbitrary feature values, and — the demo's centerpiece — **live
replay of a real FD001 test engine**, streamed cycle by cycle through the
model via Server-Sent Events, each point carrying a conformal uncertainty
interval, a remaining-life warning flag, and a live SHAP explanation
narrated in plain language ("sensor_11 pushing predicted RUL up,
contribution +15.4 cycles..."). It's real test-engine data, not a synthetic
signal built to look good — though, per the disclosure above, that same
test set already informed model selection via the trust gate, so it is not
(yet) a held-out set in the strict sense.

**The stack, end to end, is real and tested:** a SQLAlchemy-backed FastAPI
service (datasets, SSE-streamed run progress, promotion enforcement,
prediction, replay), SHAP explainability (global and per-prediction),
MLflow experiment tracking, and a Next.js/TypeScript frontend covering all
of it — verified with a headless end-to-end browser run against the live
backend, not just unit tests. A Docker Compose stack (Postgres + MLflow +
backend) has been built and run against a real Docker daemon.

## Problem statement

Turning industrial sensor data into a trustworthy failure-prediction model
requires scarce data-science expertise for every step — data cleaning,
model selection, feature engineering, honest validation, and ongoing
monitoring as the model drifts. Most maintenance teams lack that expertise
for every asset class, so sensor data goes unused and maintenance stays
reactive instead of predictive — and the rare model that does get built
tends to be trusted (or distrusted) by instinct rather than evidence.

## Why this is different from the likely median Theme 1 submission

| A typical entry | Argus |
|---|---|
| Fixed pipeline: profile → train → explain, once | A Planner→Trainer→Critic loop that revises its own plan on a genuine failure |
| An LLM (or a person's gut) decides "good enough" | A fixed, five-check deterministic gate decides — the LLM only proposes, never approves |
| "Our model trained successfully" as the demo | An honest retry, reproducible by anyone, shown live: attempt 1 fails at 23.4 RMSE, attempt 2 passes at 18.4 (numbers from the current test-set-driven gate — see methodology disclosure above) |
| A synthetic stream tuned to look dramatic | Real FD001 test-engine replay, unmodified, with real conformal intervals (not yet a strictly held-out set — see disclosure above) |
| Chatbot bolted onto a script | An LLM adapter with a genuinely fail-closed contract — it can be swapped out entirely and nothing breaks |

## Technology stack

FastAPI + SQLAlchemy 2.0 (backend service), scikit-learn / XGBoost
(models), SHAP (explainability), MLflow (experiment tracking), Anthropic
Claude (schema-constrained planning/revision proposals via forced tool
use), Next.js 16 + TypeScript + Tailwind + Recharts (frontend), Postgres +
Docker Compose (deployment), pytest (test suite, unit + one real live-API
end-to-end test).

## Dataset

NASA C-MAPSS FD001 Turbofan Engine Degradation Simulation Data Set (A.
Saxena & K. Goebel, NASA Ames Prognostics Data Repository) — a public
research dataset, used here strictly for training/evaluation demonstration.
See **Originality, licensing & attribution** below.

## Proof of concept attached

- `docs/proof/argus_demo_40s.mp4` — a ~40-second screen recording (real
  browser, real backend, sped up ~4.6x from real time, no cuts or staging)
  for the idea-phase attachment: load the real FD001 dataset → start a run
  → attempt 1 fails the trust gate → attempt 2 passes it, with the real
  SHAP chart → promote → replay a real FD001 test engine (see the
  methodology disclosure above — not yet a strictly held-out evaluation).
- `docs/proof/argus_demo_90s.mp4` — the same recording at a gentler ~2x
  speed (~90s), for the prototype-phase demo slot.
- `docs/proof/argus_demo_full_180s.mp4` — the unmodified, real-time
  recording (~3 min, includes the actual ~45-60s model training wait) for
  anyone who wants to see it with nothing sped up.
- `docs/proof/argus_demo_walkthrough.gif` and `docs/proof/*.png` — the same
  moments as a lightweight GIF and individual full-resolution screenshots,
  for forms that don't accept video attachments.
- `backend/scripts/run_pipeline_demo.py` — anyone can rerun the reflection
  loop from a bare terminal and get the same honest retry, no UI required.

Most Theme 1 idea-phase submissions at this stage will be text only; this
one already has a running, rerunnable system behind it.

---

## Originality, licensing & attribution

- **Originality.** All source code — the reflection-loop orchestrator,
  trust gate, feature engineering, FastAPI service layer, SHAP/MLflow
  integration, Claude planner adapter, and the Next.js frontend — was
  written from scratch for this hackathon submission. No proprietary ABB
  code, data, or internal materials were used or viewed at any point.
- **Dataset.** Training and evaluation use NASA's C-MAPSS FD001 Turbofan
  Engine Degradation Simulation Data Set (A. Saxena, K. Goebel, "Turbofan
  Engine Degradation Simulation Data Set," NASA Ames Prognostics Data
  Repository), a public dataset intended for prognostics research and
  benchmarking. It is used here only for demo/evaluation and is not
  redistributed as part of the submitted source — the repo includes a
  download step, not the raw data files.
- **Third-party libraries.** All dependencies (FastAPI, SQLAlchemy,
  scikit-learn, XGBoost, SHAP, MLflow, Next.js, React, Tailwind CSS,
  Recharts, and their transitive dependencies) are open source under
  permissive licenses (MIT/BSD/Apache-2.0) and used per their terms; none
  are modified or redistributed beyond normal dependency use. Full pinned
  versions are in `backend/requirements.txt` and `frontend/package.json`.
- **LLM use.** Anthropic's Claude API is used at runtime, with the
  person's own API key, strictly as a schema-constrained planning
  component (forced tool use against a validated Pydantic schema) — it
  never sees raw sensor rows, never writes code that gets executed, and
  every one of its proposals is independently validated before use and
  independently graded by the deterministic trust gate before any
  promotion. This use is disclosed here and in `docs/architecture.md`.
- **No confidential or personal data.** No real ABB plant data, personally
  identifiable information, or confidential material of any kind appears
  anywhere in this submission.

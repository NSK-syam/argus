# ABB Accelerator 2026 — Idea Phase Submission Plan
**Theme 1: Agentic Predictive Maintenance Studio — AutoML + MLOps Copilot for Industrial Equipment**

Deadline: Idea Phase closes **Sep 13–14, 2026** · Prototype Phase: **Sep 18–27, 2026** · Onsite Grand Finale (Fort Smith, AR): **Oct 13–14, 2026**

Context: **~750+ teams registered.** This plan is built explicitly to stand out in that field, not just to check the theme's boxes.

---

## 0. The Strategy: How We Beat 750 Teams

With this many teams on the same two themes, most Theme 1 submissions will converge on the same shape: upload a CSV, run AutoML (probably PyCaret or auto-sklearn), show a SHAP plot, maybe bolt a chatbot on top and call it "agentic." That's a real risk — if we build *that*, we're just a well-executed version of the median submission.

Three things are designed specifically to separate this from the pack:

1. **Actually agentic, not agentic-flavored.** Most teams will build a fixed pipeline (profile → train → explain) with a chat UI in front of it — that's automation with a chatbot, not agency. Sentra's pipeline can **fail, notice it failed, and try something else on its own** (a reflection loop, §3.1). That's the literal definition of "agentic" the theme name uses, and it's the single hardest thing for a rushed team to fake — which is exactly why it's worth building.
2. **Production rigor most student projects skip.** Confidence-based auto-deploy routing and drift detection (§3.2, §3.3) are the kind of thing ABB's own engineers actually worry about in production — and reviewers who are ABB engineers will recognize the difference between "a model that runs" and "a model that's operationally trustworthy."
3. **A live demo moment, not a slideshow.** A simulated real-time sensor stream (§3.4) that visibly degrades, gets caught, gets explained, and triggers a deploy/no-deploy decision in front of the judges is more memorable than a static walkthrough of a dashboard — and memorability matters when judges are reviewing dozens of similar-looking AutoML tools back to back.

Everything below is scoped so a solo builder can actually ship it in the Prototype Phase window — the ambition is in *which* few things we build deep, not in doing more of everything.

---

## 1. Project Name

**Argus** — named for the all-seeing giant of myth. *An agentic ML engineer that watches your equipment, builds and second-guesses its own models, and tells you exactly when to trust — or not trust — what it's seeing.*

(Working name — easy to swap before final submission. "Sentra" was the earlier placeholder.)

---

## 2. Problem Statement

Industrial facilities collect huge volumes of sensor data (vibration, temperature, current, pressure), but turning that data into a working failure-prediction model normally requires a data scientist: someone to clean the data, pick a modeling approach, engineer features, tune hyperparameters, validate results, and then hand the model to an ops team to deploy — and then keep watching it, because models silently degrade as equipment ages and conditions drift. Most maintenance engineers don't have that skillset, and most industrial teams don't have a data scientist to spare for every asset class, let alone one to babysit every deployed model forever.

The result: valuable sensor data sits unused, maintenance stays reactive or calendar-based instead of predictive, and the rare models that *do* get built quietly go stale with nobody watching.

## 3. Proposed Solution

Argus is a platform where a maintenance engineer describes a goal in plain language ("predict bearing failure at least 48 hours in advance from vibration and temperature") or uploads sensor data, and an agentic pipeline designs, builds, explains, deploys, and **continues to watch** a failure-prediction model — without the engineer writing ML code, and without the engineer having to blindly trust a black box.

### 3.1 The core differentiator: a reflection loop, not a fixed pipeline

Instead of one straight line (profile → train → explain → deploy), Argus's orchestrator (LangGraph) runs a **plan → act → critique → retry** loop:

- The **Planner Agent** turns the goal or dataset into a modeling plan (task type, target, candidate approach).
- The **Trainer** executes it (AutoML search with LightGBM/XGBoost + Optuna).
- The **Critic Agent** looks at the result — cross-validated metrics, calibration, feature-importance sanity — and decides: *good enough to proceed*, or *not good enough, here's what to try differently* (different feature set, different model family, flag a data quality issue back to the Planner).
- This loops (bounded, e.g. max 3 iterations) until the Critic is satisfied or explicitly surfaces to the engineer *why* it's stuck — which is itself a useful, honest answer most tools never give.

This is the one feature almost no other Theme 1 submission will have, because it requires the pipeline to reason about its own output, not just execute a script — and it's a direct, literal answer to a theme called "**Agentic** Predictive Maintenance Studio."

### 3.2 Confidence-based deploy routing (trust, not just automation)

When the Critic is satisfied, Argus doesn't just auto-deploy blindly:

- **High confidence, stable metrics** → auto-deploy the model as a versioned inference service.
- **Borderline confidence, small data, or conflicting signals** → Argus deploys to a "shadow" endpoint and flags it for human review before it's promoted to the live decision path.

This mirrors how real industrial teams actually roll out models, and it's a concrete, demoable answer to "how do you know when to trust an AI system" — a question judges from an industrial automation company will almost certainly care about.

### 3.3 Drift detection (the part that separates a demo from a product)

Once deployed, Argus keeps a lightweight statistical check (e.g. population stability index / KS-test) on incoming data versus training data. If live sensor readings start drifting out of distribution, Argus flags it and proposes a retrain — closing the loop from "we shipped a model" to "we're operating a model," which is the actual definition of MLOps and something almost no hackathon submission bothers to include.

### 3.4 A live demo moment, not a static walkthrough

For the demo (and ideally for the idea-phase submission itself, see §9), Argus includes a **synthetic real-time sensor simulator** — a small script that streams synthetic vibration/temperature data that starts normal and drifts toward a failure signature. In the live demo:

1. Type the goal in chat → watch the Planner → Trainer → Critic loop reason out loud (summarized, not raw logs) → see it retry once before landing on a good model.
2. Deploy. Feed the simulated stream. Watch the risk score climb in real time on the dashboard, with a plain-language root-cause explanation ("risk is rising because `bearing_temp` and `vibration_rms` are both trending up together — this matches the failure pattern from your training data").
3. Show one high-confidence case that auto-deploys, and one borderline case that Argus flags for human review instead of guessing.
4. Show drift detection catching a distribution shift and proposing a retrain.

That's a 90-second story with a visible "aha" moment — far more memorable to a judge who's about to watch team #400 explain their leaderboard screenshot.

### Core capabilities (mapped to theme requirements)

| Theme requirement | Argus feature |
|---|---|
| Automated dataset profiling & quality assessment | Planner/Profiler step: schema detection, missing-value/outlier report, correlation heatmap, target-column suggestion |
| Intelligent task & model selection | Planner Agent chooses task framing and candidate model family, with reasoning shown |
| Data preprocessing & feature engineering | Auto-generated rolling stats, lag features, encodings — revised automatically by the Critic loop if results are weak |
| Model training & evaluation | AutoML Engine: LightGBM/XGBoost + Optuna search, cross-validated leaderboard |
| Explainable AI (feature importance, confidence) | SHAP global + per-prediction explanations, confidence bands, plain-language root-cause narration |
| Experiment tracking & comparison | MLflow-backed run history, including every reflection-loop retry, side-by-side comparison view |
| One-click deployment | Confidence-routed deploy: auto-promote or shadow-and-flag, containerized FastAPI service versioned in MLflow's registry |
| Interactive prediction/inference dashboard | Live dashboard fed by the real-time simulator: risk score, trend, explanation, deploy/flag decision log, drift alerts |

## 4. Why This Wins (vs. the likely median submission)

| A typical Theme-1 entry | Argus |
|---|---|
| Fixed pipeline: profile → train → explain, once | Plan → act → critique → retry loop that can change its own approach |
| Deploys whatever model it trained | Routes by confidence: auto-deploy vs. shadow-and-flag for human review |
| Ships and forgets the model | Monitors for drift post-deployment and proposes retraining |
| Static demo: dashboard + slides | Live simulated sensor stream with a visible real-time "catch and explain" moment |
| Chatbot bolted onto a script | Agents that actually alter the pipeline's own decisions based on their own critique |

## 5. System Architecture

```
                     ┌─────────────────────────────────────────────┐
                     │              Argus Web Dashboard              │
                     │  (goal/data input · copilot chat · leaderboard│
                     │   · SHAP views · live risk stream · alerts)   │
                     └───────────────────────┬─────────────────────┘
                                              │ REST / WebSocket
                     ┌───────────────────────▼─────────────────────┐
                     │           FastAPI Application Layer           │
                     └───────────────────────┬─────────────────────┘
                                              │
          ┌───────────────────────────────────────────────────────────┐
          │              LangGraph Agent Orchestrator                  │
          │                                                             │
          │   Planner Agent → Trainer (Optuna + LightGBM/XGBoost)       │
          │        → Critic Agent ──(retry: revise plan)──┐             │
          │        → Explainer Agent (SHAP)  <─────────────┘            │
          │        → Deploy Router (confidence-based)                   │
          │        → Drift Monitor (post-deployment)                    │
          └───────────────────────────┬───────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                               │
┌───────▼────────┐          ┌──────────▼─────────┐          ┌──────────▼─────────┐
│  MLflow Tracking │          │  PostgreSQL          │          │  Model Registry /   │
│  (experiments,    │          │  (datasets metadata, │          │  Deployed Inference  │
│   every retry,    │          │   sessions, drift     │          │  Service (Docker +   │
│   metrics)        │          │   logs)               │          │  FastAPI, versioned) │
└───────────────────┘          └──────────────────────┘          └──────────────────────┘
                                                                          ▲
                                                             ┌────────────┴────────────┐
                                                             │ Synthetic Sensor Stream  │
                                                             │ (real-time demo input)   │
                                                             └──────────────────────────┘
```

### Tech stack (per theme's suggested technologies)

- **Orchestration / agents:** LangGraph (agent graph + state machine with a retry/loop edge for the Critic), Python
- **AutoML:** LightGBM / XGBoost, Optuna for hyperparameter search
- **Explainability:** SHAP
- **Experiment tracking:** MLflow (including reflection-loop retries as linked runs)
- **Backend API:** FastAPI
- **Deployment:** Docker (containerized inference service; MLflow model registry for versioning)
- **Drift detection:** lightweight statistical check (PSI / KS-test) on a scheduled or streaming basis
- **Storage:** PostgreSQL (metadata, run history, pipeline/session state, drift logs)
- **Frontend:** Streamlit for the Prototype Phase MVP (fastest path to a working, demoable product); React/Next.js as a stretch goal only after the core loop is solid

## 6. Build Priorities for Prototype Phase (Sep 18–27)

Ranked so that if time runs short, we cut from the bottom — never the top. Technical Excellence (25%) and Innovation (20%) together are 45% of the score, and both live in Tier 1.

**Tier 1 — the differentiators (build first, protect at all costs):**
1. The Planner → Trainer → Critic reflection loop, actually looping at least once in the demo (this is the single most important thing to get working — it's the proof that this is agentic, not just automated)
2. SHAP explanation surfaced as plain-language narration, not a raw plot
3. Confidence-based deploy routing (auto-deploy vs. shadow-and-flag), visibly shown in the UI
4. Synthetic real-time sensor stream feeding the live dashboard

**Tier 2 — production rigor (build if Tier 1 is solid):**
5. Drift detection with a proposed-retrain flag
6. MLflow logging of every retry as a linked run, with a comparison view
7. One-click Docker deploy actually running as a callable inference endpoint

**Tier 3 — polish (only with time to spare):**
8. React/Next.js frontend instead of Streamlit
9. Multiple dataset/equipment templates
10. Auto-generated technical documentation from pipeline metadata

**Sample dataset:** a public predictive-maintenance dataset (e.g., NASA C-MAPSS turbofan degradation, or the AI4I 2020 Predictive Maintenance dataset) for training, paired with the synthetic streaming simulator (built on the same distribution) for the live demo — realistic enough to be convincing without needing real ABB plant data.

## 7. Judging Criteria Alignment

| Criteria | Weight | How this plan addresses it |
|---|---|---|
| Innovation & Creativity | 20% | A genuinely self-correcting agent loop (plan→act→critique→retry) plus confidence-based trust routing — not a fixed pipeline with a chatbot skin |
| Technical Excellence | 25% | Tiered build plan protects a fully-working core loop over a shallow pass at every possible feature; MLflow tracks every retry, not just final runs |
| Problem-Solution Fit | 20% | Every required capability in the theme brief is mapped to a specific Argus feature (table in §3), plus drift monitoring extends "predictive maintenance" to the model's own lifecycle |
| Scalability & Feasibility | 15% | Containerized microservices, MLflow model registry, Postgres-backed state, drift-triggered retraining — designed to run unattended across many asset types |
| User Experience | 10% | Plain-language narration of both predictions *and* the agent's own reasoning/retries, for non-ML maintenance engineers |
| Presentation & Demo | 10% | Live simulated real-time failure scenario with a visible catch-and-explain moment, not a static leaderboard screenshot |

## 8. Submission Deliverables Checklist (per hackathon Submission Format)

- [ ] **Project Summary** — condense §1–§5 to ~1 page
- [ ] **Working Prototype** — built during Prototype Phase (Sep 18–27), Tier 1 features functioning end-to-end
- [ ] **Demo Video** — the live synthetic-stream scenario from §3.4, not a slide-by-slide narration
- [ ] **Source Code Repository** — GitHub repo with clear README/setup instructions
- [ ] **Technical Documentation** — architecture, tech choices, setup guide, explanation of the reflection loop and confidence routing logic
- [ ] **Presentation Deck** (optional) — problem, solution, architecture, the "why this wins" contrast (§4), impact, roadmap

## 9. Timeline — and What to Do *Right Now*

| Phase | Dates | Goal |
|---|---|---|
| Idea Phase (current) | Aug 11 – Sep 13/14, 2026 | Submit this concept; if at all possible, attach a short demo clip of even a bare-bones version of the reflection loop — most competitors will submit text-only ideas, so *any* working proof-of-concept at this stage is disproportionately persuasive |
| Prototype Phase | Sep 18 – 27, 2026 | Build Tier 1 → Tier 2 → Tier 3 in that order, record the live demo, write docs, push code repo |
| Onsite Grand Finale | Oct 13 – 14, 2026 | If selected: live presentation at ABB's Fort Smith, AR facility — rehearse the live-stream demo so it's bulletproof in front of judges |

**Between now and Sep 13/14, the highest-leverage use of the remaining days** (given you're solo and comfortable coding) is a scrappy proof of the Planner→Trainer→Critic loop on a toy dataset — even a 2-minute terminal recording of it retrying once and improving is a stronger idea-phase submission than a polished doc with no code behind it, because it proves the hardest, most differentiating claim in this whole plan before anyone asks you to prove it.

## 10. Rules Compliance Notes

- Only one challenge theme per team submission (this plan commits to Theme 1)
- All work must be original and developed during the hackathon period
- Open-source libraries/frameworks are allowed with proper attribution (all listed stack components qualify)
- All deliverables must be submitted before each phase's deadline — no late/incomplete submissions

## 11. Next Steps

1. Trim §1–§5 into the Project Summary field on the HackerEarth submission form; use §4's contrast table sparingly in the summary itself, but keep it in your back pocket for the presentation deck.
2. Before Sep 13/14: build the smallest possible proof of the reflection loop (a script that trains, evaluates, and retries once with a different feature set/model if the first attempt is weak) and capture it as a short clip to attach to the idea submission if the form allows attachments.
3. Once Prototype Phase opens (Sep 18): scaffold the repo — FastAPI backend, LangGraph agent graph (Planner/Trainer/Critic/Explainer/Deploy Router/Drift Monitor nodes), MLflow tracking server, Postgres schema, the synthetic sensor simulator, and a Streamlit front end. Happy to start that scaffolding whenever you're ready — say the word.

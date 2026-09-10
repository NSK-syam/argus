# Argus — Agentic Predictive Maintenance Studio

ABB Accelerator 2026, Theme 1. See `docs/architecture.md` for the system
design and `docs/day1_status.md` for a day-by-day log of what's implemented
vs. scaffolded (short version: the whole core system is implemented and
tested, end to end, including the frontend — see "Status at a glance").

## Quickstart — full stack

```bash
# 1. fetch the real NASA C-MAPSS FD001 dataset (100 train + 100 test engines)
mkdir -p data/cmapss
base="https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData"
for f in train_FD001.txt test_FD001.txt RUL_FD001.txt; do
  curl -sL "$base/$f" -o "data/cmapss/$f"
done

# 2. backend
cd backend
python3 -m venv .venv && source .venv/bin/activate   # or use your own env
pip install -r requirements.txt
uvicorn app.main:app --reload          # http://localhost:8000, GET /health

# 3. frontend, in another terminal
cd frontend
cp .env.local.example .env.local       # NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm install
npm run dev                            # http://localhost:3000
```

Open `http://localhost:3000`: load the bundled FD001 dataset, start a run,
watch the Planner→Trainer→Critic loop revise itself live, promote the model
that passes the trust gate, then replay a held-out test engine cycle by
cycle with live SHAP explanations.

Or, no server required — just the reflection loop in a terminal:

```bash
cd backend
python3 scripts/run_pipeline_demo.py            # deterministic planner
python3 scripts/run_pipeline_demo.py --use-claude \
  # tries the live Claude planner first (needs ANTHROPIC_API_KEY);
  # falls back automatically and says why on any failure, including no key
python3 -m pytest -v                             # full test suite
```

Docker: `docker compose up` brings up Postgres + MLflow + the backend
(`mlflow/Dockerfile` and `backend/Dockerfile` both bake their dependencies
in at build time, verified against a real Docker daemon). The frontend
container isn't wired into `docker-compose.yml` yet — run it with `npm run
dev` per the quickstart above in the meantime.

Attribution: the C-MAPSS FD001 dataset is NASA's Turbofan Engine
Degradation Simulation Data Set (A. Saxena & K. Goebel, NASA Ames
Prognostics Data Repository), used here for demo/evaluation purposes only.

## Repository layout

```
backend/
  app/
    ml/
      data/       # FD001 loader, feature engineering, metrics
      pipeline/   # trust gate, reflection-loop orchestrator, Claude planner adapter,
                   # SHAP explainability, MLflow logging
    api/          # datasets, runs (+SSE), models/promote, predict, replay (+SSE)
    db/           # SQLAlchemy models + session
    services/     # background-threaded run orchestration
  scripts/
    run_pipeline_demo.py     # end-to-end CLI demo, no server required
    build_demo_artifact.py   # precomputes a demo bundle for resilience
  tests/          # unit + one comprehensive live-API end-to-end test
mlflow/           # dedicated Dockerfile for the local dev MLflow server
frontend/         # Next.js app: the 4 real views described in frontend/README.md
docs/
  architecture.md   # full system diagram + design rationale
  day1_status.md    # what's real, day by day, including honest findings
data/cmapss/        # NASA C-MAPSS FD001 (download step above; not committed)
docker-compose.yml  # postgres + mlflow + backend, verified against a real Docker daemon
```

## Status at a glance

Working today, all real and tested (not stubbed, not staged): FD001 data
loading with leakage-safe feature engineering; a deterministic five-check
model-promotion trust gate; a real Planner→Trainer→Critic reflection loop
that visibly revises a genuinely failed first attempt; a live Claude-backed
planner with a fail-closed fallback to the deterministic sequence; the full
FastAPI service (datasets, runs with SSE progress, promotion enforcement,
predict, held-out-engine replay with SSE); SHAP explainability (global and
per-prediction); MLflow experiment logging; a Next.js frontend exercising
every one of those endpoints, verified end to end with a headless browser
run against the real, running backend; and a Docker Compose stack verified
against a real Docker daemon.

Not yet built: generic CSV upload wired into the training loop (accepted
and stored today, but only the bundled FD001 dataset can start a run — a
deliberate Tier-2 scope cut) and drift detection (explicitly Tier-2 per the
plan). Neither blocks the Sep 13/14 idea-phase submission.

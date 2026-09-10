# Argus — Agentic Predictive Maintenance Studio

ABB Accelerator 2026, Theme 1. See `docs/architecture.md` for the system design
and `docs/day1_status.md` for exactly what's implemented vs. scaffolded.

## Quickstart (ML core — implemented today)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # or use your own env
pip install -r requirements.txt

# fetch the real NASA C-MAPSS FD001 dataset (100 train + 100 test engines)
mkdir -p ../data/cmapss
base="https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData"
for f in train_FD001.txt test_FD001.txt RUL_FD001.txt; do
  curl -sL "$base/$f" -o "../data/cmapss/$f"
done

# run the full Planner -> Trainer -> Critic reflection loop end-to-end
python3 scripts/run_pipeline_demo.py

# run the test suite (23 tests, ~90s — most of that is real model training)
python3 -m pytest -v
```

Attribution: the C-MAPSS FD001 dataset is NASA's Turbofan Engine
Degradation Simulation Data Set (A. Saxena & K. Goebel, NASA Ames
Prognostics Data Repository), used here for demo/evaluation purposes only.

## Repository layout

```
backend/
  app/
    ml/
      data/       # FD001 loader, feature engineering, metrics (implemented)
      pipeline/    # trust gate + reflection loop orchestrator (implemented)
      # api/, core/, db/ are scaffolded for the FastAPI service layer (Sep 14-17)
  scripts/
    run_pipeline_demo.py   # end-to-end CLI demo, no server required
  tests/           # 23 passing tests, see docs/day1_status.md
frontend/          # placeholder — Next.js app lands Sep 18-23
docs/
  architecture.md  # full system diagram + design rationale
  day1_status.md   # what's real today vs. scaffolded, and why
data/cmapss/       # NASA C-MAPSS FD001 (download step above; not committed)
docker-compose.yml # postgres + mlflow + backend, for local dev once wired up
```

## Status at a glance

Working today: FD001 data loading, leakage-safe feature engineering, a
deterministic model-promotion trust gate, and a real Planner→Trainer→Critic
reflection loop that retries on a genuine (not staged) trust-gate failure.
Run `python3 backend/scripts/run_pipeline_demo.py` to see it end-to-end.

Not yet built: the FastAPI service, live Claude-backed Planner/Critic, SHAP
explanations, MLflow/Postgres persistence, the Next.js frontend, and the
held-out-engine replay stream. None of that blocks the Sep 13/14 idea-phase
submission — see `docs/day1_status.md` for the schedule.

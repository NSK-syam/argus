# Third-party notices

Argus's own source code (the reflection-loop orchestrator, trust gate,
feature engineering, FastAPI service layer, SHAP/MLflow integration, Claude
planner adapter, and the Next.js frontend) is original and licensed under
the MIT license in `LICENSE`. It depends on the following open-source
software and one public research dataset, used per their own terms; none of
them are modified or redistributed beyond normal dependency use.

## Dataset

- **NASA C-MAPSS FD001 Turbofan Engine Degradation Simulation Data Set** —
  A. Saxena and K. Goebel, NASA Ames Prognostics Data Repository
  (https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
  A public dataset intended for prognostics research and benchmarking, used
  here strictly for training/evaluation demonstration. It is not
  redistributed as part of this repository — `data/cmapss/` is populated by
  a download step (see `README.md`), not committed raw data files.

## Backend (Python), see `backend/requirements.txt` for exact pinned versions

- FastAPI — MIT
- Uvicorn — BSD-3-Clause
- SQLAlchemy — MIT
- Pydantic — MIT
- scikit-learn — BSD-3-Clause
- XGBoost — Apache-2.0
- pandas — BSD-3-Clause
- NumPy — BSD-3-Clause
- SHAP — MIT
- MLflow — Apache-2.0
- pytest — MIT

## Frontend (Node), see `frontend/package.json` for exact pinned versions

- Next.js — MIT
- React / React DOM — MIT
- Tailwind CSS — MIT
- Recharts — MIT

## LLM use

Anthropic's Claude API is used at runtime, with the person's own API key, as
a schema-constrained planning component only (forced tool use against a
validated Pydantic schema). This use is disclosed here, in
`docs/submission_copy.md`, and in `docs/architecture.md`.

If any dependency's actual license terms differ from what's listed above by
the time you read this, the dependency's own repository/package metadata is
authoritative — this file is a good-faith summary, not a substitute for it.

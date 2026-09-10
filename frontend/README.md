# Argus frontend

Next.js 16 (App Router) + TypeScript + Tailwind v4 + Recharts, talking to
the real FastAPI backend — no mocked data anywhere in this app.

## Views

1. **Goal & dataset selection** (`/`) — loads the bundled FD001 dataset,
   shows its real data-quality profile (engine counts, constant-sensor
   detection, leakage-risk notes), takes a goal, and starts a run.
2. **Live planner/trainer/critic timeline** (`/runs/[runId]`) — consumes
   `GET /api/v1/runs/{id}/events` (SSE) while the run is in progress, then
   falls back to polling the persisted state once terminal. Renders every
   attempt's model family, metrics, and all five trust-gate checks with
   their real pass/fail detail strings — including the honest attempt-1
   failure / attempt-2 pass most runs produce.
3. **Model comparison & explainability** (same page) — one card per model
   version with its real SHAP global-importance bar chart and a
   promote-to-production action that the backend itself refuses (409) for
   any model that failed the trust gate — the button is disabled
   client-side too, but the server is the actual authority.
4. **Deployment & held-out engine replay** (`/runs/[runId]/replay`) —
   streams `GET /api/v1/replay/{model}/{engine}/events` (SSE) cycle by
   cycle through a real held-out FD001 test engine (not synthetic),
   plotting predicted vs. true RUL with the conformal interval band and a
   live per-cycle SHAP narration.

## Running locally

```bash
# backend, from repo root
cd backend && uvicorn app.main:app --reload

# frontend, in another terminal
cd frontend
cp .env.local.example .env.local   # NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm install
npm run dev
```

Or via `docker compose up` from the repo root once the frontend service is
added to `docker-compose.yml` (not yet wired in — see `docs/day1_status.md`).

## Verified

Built (`npm run build`), linted, and exercised end-to-end with a headless
Playwright run against the real backend (real FD001 training, real SSE
streams, real promote/predict/replay calls) — zero console errors, full
flow from dataset load through a promoted model through a completed replay.

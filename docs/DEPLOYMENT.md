# Deploying Argus to a live URL

Everything in this doc has been verified from this repo's committed files
(see the verification notes at the end) — but getting an actual `https://...`
URL up requires two accounts this environment isn't connected to: Render
(backend) and Vercel (frontend). Both have a permanent free tier and neither
needs a credit card to start. This doc is the complete checklist for
whoever has hands on those accounts.

## Why two services

The backend (FastAPI) and frontend (Next.js) are deployed separately:
- **Backend → Render**, from the committed `render.yaml` Blueprint, as a
  Docker web service.
- **Frontend → Vercel**, which auto-detects Next.js with zero config beyond
  one environment variable pointing at the deployed backend.

They don't have to be on these exact providers — the backend just needs
anything that can build a Dockerfile and expose a port, and the frontend is
a stock Next.js app — but this is the fastest zero-cost path.

## 1. Deploy the backend on Render

1. Go to <https://dashboard.render.com/blueprints> and sign in (GitHub login
   is enough, no card required for the free tier).
2. Click **New Blueprint Instance**, connect this GitHub repo, and select
   the branch to deploy. Render finds `render.yaml` at the repo root
   automatically and proposes one service: `argus-backend`.
3. Before the first deploy, open the service's **Environment** tab and,
   optionally, set `ANTHROPIC_API_KEY` (the repo's own planner falls back to
   a deterministic, non-LLM planning path automatically whenever this is
   unset — it's a real judge-usable app either way, just without the
   Claude-authored plan rationale). Nothing else needs to be set; every
   other env var Render needs (`DATABASE_URL`, `MLFLOW_TRACKING_URI`) is
   already in `render.yaml`.
4. Deploy. The first build takes a few minutes — it installs the pinned
   Python dependencies, then downloads the NASA C-MAPSS FD001 dataset and
   rebuilds the demo model bundle *during the image build* (see "What
   changed in the Dockerfile" below). Render then runs the container and
   polls `/health` until it's up.
5. Once live, note the backend's URL — something like
   `https://argus-backend-xxxx.onrender.com`. Confirm it works:
   ```
   curl https://argus-backend-xxxx.onrender.com/health
   curl https://argus-backend-xxxx.onrender.com/api/v1/runs/demo-seed-run
   ```
   The second call should come back with `"status": "succeeded"` and two
   attempts — that's the preloaded demo, already seeded at startup with no
   external data source needed.

**Free-tier tradeoffs, on purpose:** `render.yaml` pins `plan: free`, which
sleeps the service after 15 minutes idle (a cold request takes 30-60s to
wake it) and uses ephemeral disk. Ephemeral disk is fine here — the
preloaded demo reseeds itself from the baked-in bundle on every boot — but
any run a judge starts live (uploading new data, training a new model) is
lost on the next sleep/restart. That's an acceptable tradeoff for a
zero-cost judged demo; see "Optional: persistent storage" below to remove
it.

## 2. Deploy the frontend on Vercel

1. Go to <https://vercel.com/new> and sign in (GitHub login, no card
   required).
2. Import this repo. Vercel auto-detects Next.js. Set the **Root
   Directory** to `frontend` (this is a monorepo — the Next.js app isn't at
   the repo root).
3. Add one environment variable before deploying:
   - `NEXT_PUBLIC_API_BASE_URL` = the Render backend URL from step 1.5 above
     (e.g. `https://argus-backend-xxxx.onrender.com`, no trailing slash).

   This is a Next.js build-time public env var (`frontend/src/lib/api.ts`
   reads it, falling back to `http://localhost:8000` for local dev) — it
   gets baked into the client bundle, so it must be set *before* the first
   deploy, and any change to it requires a redeploy to take effect.
4. Deploy. No `vercel.json` is needed — Vercel's Next.js preset handles
   build and output automatically.
5. Once live, open the Vercel URL and confirm the "preloaded demo" banner
   on the home page loads and links through to `/runs/demo-seed-run`. If it
   doesn't, the most likely cause is `NEXT_PUBLIC_API_BASE_URL` pointing at
   the wrong host, or the Render service still asleep (first request will
   be slow — see the free-tier note above).

## What changed in the Dockerfile for this

`backend/Dockerfile` already worked for `docker-compose` (used throughout
local dev — see `docs/day1_status.md`), which bind-mounts `./data` from the
host into the container at `/data`. Render has no equivalent to that bind
mount, and the backend's default `data_dir` (`backend/app/core/config.py`)
resolves to `/data/cmapss` inside the image — so without a change, a Render
deploy would boot with no FD001 data at all, and the preloaded demo (whose
whole point is skipping any wait) would have nothing to seed from.

The fix: the Dockerfile now downloads the three FD001 CMAPSS files with
Python's stdlib `urllib` (deliberately not `curl` — `python:3.11-slim`
doesn't guarantee it's installed, and stdlib avoids an extra apt layer) and
re-runs `scripts/build_demo_artifact.py` during the image build itself, so
the built image is fully self-contained. Under `docker-compose` this step
is a harmless no-op — the bind mount already put real data at `/data`, so
the freshly-downloaded copy is simply overwritten by the mount when the
container starts.

**Verified in this sandbox** (2026-09-11, against a real local Docker
daemon — not a paper read-through): built the image end-to-end, ran a
container from it with *no volume mounts at all* and a bare sqlite
`DATABASE_URL`, and confirmed from the logs and API that:
- the container downloaded FD001 data and rebuilt the demo bundle during
  the build (not at runtime),
- `seed_demo_run()` ran at startup and logged `seeded preloaded demo run
  demo-seed-run (2 attempts)`,
- `GET /health` → `200 {"status": "ok", ...}`,
- `GET /api/v1/runs/demo-seed-run` → `status: succeeded`, 2 attempts, gate
  results `[False, True]` (the honest fail-then-pass retry), 2 model
  versions,
- `GET /api/v1/datasets` → the bundled FD001 dataset, 20,631 rows, 100
  engines — proving the baked-in data is real and queryable, not just
  present on disk.

This is the exact set of guarantees a Render deploy depends on, since
Render gives the container no volume mount either.

## Optional: persistent storage (Supabase Postgres)

The free-tier setup above uses sqlite on Render's ephemeral disk, so
anything beyond the preloaded demo (a judge's own live training run, say)
doesn't survive a restart. To fix that, swap in a real Postgres instance
and point both `DATABASE_URL` and (optionally) `MLFLOW_TRACKING_URI` at it
— `docker-compose.yml` already runs against Postgres locally, so no code
change is needed, just connection strings.

Two ways to get that Postgres instance:
- **Render Postgres** — add one from the Render dashboard (Render's own
  free Postgres tier expires after 30 days, so this is really a
  "months-not-forever" free option) and copy its internal connection string
  into `argus-backend`'s `DATABASE_URL`.
- **Supabase Postgres** — this session already has an authenticated
  Supabase connection, but only to pre-existing projects unrelated to this
  one (under a personal org with other real work in it). I didn't create a
  new project there without asking first. If you'd rather use Supabase,
  say the word and I'll provision a fresh project scoped to just this app,
  or you can create one yourself at <https://supabase.com/dashboard> and
  hand me the connection string.

Either way this is optional — the free sqlite setup is enough for a judge
to run the full preloaded-demo flow (retry evidence → promote → replay),
since that flow only *reads* the seeded data, and re-seeds automatically on
every boot regardless of what storage backend is behind it.

## Why this wasn't deployed automatically

Vercel and Render aren't connected to this session (checked via the MCP
connector registry — both show up as available integrations but not
authorized here), and I can't authorize new third-party service connections
on your behalf — that has to happen through your own claude.ai connector
settings or by signing into those dashboards directly. Everything above is
written so that once either account is connected, going live is a matter of
minutes, not further engineering.

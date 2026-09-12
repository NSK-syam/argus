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
3. Before the first deploy, open the service's **Environment** tab. Two
   env vars are `sync: false` in `render.yaml` and must be handled here:
   - `ARGUS_CORS_ORIGINS` — you don't know the Vercel origin yet, so leave
     it **unset for this first deploy** (the backend then defaults to `*`,
     open). You will come back and set it in step 3 below.
   - `ANTHROPIC_API_KEY` — **leave unset for a public demo.** The planner
     falls back to a deterministic, non-LLM path automatically, and the
     app is fully judge-usable without it. `render.yaml` also ships with
     `ARGUS_ENABLE_LIVE_RUNS=false` and `ARGUS_ENABLE_UPLOADS=false`, so a
     public URL can't be used to submit training jobs (and burn credits) or
     store arbitrary uploads; only the preloaded demo run is exposed. Set a
     key only on a private/judged deployment where you also flip
     `ARGUS_ENABLE_LIVE_RUNS` to `true` and accept that cost.
   Everything else (`DATABASE_URL`, `MLFLOW_TRACKING_URI`, the two safety
   flags) is already in `render.yaml`.
4. Deploy. The first build takes a few minutes — it installs the pinned
   Python dependencies, then downloads the NASA C-MAPSS FD001 dataset
   (pinned to a specific upstream commit and SHA-256-verified, see
   `backend/scripts/fetch_cmapss_data.py`) and rebuilds the demo model
   bundle *during the image build* (see "What changed in the Dockerfile"
   below). Render then runs the container and polls **`/ready`** until it
   passes. `/ready` is a real readiness check — DB reachable, FD001 files
   present, demo bundle present, demo run seeded, and a trust-gate-passing
   demo model whose artifact exists and loads — so a broken deploy fails
   the health check visibly instead of reporting "healthy" while a judge's
   first click would fail. (`/health` still exists as a plain liveness
   probe.)
5. Once live, note the backend's URL — something like
   `https://argus-backend-xxxx.onrender.com`. Confirm it works:
   ```
   curl https://argus-backend-xxxx.onrender.com/ready
   curl https://argus-backend-xxxx.onrender.com/api/v1/runs/demo-seed-run
   ```
   The first call should return `"status": "ready"` with every check
   `true`.
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
5. Once live, note the Vercel URL (e.g. `https://argus-xxxx.vercel.app`).

## 3. Lock CORS to the Vercel origin and redeploy the backend

The order matters: backend first (so the frontend has a URL to point at),
frontend second (so you know its origin), then this step.

1. Back in Render → `argus-backend` → **Environment**, set
   `ARGUS_CORS_ORIGINS` to the exact Vercel origin from step 2.5 — scheme
   and host, no path, no trailing slash, e.g.
   `https://argus-xxxx.vercel.app`. Comma-separate if you also want a
   preview/custom domain.
2. Redeploy the backend (Render prompts for this on env changes). Until
   this is done the backend accepts any origin, which works but isn't what
   you want on a public URL.
3. Open the Vercel URL and confirm the "preloaded demo" banner on the home
   page loads and links through to `/runs/demo-seed-run`, and that
   promote → predict → replay all work. If the page loads but API calls
   fail, the most likely causes are `NEXT_PUBLIC_API_BASE_URL` pointing at
   the wrong host, `ARGUS_CORS_ORIGINS` not matching the Vercel origin
   exactly, or the Render service still asleep (first request will be
   slow — see the free-tier note above). With live runs disabled, the
   "start a new run" action returns HTTP 403 by design.

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
- `GET /health` → `200 {"status": "ok", ...}` (this predates `/ready`,
  which is what `render.yaml` now polls; `/ready` was verified in a later
  container run — see `docs/day1_status.md`),
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

**Persistent DB + ephemeral disk is a real combination to get right.** A
seeded demo run stores its model artifacts as `.joblib` files under
`ARGUS_MODEL_ARTIFACT_DIR` on the container's disk. With a persistent
Postgres and Render's ephemeral disk, a redeploy keeps the demo run's DB
rows but wipes the files — an external code review pointed out that
`seed_demo_run()` used to return early on "row already exists" and never
recreate them, so `/predict` and `/replay` would fail after the first
redeploy. That's now handled: on every startup, any demo model artifact
missing from disk is re-dumped from the baked-in bundle, and `/ready` fails
(503) unless a trust-gate-passing demo model's artifact both exists and
loads. Live-run artifacts (a judge's own training run) still don't survive
a redeploy on ephemeral disk — that needs a Render persistent disk or
object storage, which is out of scope for the demo.

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

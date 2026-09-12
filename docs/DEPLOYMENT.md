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
2. Import this repo. Set the **Root Directory** to `frontend` (this is a
   monorepo — the Next.js app isn't at the repo root).

   Vercel now auto-detects this repo as a multi-service project (its
   **Services** preset, because there's a Next.js app *and* a FastAPI app)
   and then refuses to deploy without a root `vercel.json` describing both.
   Don't take that path: the backend is a long-lived stateful server (it
   writes SQLite and ~7 MB model artifacts to disk, runs training in
   background threads, and streams SSE), whereas Vercel services are
   per-request with an ephemeral filesystem, so its startup work would
   re-run on every cold start. Set the **Application Preset** to
   **Next.js** instead and host the backend on Render or a Hugging Face
   Space (below).

   `frontend/vercel.json` pins `"framework": "nextjs"` so this can't
   regress: with the preset left at "Other", the build itself succeeds and
   then fails at the last step with *No Output Directory named "public"
   found* — Vercel looking for a static site instead of reading Next.js's
   own output.
3. Point the frontend at your backend. `frontend/.env.production` is
   committed with the Hugging Face Space URL as the default, so a fresh
   deploy works with no dashboard configuration at all. To use a different
   backend (your own Render service, say), either edit that file or set
   `NEXT_PUBLIC_API_BASE_URL` in Vercel's project settings — a real
   environment variable takes precedence over the committed file.

   **This value is inlined into the client bundle at build time**, so it
   must exist *before* the build, and changing it requires a rebuild, not
   just a restart. That's a genuine footgun: the first Vercel deploy of
   this repo built without it, fell back to the local-dev default, and the
   deployed page sat there calling `http://localhost:8000` — the page
   rendered fine and every API call failed with `ERR_CONNECTION_REFUSED`.
   Committing the production default is what stops that recurring.

   Note also that `.env.local` (gitignored, used for local dev) outranks
   `.env.production` in Next.js's load order, so a local production build
   will pick up localhost while Vercel — which has no `.env.local` — picks
   up the committed file. To reproduce a deploy build locally, move
   `.env.local` aside first.
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

## Alternative backend host: Hugging Face Spaces (no Render account needed)

If Render isn't an option, a **Gradio** Space on Hugging Face runs the
backend for free with no card. Two HF constraints shape this path: Docker
Spaces and `cpu-basic` Gradio Spaces now need a PRO subscription, so a
free account gets the **ZeroGPU** Gradio tier, whose image is pinned to
Python 3.10 (the `python_version` README key is ignored there). Four of
the backend's pins need 3.11, so `deploy/huggingface/requirements-py310.txt`
lowers just those four, and the launcher rebuilds the demo bundle at
startup because scikit-learn differs from the version the committed bundle
was pickled with (verified in a real Python 3.10 container: identical
23.43 / 18.42 RMSE numbers, `/ready` all true). `deploy/huggingface/hf_space.py` is a
launcher that does at startup what the Dockerfile does at build time --
fetch the pinned, checksummed FD001 files -- then lets Gradio own the
server (that's what the HF runner expects) and attaches the same
`app.main:app` to it under **`/backend`** (Gradio's own `/api/*` routes
would otherwise collide), with `/ready` and `/health` also at the root and
a small landing page at `/`. Three HF-runner specifics it handles, each
found the hard way: ZeroGPU refuses to start unless some Gradio event is
wired to a `@spaces.GPU` function (a hidden no-op button); Gradio's own
CORS middleware would stack headers on top of the backend's, so the
backend is passed as Gradio's "parent app" and `ARGUS_CORS_ORIGINS` is the
single source of truth; and Gradio's Node SSR front on Spaces answers
unknown paths itself, so SSR is disabled. The
public-demo safety defaults are set in the launcher (overridable in the
Space's Settings → Variables). `deploy/huggingface/build_space.sh`
assembles the Space checkout from `backend/`.

1. On huggingface.co: **New → Space**, name it (e.g. `argus-backend`),
   SDK **Gradio** → **Blank** template, visibility Public. A free account
   is offered ZeroGPU hardware; that's fine (the app never touches the
   GPU, it only gets the container).
2. Create a **write** access token at huggingface.co/settings/tokens.
3. From any machine with Python (`pip install huggingface_hub`; plain
   `git push` is rejected for the 7 MB model pickle unless git-lfs is set
   up, the API upload handles that):
   ```bash
   mkdir space && deploy/huggingface/build_space.sh space
   HF_TOKEN=<write token> python3 -c "from huggingface_hub import HfApi; \
     HfApi().upload_folder(folder_path='space', repo_id='<user>/argus-backend', \
     repo_type='space', delete_patterns=['*'])"
   ```
4. Watch the build in the Space's **Logs** tab (first build a few minutes:
   pip install of the pinned requirements, then the FD001 fetch and demo
   seed on first start). When it's running, the backend is at
   `https://<user>-argus-backend.hf.space` -- check `/ready` there.
5. Use that URL as `NEXT_PUBLIC_API_BASE_URL` on Vercel (step 2 above),
   then set `ARGUS_CORS_ORIGINS` in the Space's **Settings → Variables**
   to the Vercel origin and restart the Space (step 3 above).

Free Spaces sleep after 48h idle and have ephemeral disk; the preloaded
demo re-seeds on every boot, same as Render.

**Live (2026-09-12):** `https://nsk1718-argus-backend.hf.space/backend` --
`/ready` all true; promote (409 for the failing model, 200 for the
passing one), predict with SHAP explanation, 100 replay engines with a
working SSE stream, and both safety gates (403 on live runs and uploads)
verified against the public URL. Frontend live at
`https://argus-five-weld.vercel.app`, with the whole flow (retry evidence
-> promote -> replay with conformal intervals) clicked through in a real
browser against that backend.

**CORS on `*.hf.space` cannot be enforced by response headers.** The
Space sits behind a proxy that echoes whatever `Origin` it is given: a
request claiming `Origin: https://evil.example.com` comes back with
`access-control-allow-origin: https://evil.example.com` and an
`access-control-expose-headers: *` this app never sets, on every path
including Gradio's own root. The app's `CORSMiddleware` applies
`ARGUS_CORS_ORIGINS` correctly underneath, but a browser only ever sees
the outermost header, so on that host the setting alone is decorative.
That is why `enforce_allowed_origin` (app/main.py) *refuses* a request
whose `Origin` is outside the allowlist (403) rather than just omitting a
header -- enforcement a downstream proxy can't undo. It inspects only
`Origin`, so curl and server-to-server callers are unaffected, and it is
a no-op while `ARGUS_CORS_ORIGINS` is unset or `*`.

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

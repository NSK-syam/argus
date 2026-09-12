FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# app/core/config.py's default data_dir is REPO_ROOT/data/cmapss, which
# resolves to /data/cmapss in this image (WORKDIR /app -> REPO_ROOT is
# /app's parent, "/"). docker-compose overrides this with a bind mount of
# the host's ./data (see docker-compose.yml), so this download is a no-op
# there -- it exists so the image is self-contained on a platform with no
# volume mount available (e.g. Render), which otherwise has no FD001 data
# to seed the preloaded demo or start a run.
#
# Fetches from a pinned commit SHA (not a mutable branch ref) and verifies
# each file's SHA-256 before it's used for anything -- found in external
# code review that a build pulling from someone else's unpinned "master"
# has no integrity guarantee: the maintainer force-pushing, the repo being
# deleted/renamed, or a compromised upstream would all silently change
# what gets baked into the image. See scripts/fetch_cmapss_data.py for the
# pinned SHA and expected hashes.
RUN python3 scripts/fetch_cmapss_data.py /data/cmapss && \
    python3 scripts/build_demo_artifact.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

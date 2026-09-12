#!/usr/bin/env bash
# Assembles a Hugging Face Space checkout from this repo's backend/.
# Usage: deploy/huggingface/build_space.sh <path-to-space-clone>
set -euo pipefail
dest="${1:?destination Space checkout}"
root="$(cd "$(dirname "$0")/../.." && pwd)"

# copy the backend as the Space root (Dockerfile must be at the root)
python3 - "$root/backend" "$dest" <<'PY'
import shutil, sys, pathlib
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
skip = {'.venv', '__pycache__', '.pytest_cache', 'artifacts', '.env', '.git', 'README.md'}
for child in dst.iterdir():
    if child.name != '.git':
        shutil.rmtree(child) if child.is_dir() else child.unlink()
def ignore(d, names):
    return [n for n in names if n in skip or n.endswith('.db')]
shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
PY

cp "$root/deploy/huggingface/README.md" "$dest/README.md"

# Public-demo safety defaults baked in as image ENV so the Space is safe
# even before anyone opens its Settings. Same flags render.yaml sets.
python3 - "$dest/Dockerfile" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
env = (
    "\n# Hugging Face Space public-demo defaults (mirrors render.yaml)\n"
    "ENV ARGUS_ENABLE_LIVE_RUNS=false \\\n"
    "    ARGUS_ENABLE_UPLOADS=false \\\n"
    "    DATABASE_URL=sqlite:////app/argus.db \\\n"
    "    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db\n"
)
if "ARGUS_ENABLE_LIVE_RUNS" not in s:
    s = s.replace('\nCMD ["uvicorn"', env + '\nCMD ["uvicorn"')
p.write_text(s)
PY
echo "Space assembled at $dest"

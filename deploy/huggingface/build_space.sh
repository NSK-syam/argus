#!/usr/bin/env bash
# Assembles a Hugging Face *Gradio* Space checkout from this repo's backend/.
# Usage: deploy/huggingface/build_space.sh <path-to-space-clone>
set -euo pipefail
dest="${1:?destination Space checkout}"
root="$(cd "$(dirname "$0")/../.." && pwd)"

python3 - "$root/backend" "$dest" <<'PY'
import shutil, sys, pathlib
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
skip = {'.venv', '__pycache__', '.pytest_cache', 'artifacts', '.env', '.git',
        'README.md', 'Dockerfile', 'tests', 'conftest.py', 'pytest.ini'}
for child in dst.iterdir():
    if child.name != '.git':
        shutil.rmtree(child) if child.is_dir() else child.unlink()
def ignore(d, names):
    return [n for n in names if n in skip or n.endswith('.db')]
shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
PY

cp "$root/deploy/huggingface/README.md" "$dest/README.md"
cp "$root/deploy/huggingface/hf_space.py" "$dest/hf_space.py"
# the launcher needs gradio; everything else is the backend's own pins
grep -q '^gradio==' "$dest/requirements.txt" || printf '\n# Space launcher (deploy/huggingface/hf_space.py)\ngradio==6.27.0\n' >> "$dest/requirements.txt"
echo "Space assembled at $dest"

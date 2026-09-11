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
# Uses python3's stdlib urllib instead of curl: python:3.11-slim doesn't
# guarantee curl is installed, and stdlib avoids an extra apt-get layer.
RUN mkdir -p /data/cmapss && \
    python3 -c "\
import urllib.request; \
base = 'https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData'; \
files = ['train_FD001.txt', 'test_FD001.txt', 'RUL_FD001.txt']; \
[urllib.request.urlretrieve(f'{base}/{f}', f'/data/cmapss/{f}') for f in files]" && \
    python3 scripts/build_demo_artifact.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

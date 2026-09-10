# Dedicated image for the local dev MLflow tracking server.
#
# The previous docker-compose setup installed mlflow/psycopg2-binary at
# *container start* time via a plain python:3.11-slim image and an inline
# `pip install && mlflow server ...` command. That works, but it re-downloads
# and reinstalls every dependency on every `docker compose up`, is slower,
# and (like any runtime pip install) is more exposed to registry hiccups or
# network policy differences between environments. Baking the dependencies
# into the image at build time is the standard, more robust pattern -- the
# image is immutable and starts instantly.
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "postgresql://argus:argus@postgres/argus", \
     "--default-artifact-root", "/mlflow/artifacts"]

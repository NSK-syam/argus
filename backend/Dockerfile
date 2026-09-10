FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The FastAPI app (app/main.py) lands Sep 14-17 per the build plan; until
# then this image is used to run the ML pipeline scripts/tests.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Argus frontend (placeholder)

Not yet built — scheduled Sep 18-23 in the build plan. Will be a Next.js +
TypeScript app (Tailwind, shadcn/ui, Recharts) with five views:

1. Goal and dataset selection
2. Data-quality profile and schema mapping
3. Live planner/trainer/critic attempt timeline (consuming the backend's
   `GET /api/v1/runs/{id}/events` SSE stream)
4. Model comparison, SHAP explanations, and trust-gate evidence
5. Deployment dashboard with held-out-engine replay

Until this exists, `backend/scripts/run_pipeline_demo.py` is the
server-less way to see the pipeline run end-to-end from a terminal.

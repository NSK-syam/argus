// Thin client for the Argus backend. No auth, no retries beyond the
// browser default -- this is an idea-phase demo frontend against a
// single-tenant local API, not a production client SDK.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

export interface DatasetProfile {
  n_rows: number;
  n_engines: number;
  constant_sensors: string[];
  near_constant_sensors: string[];
  engine_life_stats: { min_cycles: number; max_cycles: number; mean_cycles: number };
  leakage_risk_notes: string[];
}

export interface Dataset {
  id: string;
  name: string;
  source: string;
  n_rows: number;
  profile: DatasetProfile;
  created_at: string;
  expires_at: string | null;
  warning?: string;
}

export interface GateCheck {
  name: string;
  passed: boolean;
  detail: string;
}

export interface Attempt {
  attempt_number: number;
  model_family: string;
  hyperparams: Record<string, unknown> | null;
  validation_metrics: Record<string, number> | null;
  test_metrics: Record<string, number> | null;
  conformal_coverage: number | null;
  gate: { passed: boolean; checks: GateCheck[] } | null;
  gate_passed: boolean;
  revision_action: string | null;
  revision_rationale: string | null;
  plan_source: string;
  fallback_reason: string | null;
  mlflow_run_id: string | null;
}

export interface ModelVersionSummary {
  id: string;
  attempt_number: number;
  stage: "shadow" | "production";
  trust_gate_passed: boolean;
}

export interface FeatureImportance {
  feature: string;
  mean_abs_shap: number;
}

export interface ModelVersion extends ModelVersionSummary {
  run_id: string;
  model_family: string;
  feature_columns: string[];
  conformal_q: number;
  shap_summary: { global_importance: FeatureImportance[] } | null;
  created_at: string;
}

export interface Explanation {
  base_value: number;
  contributions: { feature: string; shap_value: number }[];
  narration: string;
}

export interface PredictResponse {
  predicted_rul: number;
  interval: [number, number];
  warning: boolean;
  warning_threshold: number;
  model_stage: string;
  explanation: Explanation | null;
}

export interface ReplayCycle {
  engine_id: number;
  time_cycles: number;
  predicted_rul: number;
  interval: [number, number];
  true_rul_estimate: number | null;
  warning: boolean;
  narration: string | null;
}

export interface PipelineRun {
  id: string;
  dataset_id: string;
  goal: string;
  status: "pending" | "running" | "succeeded" | "failed";
  winning_attempt_number: number | null;
  stopped_reason: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  attempts: Attempt[];
  model_versions: ModelVersionSummary[];
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ? JSON.stringify(body.detail) : detail;
    } catch {
      // ignore
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

export const api = {
  createBundledDataset: () =>
    fetch(`${API_BASE}/api/v1/datasets?source=bundled_fd001`, { method: "POST" }).then((r) =>
      json<Dataset>(r)
    ),
  listDatasets: () => fetch(`${API_BASE}/api/v1/datasets`).then((r) => json<Dataset[]>(r)),
  getDataset: (id: string) => fetch(`${API_BASE}/api/v1/datasets/${id}`).then((r) => json<Dataset>(r)),

  startRun: (datasetId: string, goal: string) =>
    fetch(`${API_BASE}/api/v1/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId, goal }),
    }).then((r) => json<PipelineRun>(r)),
  getRun: (id: string) => fetch(`${API_BASE}/api/v1/runs/${id}`).then((r) => json<PipelineRun>(r)),
  listRuns: () => fetch(`${API_BASE}/api/v1/runs`).then((r) => json<PipelineRun[]>(r)),
  retryRun: (id: string) =>
    fetch(`${API_BASE}/api/v1/runs/${id}/retry`, { method: "POST" }).then((r) => json<PipelineRun>(r)),

  promoteModel: (modelVersionId: string) =>
    fetch(`${API_BASE}/api/v1/models/${modelVersionId}/promote`, { method: "POST" }).then((r) =>
      json<ModelVersion & { promoted_by: string }>(r)
    ),
  getModel: (modelVersionId: string) =>
    fetch(`${API_BASE}/api/v1/models/${modelVersionId}`).then((r) => json<ModelVersion>(r)),

  predict: (modelVersionId: string, features: Record<string, number>) =>
    fetch(`${API_BASE}/api/v1/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_version_id: modelVersionId, features }),
    }).then((r) => json<PredictResponse>(r)),

  listReplayEngines: () =>
    fetch(`${API_BASE}/api/v1/replay/engines`).then((r) => json<{ engines: number[]; count: number }>(r)),
};

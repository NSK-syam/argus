"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { api, Attempt, ModelVersion, PipelineRun } from "@/lib/api";
import { streamSse } from "@/lib/sse";
import { Badge, Button, Card, MetricPill, Spinner } from "@/components/ui";

export default function RunPage() {
  const params = useParams<{ runId: string }>();
  const runId = params.runId;
  const router = useRouter();

  const [run, setRun] = useState<PipelineRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelDetails, setModelDetails] = useState<Record<string, ModelVersion>>({});
  const [promoting, setPromoting] = useState<string | null>(null);
  const stopRef = useRef<(() => void) | null>(null);

  // initial fetch
  useEffect(() => {
    api.getRun(runId).then(setRun).catch((e) => setError(String(e)));
  }, [runId]);

  // live SSE while the run is not terminal
  useEffect(() => {
    if (!run || run.status === "succeeded" || run.status === "failed") return;
    if (stopRef.current) return;

    stopRef.current = streamSse(
      `${process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"}/api/v1/runs/${runId}/events`,
      {
        onEvent: (event, data) => {
          if (event === "attempt") {
            const attempt = JSON.parse(data) as Attempt;
            setRun((prev) =>
              prev
                ? {
                    ...prev,
                    attempts: [
                      ...prev.attempts.filter(
                        (a) => a.attempt_number !== attempt.attempt_number
                      ),
                      attempt,
                    ].sort((a, b) => a.attempt_number - b.attempt_number),
                  }
                : prev
            );
          } else if (event === "run_status") {
            const parsed = JSON.parse(data);
            if (parsed.status === "succeeded" || parsed.status === "failed") {
              setRun(parsed as PipelineRun);
              stopRef.current?.();
            } else {
              setRun((prev) => (prev ? { ...prev, status: parsed.status } : prev));
            }
          } else if (event === "error") {
            setError(JSON.parse(data).detail);
          }
        },
        onError: (e) => setError(String(e)),
      }
    );

    return () => {
      stopRef.current?.();
      stopRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.status, runId]);

  // once terminal, fetch full model-version detail (feature columns + SHAP) for each
  useEffect(() => {
    if (!run || run.model_versions.length === 0) return;
    run.model_versions.forEach((mv) => {
      if (modelDetails[mv.id]) return;
      api.getModel(mv.id).then((detail) =>
        setModelDetails((prev) => ({ ...prev, [mv.id]: detail }))
      );
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.model_versions.length]);

  async function promote(modelVersionId: string) {
    setPromoting(modelVersionId);
    setError(null);
    try {
      const updated = await api.promoteModel(modelVersionId);
      setModelDetails((prev) => ({ ...prev, [modelVersionId]: updated }));
      setRun((prev) =>
        prev
          ? {
              ...prev,
              model_versions: prev.model_versions.map((mv) =>
                mv.id === modelVersionId ? { ...mv, stage: "production" } : mv
              ),
            }
          : prev
      );
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setPromoting(null);
    }
  }

  if (!run) {
    return (
      <div className="mx-auto max-w-6xl px-4 sm:px-6 py-10">
        {error ? (
          <div className="text-danger text-sm">{error}</div>
        ) : (
          <div className="flex items-center gap-2 text-muted text-sm">
            <Spinner /> Loading run…
          </div>
        )}
      </div>
    );
  }

  const productionModel = run.model_versions.find((mv) => mv.stage === "production");

  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6 py-10 space-y-8">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Pipeline run</h1>
          <p className="text-sm text-muted mt-1 max-w-2xl">{run.goal}</p>
          <p className="text-xs text-muted mono mt-1">{run.id}</p>
        </div>
        <StatusBadge status={run.status} />
      </div>

      {error && (
        <div className="rounded-md border border-[#5a2c2c] bg-danger-dim px-4 py-3 text-sm text-danger">
          {error}
        </div>
      )}

      {run.status === "failed" && (
        <Card className="border-[#5a2c2c]">
          <div className="text-sm text-danger">
            Run failed{run.error ? `: ${run.error}` : "."} Every attempt made
            it exhausted the two-revision budget without clearing the trust
            gate — nothing was silently promoted.
          </div>
          <Button
            variant="secondary"
            className="mt-3"
            onClick={() => api.retryRun(run.id).then((r) => router.push(`/runs/${r.id}`))}
          >
            Retry with a fresh run
          </Button>
        </Card>
      )}

      <section className="space-y-4">
        <h2 className="font-medium">Planner → Trainer → Critic timeline</h2>
        <div className="space-y-4">
          {run.attempts.length === 0 && (
            <div className="flex items-center gap-2 text-sm text-muted">
              <Spinner /> Planning attempt 1…
            </div>
          )}
          {run.attempts.map((attempt) => (
            <AttemptCard key={attempt.attempt_number} attempt={attempt} />
          ))}
          {run.status === "running" && run.attempts.length > 0 && (
            <div className="flex items-center gap-2 text-sm text-muted pl-1">
              <Spinner /> Working on attempt {run.attempts.length + 1}
              {run.attempts.length >= 3 ? "" : "…"}
            </div>
          )}
        </div>
      </section>

      {run.model_versions.length > 0 && (
        <section className="space-y-4">
          <h2 className="font-medium">Model versions &amp; explainability</h2>
          <div className="grid gap-4 md:grid-cols-2">
            {run.model_versions.map((mv) => (
              <ModelCard
                key={mv.id}
                summary={mv}
                detail={modelDetails[mv.id]}
                onPromote={() => promote(mv.id)}
                promoting={promoting === mv.id}
              />
            ))}
          </div>
        </section>
      )}

      {productionModel && (
        <div className="flex justify-end">
          <Button onClick={() => router.push(`/runs/${run.id}/replay?model=${productionModel.id}`)}>
            Go to deployment &amp; replay →
          </Button>
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: PipelineRun["status"] }) {
  if (status === "succeeded") return <Badge tone="good">succeeded</Badge>;
  if (status === "failed") return <Badge tone="bad">failed</Badge>;
  return (
    <Badge tone="accent">
      <Spinner className="mr-0.5" /> {status}
    </Badge>
  );
}

function AttemptCard({ attempt }: { attempt: Attempt }) {
  return (
    <Card className={attempt.gate_passed ? "border-[#22543d]" : undefined}>
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium">Attempt {attempt.attempt_number}</span>
            <Badge tone="neutral">{attempt.model_family}</Badge>
            {attempt.gate_passed ? (
              <Badge tone="good">trust gate passed</Badge>
            ) : (
              <Badge tone="bad">trust gate failed</Badge>
            )}
            {attempt.plan_source === "deterministic_fallback" ? (
              <Badge tone="neutral">deterministic plan</Badge>
            ) : (
              <Badge tone="accent">Claude-proposed plan</Badge>
            )}
          </div>
          <p className="text-sm text-muted mt-2 max-w-3xl">
            {attempt.revision_rationale}
          </p>
          {attempt.fallback_reason && (
            <p className="text-xs text-warn mt-1">
              Fell back to the deterministic planner: {attempt.fallback_reason}
            </p>
          )}
        </div>
      </div>

      {attempt.test_metrics && (
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2">
          <MetricPill label="Test RMSE" value={`${attempt.test_metrics.rmse.toFixed(2)} cyc`} />
          <MetricPill label="Test MAE" value={`${attempt.test_metrics.mae.toFixed(2)} cyc`} />
          <MetricPill
            label="Conformal coverage"
            value={`${((attempt.conformal_coverage ?? 0) * 100).toFixed(0)}%`}
          />
          <MetricPill
            label="Warning F1 (RUL≤30)"
            value={attempt.test_metrics.f1_at_threshold.toFixed(2)}
          />
        </div>
      )}

      {attempt.gate && (
        <div className="mt-4 space-y-1.5">
          {attempt.gate.checks.map((check) => (
            <div key={check.name} className="flex items-start gap-2 text-xs">
              <span className={check.passed ? "text-[#4ade80]" : "text-danger"}>
                {check.passed ? "✓" : "✗"}
              </span>
              <span className="text-muted mono">{check.detail}</span>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

function ModelCard({
  summary,
  detail,
  onPromote,
  promoting,
}: {
  summary: { id: string; attempt_number: number; stage: string; trust_gate_passed: boolean };
  detail: ModelVersion | undefined;
  onPromote: () => void;
  promoting: boolean;
}) {
  const chartData =
    detail?.shap_summary?.global_importance
      .slice(0, 6)
      .map((f) => ({ feature: f.feature, importance: f.mean_abs_shap })) ?? [];

  return (
    <Card>
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="font-medium">Attempt {summary.attempt_number}</span>
          {summary.stage === "production" ? (
            <Badge tone="accent">production</Badge>
          ) : (
            <Badge tone="neutral">shadow</Badge>
          )}
        </div>
        {summary.stage !== "production" && (
          <Button
            variant={summary.trust_gate_passed ? "primary" : "secondary"}
            disabled={!summary.trust_gate_passed || promoting}
            onClick={onPromote}
            className="text-xs px-3 py-1.5"
          >
            {promoting ? (
              <span className="flex items-center gap-1.5">
                <Spinner /> Promoting…
              </span>
            ) : summary.trust_gate_passed ? (
              "Promote to production"
            ) : (
              "Gate failed — cannot promote"
            )}
          </Button>
        )}
      </div>

      <div className="mt-4">
        <div className="text-xs uppercase tracking-wide text-muted mb-2">
          Global feature importance (mean |SHAP|)
        </div>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
              <XAxis type="number" tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--border)" />
              <YAxis
                type="category"
                dataKey="feature"
                width={72}
                tick={{ fill: "var(--muted)", fontSize: 11 }}
                stroke="var(--border)"
              />
              <Tooltip
                contentStyle={{
                  background: "var(--surface-raised)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  fontSize: 12,
                }}
              />
              <Bar dataKey="importance" fill="var(--accent)" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-xs text-muted flex items-center gap-2">
            <Spinner /> loading SHAP summary…
          </div>
        )}
      </div>
    </Card>
  );
}

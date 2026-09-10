"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api, Dataset } from "@/lib/api";
import { Badge, Button, Card, Spinner } from "@/components/ui";

const DEFAULT_GOAL =
  "Predict remaining useful life (RUL) for FD001 turbofan engines.";

export default function HomePage() {
  const router = useRouter();
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [goal, setGoal] = useState(DEFAULT_GOAL);
  const [loadingDataset, setLoadingDataset] = useState(false);
  const [startingRun, setStartingRun] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadDataset() {
    setLoadingDataset(true);
    setError(null);
    try {
      const ds = await api.createBundledDataset();
      setDataset(ds);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setLoadingDataset(false);
    }
  }

  async function startRun() {
    if (!dataset) return;
    setStartingRun(true);
    setError(null);
    try {
      const run = await api.startRun(dataset.id, goal);
      router.push(`/runs/${run.id}`);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
      setStartingRun(false);
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6 py-10 space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">
          Goal &amp; dataset
        </h1>
        <p className="text-sm text-muted max-w-2xl">
          Argus plans, trains, critiques, and — when a model falls short —
          revises its own pipeline. Every attempt is scored against a fixed,
          deterministic trust gate; nothing is ever promoted on an LLM&apos;s say-so.
        </p>
      </div>

      <Card className="space-y-4">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h2 className="font-medium">1. Dataset</h2>
            <p className="text-sm text-muted mt-1">
              Real NASA C-MAPSS FD001 turbofan degradation data — 100 training
              engines, 100 held-out test engines with official RUL labels.
            </p>
          </div>
          {!dataset && (
            <Button onClick={loadDataset} disabled={loadingDataset}>
              {loadingDataset ? (
                <span className="flex items-center gap-2">
                  <Spinner /> Loading FD001…
                </span>
              ) : (
                "Load bundled FD001 dataset"
              )}
            </Button>
          )}
        </div>

        {dataset && (
          <div className="rounded-md border border-border bg-surface-raised p-4 space-y-3">
            <div className="flex items-center gap-2 flex-wrap">
              <Badge tone="accent">{dataset.name}</Badge>
              <span className="text-xs text-muted mono">
                {dataset.n_rows.toLocaleString()} rows ·{" "}
                {dataset.profile.n_engines} engines
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
              <div>
                <div className="text-[11px] uppercase tracking-wide text-muted">
                  Engine life
                </div>
                <div className="mono">
                  {dataset.profile.engine_life_stats.min_cycles}–
                  {dataset.profile.engine_life_stats.max_cycles} cycles
                </div>
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-wide text-muted">
                  Mean life
                </div>
                <div className="mono">
                  {dataset.profile.engine_life_stats.mean_cycles.toFixed(1)}{" "}
                  cycles
                </div>
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-wide text-muted">
                  Constant sensors
                </div>
                <div className="mono">
                  {dataset.profile.constant_sensors.length} dropped
                </div>
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-wide text-muted">
                  Near-constant
                </div>
                <div className="mono">
                  {dataset.profile.near_constant_sensors.length} flagged
                </div>
              </div>
            </div>
            {dataset.profile.leakage_risk_notes.length > 0 && (
              <div className="text-xs text-muted border-t border-border pt-3">
                <span className="text-warn font-medium">
                  Leakage-risk note:{" "}
                </span>
                {dataset.profile.leakage_risk_notes[0]}
              </div>
            )}
          </div>
        )}
      </Card>

      <Card className="space-y-4">
        <div>
          <h2 className="font-medium">2. Goal</h2>
          <p className="text-sm text-muted mt-1">
            What should the pipeline optimize for? The planner uses this to
            frame its first attempt; a revision after a failed attempt is
            driven by the trust gate&apos;s evidence, not by re-reading this text.
          </p>
        </div>
        <textarea
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          rows={2}
          className="w-full rounded-md border border-border bg-surface-raised px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-accent-dim"
        />
      </Card>

      {error && (
        <div className="rounded-md border border-[#5a2c2c] bg-danger-dim px-4 py-3 text-sm text-danger">
          {error}
        </div>
      )}

      <div className="flex justify-end">
        <Button
          onClick={startRun}
          disabled={!dataset || startingRun}
          className="px-6 py-2.5"
        >
          {startingRun ? (
            <span className="flex items-center gap-2">
              <Spinner /> Starting run…
            </span>
          ) : (
            "Start pipeline run →"
          )}
        </Button>
      </div>
    </div>
  );
}

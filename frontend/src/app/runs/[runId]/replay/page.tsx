"use client";

import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts";
import { API_BASE, ReplayCycle } from "@/lib/api";
import { streamSse } from "@/lib/sse";
import { Badge, Button, Card, Spinner } from "@/components/ui";

export default function ReplayPage() {
  const searchParams = useSearchParams();
  const modelVersionId = searchParams.get("model");

  const [engines, setEngines] = useState<number[]>([]);
  const [selectedEngine, setSelectedEngine] = useState<number | null>(null);
  const [speed, setSpeed] = useState(8);
  const [cycles, setCycles] = useState<ReplayCycle[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/v1/replay/engines`)
      .then((r) => r.json())
      .then((d) => {
        setEngines(d.engines);
        setSelectedEngine((prev) => prev ?? d.engines[0]);
      })
      .catch((e) => setError(String(e)));
  }, []);

  function startReplay() {
    if (!modelVersionId || selectedEngine === null) return;
    stopRef.current?.();
    setCycles([]);
    setDone(false);
    setStreaming(true);
    setError(null);

    stopRef.current = streamSse(
      `${API_BASE}/api/v1/replay/${modelVersionId}/${selectedEngine}/events?speed=${speed}`,
      {
        onEvent: (event, data) => {
          if (event === "cycle") {
            setCycles((prev) => [...prev, JSON.parse(data) as ReplayCycle]);
          } else if (event === "done") {
            setDone(true);
            setStreaming(false);
          }
        },
        onError: (e) => {
          setError(String(e));
          setStreaming(false);
        },
      }
    );
  }

  useEffect(() => () => stopRef.current?.(), []);

  if (!modelVersionId) {
    return (
      <div className="mx-auto max-w-6xl px-4 sm:px-6 py-10 text-sm text-muted">
        No production model selected. Go back to the run, promote a model
        that passed the trust gate, then return here.
      </div>
    );
  }

  const latest = cycles[cycles.length - 1];
  const chartData = cycles.map((c) => ({
    cycle: c.time_cycles,
    predicted: Math.round(c.predicted_rul * 10) / 10,
    lower: Math.round(c.interval[0] * 10) / 10,
    upper: Math.round(c.interval[1] * 10) / 10,
    truth: c.true_rul_estimate,
  }));

  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Deployment — held-out engine replay
        </h1>
        <p className="text-sm text-muted mt-1 max-w-2xl">
          Streams a real held-out FD001 test engine, cycle by cycle, through
          the production model — not a synthetic signal. Watch predicted RUL
          track (or diverge from) the true remaining life as the engine
          approaches failure, with a live SHAP explanation of every point.
        </p>
      </div>

      <Card className="flex flex-wrap items-end gap-4">
        <div>
          <label className="text-xs uppercase tracking-wide text-muted block mb-1">
            Test engine
          </label>
          <select
            value={selectedEngine ?? ""}
            onChange={(e) => setSelectedEngine(Number(e.target.value))}
            disabled={streaming}
            className="rounded-md border border-border bg-surface-raised px-3 py-2 text-sm mono"
          >
            {engines.map((id) => (
              <option key={id} value={id}>
                Engine {id}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs uppercase tracking-wide text-muted block mb-1">
            Speed (cycles/sec)
          </label>
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            disabled={streaming}
            className="rounded-md border border-border bg-surface-raised px-3 py-2 text-sm mono"
          >
            {[2, 4, 8, 16, 40].map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </div>
        <Button onClick={startReplay} disabled={streaming || selectedEngine === null}>
          {streaming ? (
            <span className="flex items-center gap-2">
              <Spinner /> Streaming…
            </span>
          ) : (
            "▶ Replay engine"
          )}
        </Button>
        {done && <Badge tone="good">replay complete</Badge>}
      </Card>

      {error && (
        <div className="rounded-md border border-[#5a2c2c] bg-danger-dim px-4 py-3 text-sm text-danger">
          {error}
        </div>
      )}

      {cycles.length > 0 && (
        <>
          <Card>
            <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
              <div className="text-sm text-muted">
                Cycle <span className="mono text-foreground">{latest.time_cycles}</span> · predicted RUL{" "}
                <span className="mono text-foreground">{latest.predicted_rul.toFixed(1)}</span>
                {latest.true_rul_estimate !== null && (
                  <>
                    {" "}
                    · true RUL{" "}
                    <span className="mono text-foreground">{latest.true_rul_estimate.toFixed(0)}</span>
                  </>
                )}
              </div>
              {latest.warning ? (
                <Badge tone="bad">⚠ warning: RUL ≤ 30</Badge>
              ) : (
                <Badge tone="good">nominal</Badge>
              )}
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData} margin={{ left: 0, right: 16, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  dataKey="cycle"
                  tick={{ fill: "var(--muted)", fontSize: 11 }}
                  stroke="var(--border)"
                  label={{ value: "cycle", position: "insideBottom", offset: -4, fill: "var(--muted)", fontSize: 11 }}
                />
                <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--border)" />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface-raised)",
                    border: "1px solid var(--border)",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine y={30} stroke="var(--warn)" strokeDasharray="4 4" label={{ value: "warning threshold", fill: "var(--warn)", fontSize: 10, position: "insideTopRight" }} />
                <Line type="monotone" dataKey="upper" stroke="var(--border)" dot={false} strokeWidth={1} name="interval upper" />
                <Line type="monotone" dataKey="lower" stroke="var(--border)" dot={false} strokeWidth={1} name="interval lower" />
                <Line type="monotone" dataKey="truth" stroke="var(--muted)" dot={false} strokeWidth={2} name="true RUL" />
                <Line type="monotone" dataKey="predicted" stroke="var(--accent)" dot={false} strokeWidth={2.5} name="predicted RUL" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <div className="text-xs uppercase tracking-wide text-muted mb-2">
              Live SHAP explanation
            </div>
            <p className="text-sm mono leading-relaxed">{latest.narration ?? "—"}</p>
          </Card>
        </>
      )}
    </div>
  );
}

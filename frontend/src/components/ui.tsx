import type { ReactNode } from "react";
import clsx from "clsx";

export function Card({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={clsx(
        "rounded-lg border border-border bg-surface p-5",
        className
      )}
    >
      {children}
    </div>
  );
}

export function Badge({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "good" | "bad" | "warn" | "accent";
  children: ReactNode;
}) {
  const toneClasses: Record<string, string> = {
    neutral: "bg-surface-raised text-muted border-border",
    good: "bg-[#173226] text-[#4ade80] border-[#22543d]",
    bad: "bg-danger-dim text-danger border-[#5a2c2c]",
    warn: "bg-[#3a2f13] text-warn border-[#4a3a17]",
    accent: "bg-accent-dim/30 text-accent border-accent-dim",
  };
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        toneClasses[tone]
      )}
    >
      {children}
    </span>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
  type = "button",
  className,
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary" | "danger";
  type?: "button" | "submit";
  title?: string;
  className?: string;
}) {
  const variants: Record<string, string> = {
    primary:
      "bg-accent text-[#04201d] hover:bg-[#6ee0d5] disabled:opacity-40 disabled:hover:bg-accent",
    secondary:
      "bg-surface-raised text-foreground border border-border hover:border-accent-dim disabled:opacity-40",
    danger: "bg-danger text-[#2a0808] hover:bg-[#f58a8a] disabled:opacity-40",
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={clsx(
        "rounded-md px-3.5 py-2 text-sm font-medium transition-colors cursor-pointer disabled:cursor-not-allowed",
        variants[variant],
        className
      )}
    >
      {children}
    </button>
  );
}

export function MetricPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-surface-raised px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-muted">{label}</div>
      <div className="mono text-sm font-medium">{value}</div>
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={clsx(
        "inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent",
        className
      )}
    />
  );
}

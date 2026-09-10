import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Argus — Agentic Predictive Maintenance",
  description:
    "An agentic AutoML copilot that plans, trains, critiques, and re-plans real predictive-maintenance pipelines against a deterministic trust gate.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        <header className="border-b border-border">
          <div className="mx-auto max-w-6xl px-4 sm:px-6 py-4 flex items-center justify-between gap-4">
            <Link href="/" className="flex items-center gap-2.5">
              <span className="h-2.5 w-2.5 rounded-full bg-accent shadow-[0_0_8px_var(--accent)]" />
              <span className="text-lg font-semibold tracking-tight">Argus</span>
              <span className="hidden sm:inline text-sm text-muted">
                agentic predictive maintenance
              </span>
            </Link>
            <nav className="flex items-center gap-4 text-sm text-muted">
              <Link href="/" className="hover:text-foreground transition-colors">
                Runs
              </Link>
              <span
                className="hidden sm:inline"
                title="ABB Accelerator 2026 — Idea Phase prototype"
              >
                ABB Accelerator 2026
              </span>
            </nav>
          </div>
        </header>
        <main className="flex-1">{children}</main>
        <footer className="border-t border-border">
          <div className="mx-auto max-w-6xl px-4 sm:px-6 py-4 text-xs text-muted">
            Trained and evaluated on real NASA C-MAPSS FD001 turbofan data.
            Promotion is gated by five deterministic checks — never by model
            judgment.
          </div>
        </footer>
      </body>
    </html>
  );
}

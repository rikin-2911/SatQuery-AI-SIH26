import { Radar } from "lucide-react";
import { ApiStatus } from "@/components/ApiStatus";
import type { ApiStatusState, TaskType } from "@/types/analysis";

const CAPABILITIES: { key: string; label: string; color: string }[] = [
  { key: "sar", label: "SAR", color: "var(--sar)" },
  { key: "change", label: "CHANGE", color: "var(--change)" },
  { key: "grounding", label: "GROUNDING", color: "var(--grounding)" },
];

export function Header({
  apiState,
  activeTask,
}: {
  apiState: ApiStatusState;
  activeTask?: TaskType | null;
}) {
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-4 px-4 py-3 md:px-6">
        <div className="flex items-center gap-3">
          <div className="relative grid size-10 place-items-center rounded-lg border border-border bg-[var(--surface-raised)]">
            <Radar className="size-5 text-primary" strokeWidth={1.5} />
            <span className="pointer-events-none absolute inset-1 rounded-md border border-primary/20" />
          </div>
          <div className="leading-tight">
            <div className="text-[15px] font-semibold tracking-tight">SatQuery AI</div>
            <div className="tech-label">Remote Sensing Intelligence</div>
          </div>
        </div>

        <nav className="order-3 flex w-full items-center gap-2 md:order-none md:ml-6 md:w-auto">
          {CAPABILITIES.map((c) => {
            const active = activeTask === c.key;
            return (
              <span
                key={c.key}
                className="rounded-md border px-2.5 py-1 font-mono text-[11px] tracking-[0.12em]"
                style={{
                  borderColor: active ? c.color : "var(--border)",
                  color: active ? c.color : "var(--muted-foreground)",
                  backgroundColor: active
                    ? "color-mix(in oklch, var(--surface-raised) 80%, transparent)"
                    : "transparent",
                }}
              >
                {c.label}
              </span>
            );
          })}
        </nav>

        <div className="ml-auto">
          <ApiStatus state={apiState} />
        </div>
      </div>
    </header>
  );
}

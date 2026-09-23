import { useState } from "react";
import { ChevronDown } from "lucide-react";

function renderItem(item: unknown): string {
  if (item == null) return "—";
  if (typeof item === "string") return item;
  if (typeof item === "object") {
    const obj = item as Record<string, unknown>;
    const parts = Object.entries(obj).map(([k, v]) => {
      const val = typeof v === "object" && v !== null ? JSON.stringify(v) : String(v);
      return `${k}: ${val}`;
    });
    return parts.join("  ·  ");
  }
  return String(item);
}

export function ExecutionTrace({ trace }: { trace: unknown[] }) {
  const [open, setOpen] = useState(true);
  if (!trace || trace.length === 0) return null;

  return (
    <section className="rounded-lg border border-border bg-[var(--surface-raised)]/50">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-3"
      >
        <span className="tech-label">Execution Trace</span>
        <span className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
          {trace.length} steps
          <ChevronDown
            className={`size-4 transition-transform ${open ? "rotate-180" : ""}`}
          />
        </span>
      </button>
      {open && (
        <ol className="space-y-px border-t border-border">
          {trace.map((item, i) => (
            <li key={i} className="flex gap-3 px-4 py-2.5">
              <span className="font-mono text-[11px] text-primary">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="min-w-0 flex-1 font-mono text-[12px] leading-relaxed break-words text-foreground/85">
                {renderItem(item)}
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

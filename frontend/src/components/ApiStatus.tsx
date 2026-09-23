import type { ApiStatusState } from "@/types/analysis";

export function ApiStatus({ state }: { state: ApiStatusState }) {
  const label =
    state === "online" ? "API Connected" : state === "checking" ? "Checking API…" : "API Offline";

  const dotColor =
    state === "online"
      ? "bg-[var(--online)]"
      : state === "checking"
        ? "bg-muted-foreground"
        : "bg-destructive";

  return (
    <div className="flex items-center gap-2 rounded-full border border-border bg-[var(--surface-raised)]/70 px-3 py-1.5">
      <span
        className={`size-2 rounded-full ${dotColor} ${state !== "offline" ? "pulse-dot" : ""}`}
        aria-hidden
      />
      <span className="font-mono text-[11px] tracking-[0.1em] uppercase text-foreground/85">
        {label}
      </span>
    </div>
  );
}

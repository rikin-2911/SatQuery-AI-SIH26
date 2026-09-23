import { Layers, X } from "lucide-react";

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function ImagePreview({
  file,
  index,
  total,
  onRemove,
  disabled,
}: {
  file: File;
  index: number;
  total: number;
  onRemove: () => void;
  disabled?: boolean | undefined;
}) {
  const slot = `IMAGE 0${index + 1}`;
  const timeLabel = total === 2 ? (index === 0 ? "TIME 1" : "TIME 2") : null;

  return (
    <div className="flex items-center gap-3 rounded-lg border border-border bg-[var(--surface-raised)]/70 p-3">
      <div className="relative grid size-12 shrink-0 place-items-center overflow-hidden rounded-md border border-border bg-background grid-motif">
        <Layers className="size-5 text-primary/80" strokeWidth={1.5} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] tracking-[0.12em] text-primary">{slot}</span>
          {timeLabel && (
            <span className="rounded border border-border px-1.5 py-0.5 font-mono text-[10px] tracking-[0.12em] text-muted-foreground">
              {timeLabel}
            </span>
          )}
          <span className="rounded border border-border px-1.5 py-0.5 font-mono text-[10px] tracking-[0.12em] text-muted-foreground">
            GeoTIFF
          </span>
        </div>
        <div className="mt-1 truncate text-sm text-foreground">{file.name}</div>
        <div className="font-mono text-[11px] text-muted-foreground">
          {formatSize(file.size)} · preview not rendered in browser
        </div>
      </div>
      <button
        type="button"
        onClick={onRemove}
        disabled={disabled}
        aria-label={`Remove ${file.name}`}
        className="grid size-8 shrink-0 place-items-center rounded-md border border-border text-muted-foreground transition-colors hover:border-destructive/50 hover:text-destructive disabled:opacity-50"
      >
        <X className="size-4" />
      </button>
    </div>
  );
}

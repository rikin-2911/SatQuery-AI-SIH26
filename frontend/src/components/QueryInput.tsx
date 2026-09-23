import { Textarea } from "@/components/ui/textarea";

const EXAMPLES = [
  "What is the dominant land-cover type in this SAR image?",
  "Where is the main water body in this image?",
  "What has changed between these two images?",
];

export function QueryInput({
  value,
  onChange,
  disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="space-y-3">
      <label htmlFor="query" className="tech-label block">
        Analysis Query
      </label>
      <Textarea
        id="query"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Ask something about your satellite imagery..."
        className="min-h-32 resize-y rounded-lg border-border bg-[var(--surface-raised)]/60 text-sm leading-relaxed placeholder:text-muted-foreground/70 focus-visible:ring-ring"
      />
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            disabled={disabled}
            onClick={() => onChange(ex)}
            className="rounded-md border border-border bg-transparent px-2.5 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground disabled:opacity-50"
          >
            {ex}
          </button>
        ))}
      </div>
      <p className="font-mono text-[11px] text-muted-foreground">
        Supported input: GeoTIFF (.tif, .tiff)
      </p>
    </div>
  );
}

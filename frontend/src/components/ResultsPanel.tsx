import { Crosshair, Radar, ScanLine, Layers } from "lucide-react";
import { ExecutionTrace } from "@/components/ExecutionTrace";
import type { AnalysisResponse, BoundingBoxItem } from "@/types/analysis";
import { API_BASE_URL } from "@/services/api";

function taskLabel(task?: string | null) {
  switch ((task ?? "").toLowerCase()) {
    case "sar":
      return "SAR Analysis";
    case "change":
    case "change_detection":
      return "Change Detection";
    case "grounding":
      return "Grounding";
    default:
      return task ? String(task) : "Unknown";
  }
}

function taskColor(task?: string | null) {
  const t = (task ?? "").toLowerCase();
  if (t.startsWith("sar")) return "var(--sar)";
  if (t.startsWith("change")) return "var(--change)";
  if (t.startsWith("ground")) return "var(--grounding)";
  return "var(--muted-foreground)";
}

function TaskIcon({ task }: { task?: string | null | undefined }) {
  const t = (task ?? "").toLowerCase();
  if (t.startsWith("change")) return <Layers className="size-4" strokeWidth={1.6} />;
  if (t.startsWith("ground")) return <Crosshair className="size-4" strokeWidth={1.6} />;
  return <ScanLine className="size-4" strokeWidth={1.6} />;
}

function asText(value: unknown): string | null {
  if (value == null) return null;
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value, null, 2);
}

function ListBlock({ title, value }: { title: string; value: unknown }) {
  if (value == null) return null;
  const items = Array.isArray(value) ? value : [value];
  if (items.length === 0) return null;
  return (
    <section className="rounded-lg border border-border bg-[var(--surface-raised)]/50 p-4">
      <h4 className="tech-label mb-2">{title}</h4>
      <ul className="space-y-1.5">
        {items.map((item, i) => (
          <li key={i} className="flex gap-2 text-sm text-foreground/85">
            <span className="mt-2 size-1 shrink-0 rounded-full bg-primary" />
            <span className="min-w-0 break-words whitespace-pre-wrap">{asText(item)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function normalizeBoxes(raw: unknown): BoundingBoxItem[] {
  if (!Array.isArray(raw)) return [];

  return raw
    .map((item): BoundingBoxItem | null => {
      if (!item || typeof item !== "object") {
        return null;
      }

      const obj = item as Record<string, unknown>;

      const box =
        obj.bounding_box ??
        obj.bbox ??
        obj.box;

      if (!Array.isArray(box) || box.length < 4) {
        return null;
      }

      return {
        box: box as number[],
        label:
          typeof obj.feature === "string"
            ? obj.feature
            : typeof obj.label === "string"
              ? obj.label
              : undefined,
        confidence:
          typeof obj.confidence === "number"
            ? obj.confidence
            : undefined,
      };
    })
    .filter(
      (item): item is BoundingBoxItem =>
        item !== null
    );
}

function BoundingBoxes({
  boxes,
  imageUrl,
}: {
  boxes: BoundingBoxItem[];
  imageUrl?: string | null;
}) {
  if (boxes.length === 0) return null;

  const normalized = boxes.every((b) =>
    (b.box ?? []).every((v) => v >= 0 && v <= 1)
  );

  return (
    <section className="rounded-lg border border-border bg-[var(--surface-raised)]/50 p-4">
      <h4 className="tech-label mb-3">Grounding Evidence</h4>

      <div className="relative w-full overflow-hidden rounded-md border border-border bg-background">
        {imageUrl ? (
          <img
            src={imageUrl}
            alt="Remote sensing evidence"
            className="block h-auto w-full"
          />
        ) : (
          <div className="aspect-square grid place-items-center grid-motif">
            <span className="font-mono text-[10px] tracking-[0.12em] text-muted-foreground/60 uppercase">
              Evidence image unavailable
            </span>
          </div>
        )}

        {imageUrl &&
          normalized &&
          boxes.map((b, i) => {
            const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] =
              b.box as number[];

            return (
              <div
                key={i}
                className="absolute border-2 border-[var(--grounding)]"
                style={{
                  left: `${x1 * 100}%`,
                  top: `${y1 * 100}%`,
                  width: `${Math.max(0, x2 - x1) * 100}%`,
                  height: `${Math.max(0, y2 - y1) * 100}%`,
                }}
              >
                <span className="absolute -top-6 left-0 max-w-48 truncate rounded bg-[var(--grounding)] px-1.5 py-0.5 font-mono text-[10px] text-background">
                  {b.label ?? `box ${i + 1}`}
                </span>
              </div>
            );
          })}
      </div>

      <ul className="mt-3 space-y-1.5">
        {boxes.map((b, i) => (
          <li
            key={i}
            className="font-mono text-[11px] text-muted-foreground"
          >
            <span className="text-foreground/80">
              {b.label ?? `box_${i + 1}`}
            </span>
            : [{(b.box ?? []).join(", ")}]
            {b.confidence != null
              ? ` · conf ${Math.round(b.confidence * 100)}%`
              : ""}
          </li>
        ))}
      </ul>

      {!normalized && (
        <p className="mt-2 font-mono text-[11px] text-muted-foreground">
          Coordinates are not normalized — overlay preview unavailable.
        </p>
      )}
    </section>
  );
}

function pick(result: unknown, data: AnalysisResponse, key: string): unknown {
  if (result && typeof result === "object" && key in (result as Record<string, unknown>)) {
    return (result as Record<string, unknown>)[key];
  }
  return data[key];
}

function ChangeEvidence({
  imageUrls,
}: {
  imageUrls: string[];
}) {
  if (imageUrls.length === 0) {
    return (
      <section className="rounded-lg border border-border bg-[var(--surface-raised)]/50 p-4">
        <h4 className="tech-label mb-3">Change Evidence</h4>

        <div className="grid aspect-video place-items-center rounded-md border border-border grid-motif">
          <span className="font-mono text-[10px] tracking-[0.12em] text-muted-foreground/60 uppercase">
            Evidence images unavailable
          </span>
        </div>
      </section>
    );
  }

  return (
    <section className="rounded-lg border border-border bg-[var(--surface-raised)]/50 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h4 className="tech-label">Change Evidence</h4>

        <span className="font-mono text-[10px] text-muted-foreground">
          BI-TEMPORAL
        </span>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        {imageUrls.slice(0, 2).map((url, index) => (
          <div
            key={url}
            className="overflow-hidden rounded-md border border-border bg-background"
          >
            <div className="border-b border-border px-3 py-2">
              <span className="font-mono text-[10px] tracking-[0.12em] text-muted-foreground uppercase">
                {index === 0 ? "T1 · Earlier" : "T2 · Later"}
              </span>
            </div>

            <img
              src={url}
              alt={index === 0 ? "Earlier satellite image" : "Later satellite image"}
              className="block h-auto w-full"
            />
          </div>
        ))}
      </div>

      {imageUrls.length === 1 && (
        <p className="mt-2 font-mono text-[10px] text-muted-foreground">
          Only one evidence image was returned.
        </p>
      )}
    </section>
  );
}

export function ResultsPanel({
  data,
  error,
  loading,
  stage,
}: {
  data: AnalysisResponse | null;
  error: string | null;
  loading: boolean;
  stage: string | null;
}) {
const task = data?.task_type;
const t = (task ?? "").toLowerCase();
const result = data?.raw_result ?? data?.result;
const answer = data?.answer ?? "No answer returned by the backend.";
const evidenceUrl = data?.evidence?.images?.[0]?.url
  ? `${API_BASE_URL}${data.evidence.images[0].url}`
  : null;

const evidenceUrls =
  data?.evidence?.images?.map((image) =>
    image.url.startsWith("http")
      ? image.url
      : `${API_BASE_URL}${image.url}`
  ) ?? [];

  return (
    <div className="panel flex min-h-[560px] flex-col p-5">
      <div className="flex items-center justify-between border-b border-border pb-4">
        <div>
          <h2 className="text-base font-semibold tracking-tight">Analysis Results</h2>
          <p className="tech-label mt-1">Agentic output · auditable trace</p>
        </div>
        {data?.request_id && (
          <span className="max-w-40 truncate font-mono text-[10px] text-muted-foreground">
            {data.request_id}
          </span>
        )}
      </div>

      {error && (
        <div className="mt-4 rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {loading && (
        <div className="mt-6 flex flex-1 flex-col items-center justify-center gap-4">
          <div className="relative grid size-28 place-items-center rounded-full border border-border grid-motif">
            <span className="radar-sweep absolute inset-0 rounded-full border-t-2 border-primary/70" />
            <Radar className="size-7 text-primary" strokeWidth={1.2} />
          </div>
          <p className="font-mono text-[11px] tracking-[0.1em] text-primary uppercase">{stage}</p>
        </div>
      )}

      {!loading && !data && !error && (
        <div className="flex flex-1 flex-col items-center justify-center gap-4 py-16 text-center">
          <div className="relative grid size-32 place-items-center overflow-hidden rounded-full border border-border grid-motif">
            <span className="absolute size-20 rounded-full border border-primary/25" />
            <span className="absolute size-11 rounded-full border border-primary/40" />
            <Crosshair className="size-6 text-primary/70" strokeWidth={1.2} />
          </div>
          <p className="max-w-xs text-sm text-muted-foreground">
            Upload imagery and submit a query to begin analysis.
          </p>
        </div>
      )}

      {!loading && data && (
        <div className="mt-4 space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className="inline-flex items-center gap-2 rounded-md border px-2.5 py-1.5 font-mono text-[11px] tracking-[0.12em] uppercase"
              style={{ borderColor: taskColor(task), color: taskColor(task) }}
            >
              <TaskIcon task={task} />
              Task: {taskLabel(task)}
            </span>
            <span className="rounded-md border border-border px-2.5 py-1.5 font-mono text-[11px] text-muted-foreground">
              Confidence:{" "}
              {data.confidence == null
                ? "Not provided"
                : typeof data.confidence === "number"
                  ? `${Math.round(data.confidence * 100)}%`
                  : String(data.confidence)}
            </span>
          </div>

          <section className="rounded-lg border border-border bg-[var(--surface-raised)]/50 p-4">
            <h3 className="tech-label mb-2">AI Analysis</h3>
            <p className="text-sm leading-relaxed whitespace-pre-wrap text-foreground/90">
              {data.answer ?? "No answer returned by the backend."}
            </p>
          </section>

          {t.startsWith("sar") && (
            <>
              <ListBlock title="Observations" value={pick(result, data, "observations")} />
              <ListBlock title="Uncertainties" value={pick(result, data, "uncertainties")} />
            </>
          )}

          {t.startsWith("ground") && (
              <>
                <ListBlock
                  title="Detected Features"
                  value={pick(result, data, "detections")}
                />

                <BoundingBoxes
                  boxes={normalizeBoxes(
                    pick(result, data, "detections")
                  )}
                  imageUrl={evidenceUrl}
                />
              </>
            )
          }

          {t.startsWith("change") && (
            <>
              <ChangeEvidence imageUrls={evidenceUrls} />

              <ListBlock
                title="Change Description"
                value={pick(result, data, "change_description")}
              />

              <ListBlock
                title="Detected Changes"
                value={pick(result, data, "detected_changes") ??
                       pick(result, data, "changes")
                }
              />

              <ListBlock
                title="Uncertainty"
                value={
                  pick(result, data, "uncertainty") ??
                  pick(result, data, "uncertainties")
                }
              />
            </>
          )}

          <section className="rounded-lg border border-border bg-[var(--surface-raised)]/50 p-4">
            <h3 className="tech-label mb-1">Model Used</h3>
            <p className="font-mono text-[12px] break-words text-foreground/85">
              {data.model_used ?? "Not provided"}
            </p>
          </section>

          <ExecutionTrace trace={data.execution_trace ?? []} />
        </div>
      )}
    </div>
  );
}

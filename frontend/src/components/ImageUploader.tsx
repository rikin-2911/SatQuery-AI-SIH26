import { useRef, useState } from "react";
import { UploadCloud } from "lucide-react";
import { ImagePreview } from "@/components/ImagePreview";

const ACCEPTED = [".tif", ".tiff"];

function isTiff(file: File) {
  const name = file.name.toLowerCase();
  return ACCEPTED.some((ext) => name.endsWith(ext));
}

export function ImageUploader({
  files,
  onChange,
  onError,
  disabled,
}: {
  files: File[];
  onChange: (files: File[]) => void;
  onError: (message: string | null) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);

  const accept = (incoming: FileList | null) => {
    if (!incoming || incoming.length === 0) return;
    const list = Array.from(incoming);

    if (list.some((f) => !isTiff(f))) {
      onError("Unsupported file format. Please upload .tif or .tiff files.");
      return;
    }
    const next = [...files, ...list];
    if (next.length > 2) {
      onError("Maximum two images are allowed.");
      return;
    }
    onError(null);
    setLoading(true);
    window.setTimeout(() => {
      onChange(next);
      setLoading(false);
    }, 200);
  };

  return (
    <div className="space-y-3">
      <div
        role="button"
        tabIndex={0}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) accept(e.dataTransfer.files);
        }}
        className={`relative overflow-hidden rounded-xl border border-dashed p-6 text-center transition-colors ${
          dragging ? "border-primary bg-primary/5" : "border-border bg-[var(--surface-raised)]/40"
        } ${disabled ? "pointer-events-none opacity-60" : "cursor-pointer hover:border-primary/50"}`}
      >
        <div className="pointer-events-none absolute inset-0 grid-motif opacity-25" />
        <div className="relative flex flex-col items-center gap-2">
          <div className="grid size-11 place-items-center rounded-lg border border-border bg-background">
            <UploadCloud className="size-5 text-primary" strokeWidth={1.5} />
          </div>
          <div className="text-sm font-medium">Upload Remote Sensing Image</div>
          <div className="text-xs text-muted-foreground">
            Drop GeoTIFF files here or browse from your computer
          </div>
          <div className="mt-2 flex flex-wrap justify-center gap-2 font-mono text-[10px] tracking-[0.1em] text-muted-foreground uppercase">
            <span className="rounded border border-border px-2 py-1">1 image for SAR / Grounding</span>
            <span className="rounded border border-border px-2 py-1">2 images for Change Detection</span>
          </div>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".tif,.tiff,image/tiff"
          multiple
          className="hidden"
          onChange={(e) => {
            accept(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {loading && (
        <div className="font-mono text-[11px] text-primary">Reading file…</div>
      )}

      {files.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="tech-label">Loaded imagery</span>
            <span className="font-mono text-[11px] text-muted-foreground">{files.length} / 2</span>
          </div>
          {files.map((file, i) => (
            <ImagePreview
              key={`${file.name}-${i}`}
              file={file}
              index={i}
              total={files.length}
              disabled={disabled}
              onRemove={() => {
                onError(null);
                onChange(files.filter((_, idx) => idx !== i));
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

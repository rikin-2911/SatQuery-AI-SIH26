import { Loader2, SatelliteDish } from "lucide-react";
import { Button } from "@/components/ui/button";

export function AnalysisButton({
  disabled,
  loading,
  stage,
  onClick,
}: {
  disabled: boolean;
  loading: boolean;
  stage: string | null;
  onClick: () => void;
}) {
  return (
    <div className="space-y-2">
      <Button
        onClick={onClick}
        disabled={disabled}
        className="h-12 w-full rounded-lg bg-primary text-sm font-semibold tracking-wide text-primary-foreground hover:bg-primary/90"
      >
        {loading ? (
          <Loader2 className="size-4 animate-spin" />
        ) : (
          <SatelliteDish className="size-4" strokeWidth={1.75} />
        )}
        Analyze Imagery
      </Button>
      {loading && stage && (
        <p className="text-center font-mono text-[11px] tracking-[0.08em] text-primary">{stage}</p>
      )}
    </div>
  );
}

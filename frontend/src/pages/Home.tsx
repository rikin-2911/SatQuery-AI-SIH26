import { useEffect, useState } from "react";
import { Header } from "@/components/Header";
import { QueryInput } from "@/components/QueryInput";
import { ImageUploader } from "@/components/ImageUploader";
import { AnalysisButton } from "@/components/AnalysisButton";
import { ResultsPanel } from "@/components/ResultsPanel";
import { analyzeImages, checkHealth, API_BASE_URL } from "@/services/api";
import { ApiError, type AnalysisResponse, type ApiStatusState } from "@/types/analysis";

const STAGES = [
  "Processing satellite imagery...",
  "Routing query...",
  "Running specialist analysis...",
  "Preparing results...",
];

export default function Home() {
  const [apiState, setApiState] = useState<ApiStatusState>("checking");
  const [query, setQuery] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [data, setData] = useState<AnalysisResponse | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    checkHealth(controller.signal)
      .then((res) => setApiState(res.status === "ok" ? "online" : "offline"))
      .catch(() => setApiState("offline"));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!loading) return;
    let i = 0;
    setStage(STAGES[0]!);
    const id = window.setInterval(() => {
      i = Math.min(i + 1, STAGES.length - 1);
      setStage(STAGES[i]!);
    }, 1400);
    return () => window.clearInterval(id);
  }, [loading]);

  const canSubmit =
    query.trim().length > 0 && files.length > 0 && files.length <= 2 && !loading;

  const handleAnalyze = async () => {
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const res = await analyzeImages(query.trim(), files);
      setData(res);
      setApiState("online");
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
        if (err.kind === "network") setApiState("offline");
      } else {
        setError("Analysis failed. Please try again.");
      }
    } finally {
      setLoading(false);
      setStage(null);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="pointer-events-none fixed inset-0 grid-motif opacity-[0.18]" />
      <div className="relative">
        <Header apiState={apiState} activeTask={data?.task_type ?? null} />

        <main className="mx-auto max-w-[1500px] px-4 py-6 md:px-6 md:py-8">
          <div className="mb-6">
            <h1 className="text-xl font-semibold tracking-tight md:text-2xl">
              Interactive Vision-Language Assistant for Remote Sensing Image Analysis
            </h1>
            <p className="tech-label mt-1">Endpoint · {API_BASE_URL}/api/v1/analyze</p>
          </div>

          {apiState === "offline" && (
            <div className="mb-6 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              Unable to connect to SatQuery AI backend. You can still prepare a request, but analysis
              will fail until the API is reachable.
            </div>
          )}

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
            <div className="panel space-y-6 p-5">
              <div className="border-b border-border pb-4">
                <h2 className="text-base font-semibold tracking-tight">New Analysis</h2>
                <p className="mt-1 text-xs text-muted-foreground">
                  Upload satellite imagery and ask a natural-language question.
                </p>
              </div>

              <QueryInput value={query} onChange={setQuery} disabled={loading} />

              <ImageUploader
                files={files}
                onChange={setFiles}
                onError={setUploadError}
                disabled={loading}
              />

              {uploadError && (
                <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                  {uploadError}
                </p>
              )}

              <div className="rounded-lg border border-border bg-[var(--surface-raised)]/40 p-4">
                <h3 className="tech-label mb-2">Detected Task</h3>
                {data?.task_type ? (
                  <p className="font-mono text-sm text-primary">
                    Detected Task: {String(data.task_type).toUpperCase()}
                  </p>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    Task will be determined by the AI router
                  </p>
                )}
              </div>

              <AnalysisButton
                disabled={!canSubmit}
                loading={loading}
                stage={stage}
                onClick={handleAnalyze}
              />
            </div>

            <ResultsPanel data={data} error={error} loading={loading} stage={stage} />
          </div>
        </main>
      </div>
    </div>
  );
}

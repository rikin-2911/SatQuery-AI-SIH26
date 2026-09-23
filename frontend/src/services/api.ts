import { ApiError, type AnalysisResponse, type HealthResponse } from "@/types/analysis";

export const API_BASE_URL =
  (import.meta.env['VITE_API_BASE_URL'] as string | undefined)?.replace(/\/$/, "") ??
  "http://localhost:8000";

export async function checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/v1/health`, { signal: signal ?? null });
  } catch {
    throw new ApiError("Unable to connect to SatQuery AI backend.", "network");
  }
  if (!res.ok) throw new ApiError(`Health check failed (${res.status}).`, "http");
  try {
    return (await res.json()) as HealthResponse;
  } catch {
    throw new ApiError("The backend returned an unexpected response.", "invalid");
  }
}

export async function analyzeImages(query: string, files: File[]): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("query", query);
  for (const file of files) {
    form.append("images", file, file.name);
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
      method: "POST",
      body: form,
    });
  } catch {
    throw new ApiError(
      "Unable to connect to SatQuery AI backend. Check that the server is running and allows requests from this origin (CORS).",
      "network",
    );
  }

  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.text();
      detail = body.slice(0, 300);
    } catch {
      /* ignore */
    }
    throw new ApiError(
      `Analysis failed. Please try again. (HTTP ${res.status}${detail ? ` — ${detail}` : ""})`,
      "http",
    );
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new ApiError("The backend returned an unexpected response.", "invalid");
  }
  if (!data || typeof data !== "object") {
    throw new ApiError("The backend returned an unexpected response.", "invalid");
  }
  return data as AnalysisResponse;
}

export type TaskType = "sar" | "change" | "grounding" | string;

export interface BoundingBoxItem {
  label?: string;
  box?: number[];
  bbox?: number[];
  confidence?: number | null;
}

export interface AnalysisResponse {
  request_id: string;
  query: string;
  image_paths: string[];

  task_type: string | null;

  answer?: string;

  result?: string;

  raw_result?: unknown;

  model_used?: string | null;

  confidence?: number | null;

  execution_trace: string[];

  evidence?: {
    images: {
      filename?: string;
      url: string;
    }[];
  };
  // optional task-specific extras the backend may return
  observations?: unknown;
  uncertainties?: unknown;
  uncertainty?: unknown;
  detected_features?: unknown;
  bounding_boxes?: BoundingBoxItem[] | number[][];
  detected_changes?: unknown;
  change_description?: string;
  [key: string]: unknown;
}

export interface HealthResponse {
  status?: string;
  service?: string;
  version?: string;
}

export type ApiStatusState = "checking" | "online" | "offline";

export class ApiError extends Error {
  kind: "network" | "http" | "invalid";
  constructor(message: string, kind: "network" | "http" | "invalid") {
    super(message);
    this.kind = kind;
  }
}

export type ForecastModelId =
  | "naive"
  | "moving_average"
  | "exponential_smoothing"
  | "linear_trend"
  | "ets"
  | "elm"
  | "lstm";

export type DecompositionModelId = "none" | "vmd" | "ssa" | "ewt";

export type CombinationModelId = "additive" | "pso_cs";

export type ModelParameter = {
  name: string;
  label: string;
  type: "number" | "integer";
  default: number;
  min?: number;
  max?: number;
  step?: number;
  meaning: string;
  recommendation: string;
};

export type ModelInfo = {
  id: ForecastModelId;
  name: string;
  description: string;
  best_for: string;
  parameters: ModelParameter[];
};

export type DecompositionInfo = {
  id: DecompositionModelId;
  name: string;
  description: string;
  best_for: string;
  parameters: ModelParameter[];
};

export type SeriesPoint = {
  date: string;
  value: number;
};

export type ForecastSummary = {
  history_count: number;
  forecast_count: number;
  last_actual: number;
  first_forecast: number;
  final_forecast: number;
  forecast_average: number;
  forecast_min: number;
  forecast_max: number;
  absolute_change: number;
  percentage_change?: number | null;
  trend: "上升" | "下降" | "持平";
};

export type JobUsage = {
  input_bytes: number;
  history_points: number;
  forecast_steps: number;
  parameter_candidates: number;
  decomposition_candidates: number;
  decomposition_components: number;
  model_runs: number;
  generated_points: number;
};

export type DecompositionComponentSummary = {
  name: string;
  history_min: number;
  history_max: number;
  history_average: number;
  forecast_final: number;
  forecast_average: number;
  parameters: Record<string, number>;
};

export type DecompositionRun = {
  model: DecompositionModelId;
  parameters: Record<string, number>;
  auto_tune: boolean;
  tuning_score?: number | null;
  tuning_test_size?: number | null;
  component_count: number;
  component_names: string[];
  components: DecompositionComponentSummary[];
};

export type ForecastJob = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  progress: number;
  model?: ForecastModelId;
  parameters: Record<string, number>;
  auto_tune: boolean;
  tuning_score?: number | null;
  tuning_test_size?: number | null;
  submitted_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  elapsed_ms?: number | null;
  usage: JobUsage;
  history: SeriesPoint[];
  forecast: SeriesPoint[];
  decomposition_result?: DecompositionRun | null;
  summary?: ForecastSummary | null;
  error?: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function fetchHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) {
    throw new Error("预测服务不可用");
  }
  return response.json();
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const response = await fetch(`${API_BASE}/api/models`);
  if (!response.ok) {
    throw new Error("无法获取模型列表");
  }
  return response.json();
}

export async function fetchDecompositions(): Promise<DecompositionInfo[]> {
  const response = await fetch(`${API_BASE}/api/decompositions`);
  if (!response.ok) {
    throw new Error("无法获取分解模型列表");
  }
  return response.json();
}

export async function startForecast(
  file: File,
  model: ForecastModelId,
  horizon: number,
  parameters: Record<string, number>,
  autoTune: boolean,
  decomposition: DecompositionModelId,
  decompositionParameters: Record<string, number>,
  autoTuneDecomposition: boolean,
  combination: CombinationModelId = "additive",
): Promise<ForecastJob> {
  const form = new FormData();
  form.append("file", file);
  form.append(
    "request",
    JSON.stringify({
      model,
      horizon,
      parameters,
      auto_tune: autoTune,
      decomposition,
      decomposition_parameters: decompositionParameters,
      auto_tune_decomposition: autoTuneDecomposition,
      combination,
    }),
  );
  const response = await fetch(`${API_BASE}/api/forecast`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    const detail = await readError(response);
    throw new Error(detail);
  }
  return response.json();
}

export async function pollForecast(jobId: string): Promise<ForecastJob> {
  const response = await fetch(`${API_BASE}/api/forecast/${jobId}`);
  if (!response.ok) {
    throw new Error("无法获取任务状态");
  }
  return response.json();
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (Array.isArray(payload.detail)) {
      return payload.detail
        .map((item: { msg?: string; message?: string }) => item.msg ?? item.message)
        .filter(Boolean)
        .join("；");
    }
    return "请求失败";
  } catch {
    return "请求失败";
  }
}

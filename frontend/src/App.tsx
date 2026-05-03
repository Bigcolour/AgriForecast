import type { ChangeEvent, DragEvent, ReactNode } from "react";
import { memo, useCallback, useEffect, useMemo, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend as ChartLegend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  FileUp,
  Info,
  Layers,
  Loader2,
  Play,
  Server,
  Trash2,
  UploadCloud,
} from "lucide-react";

import {
  CombinationModelId,
  DecompositionInfo,
  DecompositionModelId,
  fetchDecompositions,
  fetchHealth,
  fetchModels,
  ForecastJob,
  ForecastModelId,
  ModelInfo,
  ModelParameter,
  pollForecast,
  startForecast,
} from "./api";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, ChartTooltip, ChartLegend);

type RunState = "idle" | "submitting" | "polling" | "completed" | "failed";
type ServerState = "checking" | "online" | "offline";
type MetricTone = "neutral" | "success" | "danger";
type ParameterOwner = { parameters: ModelParameter[] };
type ChartPoint = {
  date: string;
  actual: number | null;
  forecast: number | null;
};

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const PREVIEW_LIMIT = 6;
const SAMPLE_ROW_COUNT = 300;
const XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

type CsvPreview = {
  fileType: string;
  encoding?: string;
  columns: string[];
  rows: Record<string, string>[];
  totalRows: number;
  dateColumn?: string;
  valueColumn?: string;
  dateRange?: string;
  valueRange?: string;
  issue?: string;
};

function createSampleCsv(rowCount: number): string {
  const startDate = Date.UTC(2025, 0, 1);
  const oneDayMs = 24 * 60 * 60 * 1000;
  const rows = Array.from({ length: rowCount }, (_, index) => {
    const date = new Date(startDate + index * oneDayMs).toISOString().slice(0, 10);
    const trend = 118 + index * 0.34;
    const weeklyCycle = Math.sin((index / 7) * Math.PI * 2) * 8;
    const monthlyCycle = Math.cos((index / 30) * Math.PI * 2) * 5;
    const midSeasonLift = index >= 145 && index <= 190 ? 11 : 0;
    const shortDip = index >= 235 && index <= 250 ? -9 : 0;
    const jitter = (((index * 37) % 17) - 8) * 0.45;
    const value = trend + weeklyCycle + monthlyCycle + midSeasonLift + shortDip + jitter;

    return `${date},${value.toFixed(1)}`;
  });

  return ["date,value", ...rows].join("\n");
}

const SAMPLE_CSV = createSampleCsv(SAMPLE_ROW_COUNT);

const statusText: Record<ForecastJob["status"], string> = {
  queued: "排队中",
  running: "计算中",
  completed: "已完成",
  failed: "失败",
};

const serverText: Record<ServerState, string> = {
  checking: "检测中",
  online: "服务在线",
  offline: "服务离线",
};

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [decompositions, setDecompositions] = useState<DecompositionInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<ForecastModelId>("moving_average");
  const [selectedDecompositionId, setSelectedDecompositionId] = useState<DecompositionModelId>("none");
  const [selectedCombinationId, setSelectedCombinationId] = useState<CombinationModelId>("additive");
  const [file, setFile] = useState<File | null>(null);
  const [horizon, setHorizon] = useState(14);
  const [horizonInput, setHorizonInput] = useState("14");
  const [parameters, setParameters] = useState<Record<string, number>>({});
  const [decompositionParameters, setDecompositionParameters] = useState<Record<string, number>>({});
  const [autoTune, setAutoTune] = useState(true);
  const [autoTuneDecomposition, setAutoTuneDecomposition] = useState(true);
  const [job, setJob] = useState<ForecastJob | null>(null);
  const [runState, setRunState] = useState<RunState>("idle");
  const [serverState, setServerState] = useState<ServerState>("checking");
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchHealth()
      .then(() => setServerState("online"))
      .catch(() => setServerState("offline"));

    fetchModels()
      .then((items) => {
        setModels(items);
        const initialModel = items.find((item) => item.id === selectedModelId) ?? items[0];
        if (initialModel) {
          setSelectedModelId(initialModel.id);
          setParameters(defaultParameters(initialModel));
        }
      })
      .catch((reason: Error) => setError(reason.message));

    fetchDecompositions()
      .then((items) => {
        setDecompositions(items);
        const initialDecomposition = items.find((item) => item.id === selectedDecompositionId) ?? items[0];
        if (initialDecomposition) {
          setSelectedDecompositionId(initialDecomposition.id);
          setDecompositionParameters(defaultParameters(initialDecomposition));
        }
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId),
    [models, selectedModelId],
  );
  const selectedDecomposition = useMemo(
    () => decompositions.find((model) => model.id === selectedDecompositionId),
    [decompositions, selectedDecompositionId],
  );

  const chartData = useMemo(() => {
    if (!job?.history.length && !job?.forecast.length) {
      return [];
    }

    const lastHistoryIndex = (job?.history.length ?? 0) - 1;
    const history =
      job?.history.map((point, index) => ({
        date: point.date,
        actual: point.value,
        forecast: index === lastHistoryIndex && job.forecast.length ? point.value : null,
      })) ?? [];
    const forecast =
      job?.forecast.map((point) => ({
        date: point.date,
        actual: null,
        forecast: point.value,
      })) ?? [];
    return [...history, ...forecast];
  }, [job]);

  const displayHorizon = Number.isFinite(Number(horizonInput)) ? Number(horizonInput) : horizon;

  useEffect(() => {
    if (!job?.job_id || runState !== "polling") {
      return;
    }

    let cancelled = false;
    const refresh = async () => {
      try {
        const nextJob = await pollForecast(job.job_id);
        if (cancelled) {
          return;
        }
        setJob(nextJob);
        if (nextJob.status === "completed") {
          setRunState("completed");
        }
        if (nextJob.status === "failed") {
          setRunState("failed");
          setError(nextJob.error ?? "计算失败");
        }
      } catch (reason) {
        if (!cancelled) {
          setRunState("failed");
          setError(reason instanceof Error ? reason.message : "无法获取任务状态");
        }
      }
    };

    refresh();
    const timer = window.setInterval(refresh, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job?.job_id, runState]);

  function handleModelChange(modelId: ForecastModelId) {
    const nextModel = models.find((model) => model.id === modelId);
    setSelectedModelId(modelId);
    if (nextModel) {
      setParameters(defaultParameters(nextModel));
    }
  }

  function handleDecompositionChange(modelId: DecompositionModelId) {
    const nextModel = decompositions.find((model) => model.id === modelId);
    setSelectedDecompositionId(modelId);
    if (nextModel) {
      setDecompositionParameters(defaultParameters(nextModel));
    }
  }

  function handleCombinationChange(combinationId: CombinationModelId) {
    setSelectedCombinationId(combinationId);
  }

  async function acceptFile(nextFile: File | null) {
    if (!nextFile) {
      return;
    }
    if (!isSupportedDataFile(nextFile)) {
      setError("请上传 CSV 或 XLSX 文件。");
      return;
    }
    if (nextFile.size > MAX_FILE_SIZE) {
      setError("数据文件不能超过 10MB。");
      return;
    }
    setFile(nextFile);
    setPreview(null);
    setJob(null);
    setRunState("idle");
    setError("");
    if (isExcelFile(nextFile)) {
      setPreview({
        fileType: "Excel",
        columns: [],
        rows: [],
        totalRows: 0,
        issue: "Excel 文件会在提交后由后端解析，请确认首个工作表包含 date/time/ds 和 value/y/target 列。",
      });
      return;
    }
    try {
      const decoded = await readTextFile(nextFile);
      setPreview(parseCsvPreview(decoded.text, decoded.encoding));
    } catch (reason) {
      setPreview({
        fileType: "CSV",
        columns: [],
        rows: [],
        totalRows: 0,
        issue: reason instanceof Error ? reason.message : "无法读取 CSV 预览。",
      });
    }
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    void acceptFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragActive(false);
    void acceptFile(event.dataTransfer.files.item(0));
  }

  function loadSampleData() {
    void acceptFile(new File([SAMPLE_CSV], "sample-forecast.csv", { type: "text/csv" }));
  }

  function clearWorkspace() {
    setFile(null);
    setPreview(null);
    setJob(null);
    setRunState("idle");
    setError("");
  }

  function handleParameterChange(parameter: ModelParameter, value: string) {
    const nextValue = value === "" ? Number.NaN : Number(value);
    setParameters((current) => ({
      ...current,
      [parameter.name]: nextValue,
    }));
  }

  function handleDecompositionParameterChange(parameter: ModelParameter, value: string) {
    const nextValue = value === "" ? Number.NaN : Number(value);
    setDecompositionParameters((current) => ({
      ...current,
      [parameter.name]: nextValue,
    }));
  }

  function handleHorizonChange(value: string) {
    setHorizonInput(value);
    const nextValue = Number(value);
    if (Number.isInteger(nextValue) && nextValue >= 1 && nextValue <= 365) {
      setHorizon(nextValue);
    }
  }

  function normalizeHorizonInput() {
    const nextValue = Number(horizonInput);
    const normalized = clamp(Number.isFinite(nextValue) ? Math.round(nextValue) : horizon, 1, 365);
    setHorizon(normalized);
    setHorizonInput(String(normalized));
  }

  async function handleSubmit() {
    const currentFile = file;
    const currentModel = selectedModel;
    const currentDecomposition = selectedDecomposition;
    if (!currentFile || !currentModel || !currentDecomposition) {
      setError(currentFile ? "请选择分解方式和预测模型。" : "请先上传 CSV 或 XLSX 文件。");
      return;
    }

    const validationError = validateInputs(currentModel, horizon, parameters);
    if (validationError) {
      setError(validationError);
      return;
    }
    const decompositionValidationError = validateParameterInputs(currentDecomposition, decompositionParameters);
    if (decompositionValidationError) {
      setError(decompositionValidationError);
      return;
    }

    const normalizedHorizon = clamp(Math.round(horizon), 1, 365);
    const normalizedParameters = normalizeParameters(currentModel, parameters);
    const normalizedDecompositionParameters =
      currentDecomposition.id === "none" ? {} : normalizeParameters(currentDecomposition, decompositionParameters);
    setHorizon(normalizedHorizon);
    setHorizonInput(String(normalizedHorizon));
    setParameters(normalizedParameters);
    setDecompositionParameters(normalizedDecompositionParameters);
    setError("");
    setJob(null);
    setRunState("submitting");

    try {
      const created = await startForecast(
        currentFile,
        currentModel.id,
        normalizedHorizon,
        normalizedParameters,
        autoTune && currentModel.parameters.length > 0,
        currentDecomposition.id,
        normalizedDecompositionParameters,
        autoTuneDecomposition && currentDecomposition.id !== "none",
        selectedCombinationId,
      );
      setJob(created);
      setRunState("polling");
    } catch (reason) {
      setRunState("failed");
      setError(reason instanceof Error ? reason.message : "提交失败");
    }
  }

  const downloadForecastCsv = useCallback(() => {
    if (!job?.forecast.length) {
      return;
    }
    const rows = [
      ["date", "forecast"],
      ...job.forecast.map((point) => [point.date, String(point.value)]),
    ];
    downloadText(
      `forecast-${job.job_id.slice(0, 8)}.csv`,
      rows.map((row) => row.map(csvEscape).join(",")).join("\n"),
    );
  }, [job]);

  const isBusy = runState === "submitting" || runState === "polling";
  const canExport = job?.status === "completed" && job.forecast.length > 0;
  const canAutoTune = Boolean(selectedModel?.parameters.length);
  const canAutoTuneDecomposition = selectedDecompositionId !== "none" && Boolean(selectedDecomposition?.parameters.length);
  const summary = job?.summary;
  const decompositionResult = job?.decomposition_result;
  const jobUsage = job?.usage;
  const progressValue = clamp(job?.progress ?? 0, 0, 100);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <BarChart3 size={22} strokeWidth={2.2} />
          </div>
          <div>
            <h1>数据预测</h1>
            <p>时间序列建模工作台</p>
          </div>
        </div>
        <div className={`server-pill ${serverState}`}>
          <Server size={16} />
          {serverText[serverState]}
        </div>
      </header>

      <div className="workspace">
        <section className="left-rail">
          <Panel title="1 上传数据" icon={<UploadCloud size={18} />}>
            <label
              className={`file-drop ${dragActive ? "is-active" : ""}`}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
            >
              <FileUp size={30} />
              <span className="file-drop-title">{file ? file.name : "选择 CSV / XLSX 文件"}</span>
              <span className="file-drop-subtitle">
                {file ? `${formatFileSize(file.size)} · ${file.type || fileExtension(file.name)}` : "date,value / ds,y"}
              </span>
              <input
                className="sr-only"
                type="file"
                accept=".csv,text/csv,.xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                onChange={handleFileInput}
              />
            </label>
            <div className="button-row">
              <button className="control-button secondary" type="button" onClick={loadSampleData}>
                <FileSpreadsheet size={16} />
                示例数据
              </button>
              <button className="control-button ghost" disabled={!file && !job} type="button" onClick={clearWorkspace}>
                <Trash2 size={16} />
                清空
              </button>
            </div>
          </Panel>

          {preview && (
            <Panel title="数据预览" icon={<FileSpreadsheet size={18} />}>
              {preview.issue && <div className="preview-issue">{preview.issue}</div>}
              <div className="preview-grid">
                <Metric
                  label="文件类型"
                  value={preview.encoding ? `${preview.fileType} · ${preview.encoding}` : preview.fileType}
                />
                <Metric label="数据行数" value={preview.totalRows} />
                <Metric label="字段数量" value={preview.columns.length} />
                <Metric label="时间范围" value={preview.dateRange ?? "--"} />
                <Metric label="数值范围" value={preview.valueRange ?? "--"} />
              </div>
              {preview.columns.length > 0 && (
                <div className="preview-columns">
                  <span>识别列</span>
                  <strong>
                    {preview.dateColumn ?? "--"} / {preview.valueColumn ?? "--"}
                  </strong>
                </div>
              )}
              {preview.columns.length > 0 && (
                <>
                  <div className="table-wrap compact">
                    <table>
                      <thead>
                        <tr>
                          {preview.columns.slice(0, 4).map((column) => (
                            <th key={column}>{column}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {preview.rows.map((row, index) => (
                          <tr key={`${index}-${preview.columns[0]}`}>
                            {preview.columns.slice(0, 4).map((column) => (
                              <td key={column}>{row[column] || "--"}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </Panel>
          )}

          <Panel title="2 分解设置" icon={<Activity size={18} />}>
            <div className="form-row">
              <label htmlFor="decomposition">分解方式</label>
              <select
                id="decomposition"
                value={selectedDecompositionId}
                disabled={!decompositions.length || isBusy}
                onChange={(event) => handleDecompositionChange(event.target.value as DecompositionModelId)}
              >
                {decompositions.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>

            <label className="tune-toggle">
              <input
                checked={autoTuneDecomposition && canAutoTuneDecomposition}
                disabled={isBusy || !canAutoTuneDecomposition}
                type="checkbox"
                onChange={(event) => setAutoTuneDecomposition(event.target.checked)}
              />
              <span className="toggle-track" aria-hidden="true">
                <span />
              </span>
              <span>
                <strong>分解参数寻优</strong>
                <em>按回测误差选择更合适的分解参数。</em>
              </span>
            </label>

            {selectedDecomposition?.parameters.map((parameter) => {
              const value = decompositionParameters[parameter.name] ?? parameter.default;
              return (
                <div className="form-row" key={parameter.name}>
                  <label htmlFor={`decomposition-${parameter.name}`}>{parameter.label}</label>
                  <input
                    id={`decomposition-${parameter.name}`}
                    type="number"
                    min={parameter.min}
                    max={parameter.max}
                    step={parameter.step}
                    value={Number.isFinite(value) ? value : ""}
                    disabled={isBusy || (autoTuneDecomposition && canAutoTuneDecomposition)}
                    onChange={(event) => handleDecompositionParameterChange(parameter, event.target.value)}
                  />
                  <p>{parameter.meaning}</p>
                  <p className="recommendation">{parameter.recommendation}</p>
                </div>
              );
            })}

            {selectedDecomposition && (
              <div className="model-note">
                <strong>{selectedDecomposition.description}</strong>
                <span>{selectedDecomposition.best_for}</span>
              </div>
            )}
          </Panel>

          <Panel title="3 预测模型" icon={<Info size={18} />}>
            <div className="form-row">
              <label htmlFor="model">预测模型</label>
              <select
                id="model"
                value={selectedModelId}
                disabled={!models.length || isBusy}
                onChange={(event) => handleModelChange(event.target.value as ForecastModelId)}
              >
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-row">
              <label htmlFor="horizon">预测步数</label>
              <input
                id="horizon"
                type="number"
                min={1}
                max={365}
                step={1}
                value={horizonInput}
                disabled={isBusy}
                onBlur={normalizeHorizonInput}
                onChange={(event) => handleHorizonChange(event.target.value)}
              />
              <p>范围 1 到 365。</p>
            </div>

            <label className="tune-toggle">
              <input
                checked={autoTune && canAutoTune}
                disabled={isBusy || !canAutoTune}
                type="checkbox"
                onChange={(event) => setAutoTune(event.target.checked)}
              />
              <span className="toggle-track" aria-hidden="true">
                <span />
              </span>
              <span>
                <strong>自动寻参</strong>
                <em>预测前划分训练集和测试集，用回测误差选择参数。</em>
              </span>
            </label>

            {selectedModel?.parameters.map((parameter) => {
              const value = parameters[parameter.name] ?? parameter.default;
              return (
                <div className="form-row" key={parameter.name}>
                  <label htmlFor={parameter.name}>{parameter.label}</label>
                  <input
                    id={parameter.name}
                    type="number"
                    min={parameter.min}
                    max={parameter.max}
                    step={parameter.step}
                    value={Number.isFinite(value) ? value : ""}
                    disabled={isBusy || (autoTune && canAutoTune)}
                    onChange={(event) => handleParameterChange(parameter, event.target.value)}
                  />
                  <p>{parameter.meaning}</p>
                  <p className="recommendation">{parameter.recommendation}</p>
                </div>
              );
            })}

            {selectedModel && (
              <div className="model-note">
                <strong>{selectedModel.description}</strong>
                <span>{selectedModel.best_for}</span>
              </div>
            )}
          </Panel>

          <Panel title="4 组合模型" icon={<Layers size={18} />}>
            {selectedDecompositionId === "none" ? (
              <div className="model-note">
                <span>请先在「分解设置」中选择分解方式，才可配置组合模型。</span>
              </div>
            ) : (
              <>
                <div className="form-row">
                  <label htmlFor="combination">组合方式</label>
                  <select
                    id="combination"
                    value={selectedCombinationId}
                    disabled={isBusy}
                    onChange={(event) => handleCombinationChange(event.target.value as CombinationModelId)}
                  >
                    <option value="additive">加法模型</option>
                    <option value="pso_cs">PSO-CS 加权</option>
                  </select>
                </div>
                <p className="model-description">
                  {selectedCombinationId === "additive"
                    ? "将各子序列预测结果直接相加。计算快速，适合子序列独立性较强的场景。"
                    : "使用粒子群-杜鹃搜索混合算法优化各子序列权重，提升组合预测精度。计算耗时较长。"}
                </p>
              </>
            )}
            <button
              className="control-button primary submit-button"
              disabled={isBusy || !models.length}
              type="button"
              onClick={handleSubmit}
            >
              {isBusy ? <Loader2 className="animate-spin" size={18} /> : <Play size={18} />}
              {isBusy ? "计算中" : "开始计算"}
            </button>
          </Panel>
        </section>

        <section className="result-area">
          <Panel title="计算状态" icon={<Activity size={18} />}>
            <div className="status-grid">
              <Metric
                label="任务状态"
                tone={job?.status === "completed" ? "success" : job?.status === "failed" ? "danger" : "neutral"}
                value={job ? statusText[job.status] : "未提交"}
              />
              <Metric label="当前阶段" value={job?.stage ?? "--"} />
              <Metric label="已用时" value={formatDuration(job?.elapsed_ms)} />
              <Metric label="预测趋势" value={summary?.trend ?? "--"} />
            </div>

            {job && (
              <div className="progress-card">
                <div className="progress-heading">
                  <span>{job.stage || statusText[job.status]}</span>
                  <strong>{progressValue}%</strong>
                </div>
                <div
                  aria-label="计算进度"
                  aria-valuemax={100}
                  aria-valuemin={0}
                  aria-valuenow={progressValue}
                  className="progress-track"
                  role="progressbar"
                >
                  <span style={{ width: `${progressValue}%` }} />
                </div>
                <div className="usage-strip">
                  <span>
                    <em>文件</em>
                    <strong>{formatFileSize(jobUsage?.input_bytes ?? file?.size ?? 0)}</strong>
                  </span>
                  <span>
                    <em>历史点</em>
                    <strong>{formatNumber(jobUsage?.history_points ?? summary?.history_count ?? job.history.length)}</strong>
                  </span>
                  <span>
                    <em>预测步</em>
                    <strong>
                      {formatNumber(jobUsage?.forecast_steps ?? summary?.forecast_count ?? displayHorizon)}
                    </strong>
                  </span>
                  <span>
                    <em>参数组合</em>
                    <strong>
                      {job.auto_tune ? formatNumber(jobUsage?.parameter_candidates ?? 0) : "未开启"}
                    </strong>
                  </span>
                  <span>
                    <em>分解组合</em>
                    <strong>
                      {job.decomposition_result
                        ? job.decomposition_result.auto_tune
                          ? formatNumber(jobUsage?.decomposition_candidates ?? 0)
                          : "未开启"
                        : "未分解"}
                    </strong>
                  </span>
                  <span>
                    <em>子序列</em>
                    <strong>{formatNumber(jobUsage?.decomposition_components ?? decompositionResult?.component_count ?? 0)}</strong>
                  </span>
                  <span>
                    <em>模型运行</em>
                    <strong>{formatNumber(jobUsage?.model_runs ?? 0)} 次</strong>
                  </span>
                  <span>
                    <em>生成点</em>
                    <strong>{formatNumber(jobUsage?.generated_points ?? job.forecast.length)}</strong>
                  </span>
                </div>
              </div>
            )}

            {summary && (
              <div className="summary-strip">
                <span>末期预测 {formatNumber(summary.final_forecast)}</span>
                <span>均值 {formatNumber(summary.forecast_average)}</span>
                <span>
                  变化 {formatSignedNumber(summary.absolute_change)}
                  {summary.percentage_change == null ? "" : ` (${formatSignedPercent(summary.percentage_change)})`}
                </span>
              </div>
            )}

            {job?.auto_tune && job.status === "completed" && (
              <div className="tuning-result">
                <div>
                  <span>自动寻参</span>
                  <strong>
                    测试集 {job.tuning_test_size ?? "--"} 点，RMSE {formatNullableNumber(job.tuning_score)}
                  </strong>
                </div>
                <p>
                  {decompositionResult
                    ? "各子序列独立寻优，参数见分解结果。"
                    : formatParameters(job.parameters, selectedModel)}
                </p>
              </div>
            )}

            {isBusy && (
              <div className="progress-line">
                <Loader2 className="animate-spin" size={18} />
                服务器正在计算，结果会自动刷新。
              </div>
            )}
            {runState === "completed" && (
              <div className="success-line">
                <CheckCircle2 size={18} />
                计算完成。
              </div>
            )}
            {error && <div className="error-line">{error}</div>}
          </Panel>

          {decompositionResult && (
            <Panel title="分解结果" icon={<Activity size={18} />}>
              <div className="decomposition-summary">
                <Metric label="分解方式" value={decompositionLabel(decompositionResult.model, decompositions)} />
                <Metric label="子序列数" value={decompositionResult.component_count} />
                <Metric label="寻优 RMSE" value={formatNullableNumber(decompositionResult.tuning_score)} />
                <Metric label="分解参数" value={formatParameters(decompositionResult.parameters, selectedDecomposition)} />
              </div>
              <div className="component-list">
                {decompositionResult.components.map((component) => (
                  <div className="component-row" key={component.name}>
                    <div>
                      <strong>{component.name}</strong>
                      <span>
                        历史均值 {formatNumber(component.history_average)} · 区间{" "}
                        {formatNumber(component.history_min)} 至 {formatNumber(component.history_max)}
                      </span>
                      <span>预测参数 {formatParameters(component.parameters, selectedModel)}</span>
                    </div>
                    <div>
                      <strong>{formatNumber(component.forecast_final)}</strong>
                      <span>末期预测</span>
                    </div>
                  </div>
                ))}
              </div>
            </Panel>
          )}

          <ForecastChart chartData={chartData} />

          <ForecastResultTable canExport={canExport} job={job} onExport={downloadForecastCsv} />
        </section>
      </div>
    </main>
  );
}

function Panel({
  title,
  icon,
  actions,
  children,
}: {
  title: string;
  icon: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-title">
        <div className="panel-heading">
          {icon}
          <h2>{title}</h2>
        </div>
        {actions && <div className="panel-actions">{actions}</div>}
      </div>
      {children}
    </section>
  );
}

function Metric({ label, value, tone = "neutral" }: { label: string; value: string | number; tone?: MetricTone }) {
  return (
    <div className={`metric ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

const ForecastChart = memo(function ForecastChart({ chartData }: { chartData: ChartPoint[] }) {
  const chartDataMemo = useMemo(() => {
    if (chartData.length === 0) return null;

    return {
      labels: chartData.map((p) => p.date),
      datasets: [
        {
          label: "历史值",
          data: chartData.map((p) => p.actual),
          borderColor: "#2563eb",
          backgroundColor: "#2563eb",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0,
          spanGaps: true,
        },
        {
          label: "预测值",
          data: chartData.map((p) => p.forecast),
          borderColor: "#dc2626",
          backgroundColor: "#dc2626",
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 3,
          tension: 0,
          spanGaps: true,
        },
      ],
    };
  }, [chartData]);

  const options = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        legend: {
          position: "top" as const,
        },
        tooltip: {
          callbacks: {
            label: (ctx: { dataset: { label?: string }; raw: unknown }) =>
              `${ctx.dataset.label}: ${formatNumber(Number(ctx.raw))}`,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            maxTicksLimit: 10,
            font: { size: 12 },
          },
        },
        y: {
          ticks: {
            callback: (value: unknown) => formatNumber(Number(value)),
            font: { size: 12 },
          },
        },
      },
    }),
    [],
  );

  return (
    <Panel title="预测图表" icon={<BarChart3 size={18} />}>
      <div className="chart-box">
        {chartData.length > 0 && chartDataMemo ? (
          <div style={{ height: "85%", width: "100%" }}>
            <Line data={chartDataMemo} options={options} />
          </div>
        ) : (
          <div className="empty-state">提交任务后显示图表。</div>
        )}
      </div>
    </Panel>
  );
});

const ForecastResultTable = memo(function ForecastResultTable({
  canExport,
  job,
  onExport,
}: {
  canExport: boolean;
  job: ForecastJob | null;
  onExport: () => void;
}) {
  return (
    <Panel
      title="预测结果"
      icon={<CheckCircle2 size={18} />}
      actions={
        <button className="control-button secondary small" disabled={!canExport} type="button" onClick={onExport}>
          <Download size={15} />
          CSV
        </button>
      }
    >
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>序号</th>
              <th>日期</th>
              <th>预测值</th>
            </tr>
          </thead>
          <tbody>
            {job?.forecast.length ? (
              job.forecast.map((point, index) => (
                <tr key={point.date}>
                  <td>{index + 1}</td>
                  <td>{point.date}</td>
                  <td>{formatNumber(point.value)}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={3}>暂无结果</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Panel>
  );
});

function defaultParameters(model: ParameterOwner): Record<string, number> {
  return Object.fromEntries(model.parameters.map((parameter) => [parameter.name, parameter.default]));
}

function normalizeParameters(model: ParameterOwner, current: Record<string, number>): Record<string, number> {
  return Object.fromEntries(
    model.parameters.map((parameter) => {
      const fallback = parameter.default;
      const rawValue = Number.isFinite(current[parameter.name]) ? current[parameter.name] : fallback;
      const typedValue = parameter.type === "integer" ? Math.round(rawValue) : rawValue;
      return [parameter.name, clamp(typedValue, parameter.min, parameter.max)];
    }),
  );
}

function validateInputs(model: ModelInfo, horizon: number, current: Record<string, number>): string {
  if (!Number.isInteger(horizon) || horizon < 1 || horizon > 365) {
    return "预测步数必须是 1 到 365 的整数。";
  }
  return validateParameterInputs(model, current);
}

function validateParameterInputs(model: ParameterOwner, current: Record<string, number>): string {
  for (const parameter of model.parameters) {
    const value = current[parameter.name] ?? parameter.default;
    if (!Number.isFinite(value)) {
      return `${parameter.label}不能为空。`;
    }
    if (parameter.type === "integer" && !Number.isInteger(value)) {
      return `${parameter.label}必须是整数。`;
    }
    if (parameter.min != null && value < parameter.min) {
      return `${parameter.label}不能小于 ${parameter.min}。`;
    }
    if (parameter.max != null && value > parameter.max) {
      return `${parameter.label}不能大于 ${parameter.max}。`;
    }
  }

  return "";
}

function isSupportedDataFile(file: File): boolean {
  return isCsvFile(file) || isExcelFile(file);
}

function isCsvFile(file: File): boolean {
  return file.name.toLowerCase().endsWith(".csv") || file.type === "text/csv";
}

function isExcelFile(file: File): boolean {
  return file.name.toLowerCase().endsWith(".xlsx") || file.type === XLSX_MIME;
}

async function readTextFile(file: File): Promise<{ text: string; encoding: string }> {
  const buffer = await file.arrayBuffer();
  for (const encoding of ["utf-8", "gb18030"]) {
    try {
      return {
        text: new TextDecoder(encoding, { fatal: true }).decode(buffer),
        encoding: encoding.toUpperCase(),
      };
    } catch {
      // Try the next common encoding used by exported Chinese CSV files.
    }
  }
  return {
    text: new TextDecoder("utf-8").decode(buffer),
    encoding: "UTF-8",
  };
}

function fileExtension(filename: string): string {
  const dotIndex = filename.lastIndexOf(".");
  return dotIndex >= 0 ? filename.slice(dotIndex + 1).toUpperCase() : "unknown";
}

function parseCsvPreview(text: string, encoding: string): CsvPreview {
  const rows = parseCsvRows(text.replace(/^\uFEFF/, "").trim());
  if (rows.length < 2) {
    throw new Error("CSV 至少需要表头和一行数据。");
  }

  const columns = rows[0].map((column) => column.trim()).filter(Boolean);
  if (!columns.length) {
    throw new Error("CSV 表头为空。");
  }

  const normalized = new Map(columns.map((column) => [column.trim().toLowerCase(), column]));
  const dateColumn = normalized.get("date") ?? normalized.get("time") ?? normalized.get("ds");
  const valueColumn = normalized.get("value") ?? normalized.get("y") ?? normalized.get("target");
  const records = rows.slice(1).filter((row) => row.some((cell) => cell.trim()));
  const previewRows = records.slice(0, PREVIEW_LIMIT).map((row) => rowToRecord(columns, row));
  const dateRange = dateColumn ? summarizeDateColumn(records, columns.indexOf(dateColumn)) : undefined;
  const valueRange = valueColumn ? summarizeValueColumn(records, columns.indexOf(valueColumn)) : undefined;
  const issue = !dateColumn || !valueColumn ? "未识别到 date/time/ds 和 value/y/target 组合列，提交时可能失败。" : undefined;

  return {
    fileType: "CSV",
    encoding,
    columns,
    rows: previewRows,
    totalRows: records.length,
    dateColumn,
    valueColumn,
    dateRange,
    valueRange,
    issue,
  };
}

function parseCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const nextChar = text[index + 1];

    if (char === '"') {
      if (quoted && nextChar === '"') {
        cell += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
      continue;
    }

    if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && nextChar === "\n") {
        index += 1;
      }
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
      continue;
    }

    cell += char;
  }

  row.push(cell);
  rows.push(row);
  return rows;
}

function rowToRecord(columns: string[], row: string[]): Record<string, string> {
  return Object.fromEntries(columns.map((column, index) => [column, row[index]?.trim() ?? ""]));
}

function summarizeDateColumn(rows: string[][], columnIndex: number): string | undefined {
  const values = rows
    .map((row) => row[columnIndex]?.trim())
    .filter(Boolean)
    .sort();
  if (!values.length) {
    return undefined;
  }
  return `${values[0]} 至 ${values[values.length - 1]}`;
}

function summarizeValueColumn(rows: string[][], columnIndex: number): string | undefined {
  const values = rows
    .map((row) => Number(row[columnIndex]?.replace(/,/g, "")))
    .filter(Number.isFinite);
  if (!values.length) {
    return undefined;
  }
  return `${formatNumber(Math.min(...values))} 至 ${formatNumber(Math.max(...values))}`;
}

function clamp(value: number, min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY): number {
  return Math.min(Math.max(value, min), max);
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: 4,
  }).format(value);
}

function formatNullableNumber(value: number | null | undefined): string {
  return value == null ? "--" : formatNumber(value);
}

function formatParameters(parameters: Record<string, number>, model?: ParameterOwner): string {
  const labels = new Map(model?.parameters.map((parameter) => [parameter.name, parameter.label]) ?? []);
  const entries = Object.entries(parameters);
  if (!entries.length) {
    return "该模型没有可调参数。";
  }
  return entries.map(([key, value]) => `${labels.get(key) ?? key}: ${formatNumber(value)}`).join("，");
}

function decompositionLabel(modelId: DecompositionModelId, decompositions: DecompositionInfo[]): string {
  return decompositions.find((item) => item.id === modelId)?.name ?? modelId;
}

function formatSignedNumber(value: number): string {
  return `${value > 0 ? "+" : ""}${formatNumber(value)}`;
}

function formatSignedPercent(value: number): string {
  return `${value > 0 ? "+" : ""}${new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: 2,
  }).format(value)}%`;
}

function formatDuration(ms: number | null | undefined): string {
  if (ms == null) {
    return "--";
  }
  if (ms < 1000) {
    return `${Math.max(0, Math.round(ms))} ms`;
  }
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes > 0) {
    return `${minutes}分${String(seconds).padStart(2, "0")}秒`;
  }
  return `${seconds}秒`;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function csvEscape(value: string): string {
  if (/[",\n]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

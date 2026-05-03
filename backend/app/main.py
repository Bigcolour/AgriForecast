import asyncio
import json
import math
import uuid
from datetime import datetime, timezone
from statistics import mean
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .decomposition import (
    decomposition_candidates,
    decompose_values,
    default_decomposition_parameters,
    pso_cs_optimize_weights,
)
from .forecasting import (
    SUPPORTED_FILE_EXTENSIONS,
    auto_tune_parameters,
    choose_test_size,
    forecast_series,
    parameter_candidates,
    parse_input_file,
    rmse,
    summarize_forecast,
)
from .models import (
    CombinationModel,
    DecompositionComponentSummary,
    DecompositionInfo,
    DecompositionModel,
    DecompositionRun,
    ForecastModel,
    ForecastPoint,
    ForecastRequest,
    JobResult,
    JobUsage,
    ModelInfo,
    ModelParameter,
    SeriesPoint,
)


app = FastAPI(title="数据预测 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, JobResult] = {}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def elapsed_ms(started_at: datetime, finished_at: Optional[datetime] = None) -> int:
    end_time = finished_at or utc_now()
    return max(0, int((end_time - started_at).total_seconds() * 1000))


def update_job(job_id: str, **updates: object) -> None:
    job = jobs[job_id]
    for key, value in updates.items():
        if value is not None:
            setattr(job, key, value)
    if job.started_at and job.status in {"queued", "running"}:
        job.elapsed_ms = elapsed_ms(job.started_at)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    return [
        ModelInfo(
            id=ForecastModel.naive,
            name="朴素预测",
            description="直接把最后一个真实观测值作为所有未来点的预测值。",
            best_for="短期基线、数据波动较小或需要快速建立参考结果的场景。",
            parameters=[],
        ),
        ModelInfo(
            id=ForecastModel.moving_average,
            name="移动平均",
            description="使用最近一段窗口内的平均值预测下一期，并递推生成多期预测。",
            best_for="没有明显趋势、但存在随机波动的平稳序列。",
            parameters=[
                ModelParameter(
                    name="window",
                    label="窗口长度",
                    type="integer",
                    default=7,
                    min=1,
                    max=90,
                    step=1,
                    meaning="每次计算平均值时纳入最近多少个历史点。",
                    recommendation="日数据建议从 7 或 14 开始；周数据建议从 4 开始。窗口越大，结果越平滑。",
                ),
                ModelParameter(
                    name="volatility",
                    label="波动保留",
                    type="number",
                    default=0.45,
                    min=0,
                    max=1.5,
                    step=0.05,
                    meaning="把历史短期波动模式带入未来预测，0 表示完全平滑，数值越大波动越明显。",
                    recommendation="一般从 0.3 到 0.7 开始；业务波动强时可提高到 1.0 左右。",
                ),
            ],
        ),
        ModelInfo(
            id=ForecastModel.exponential_smoothing,
            name="简单指数平滑",
            description="使用指数衰减权重融合历史值，越新的数据权重越高。",
            best_for="水平变化缓慢、无强趋势或强季节性的业务指标。",
            parameters=[
                ModelParameter(
                    name="alpha",
                    label="平滑系数",
                    type="number",
                    default=0.35,
                    min=0.01,
                    max=1,
                    step=0.01,
                    meaning="控制新数据的权重。越接近 1，预测越贴近最近变化；越接近 0，预测越平稳。",
                    recommendation="一般从 0.2 到 0.4 开始；波动较快的数据可提高到 0.6 左右。",
                ),
                ModelParameter(
                    name="volatility",
                    label="波动保留",
                    type="number",
                    default=0.4,
                    min=0,
                    max=1.5,
                    step=0.05,
                    meaning="把历史短期波动模式带入未来预测，0 表示完全平滑，数值越大波动越明显。",
                    recommendation="指数平滑容易变平，建议保留 0.3 到 0.6 的波动。",
                ),
            ],
        ),
        ModelInfo(
            id=ForecastModel.linear_trend,
            name="线性趋势",
            description="拟合一条线性趋势线，并按趋势向未来延伸。",
            best_for="存在持续上升或下降趋势、季节性不强的序列。",
            parameters=[
                ModelParameter(
                    name="damping",
                    label="趋势阻尼",
                    type="number",
                    default=1,
                    min=0,
                    max=1.5,
                    step=0.05,
                    meaning="控制趋势延伸强度。1 表示完全沿用历史趋势，0 表示去除趋势。",
                    recommendation="保守预测建议 0.7 到 1.0；趋势可能继续加速时才考虑大于 1。",
                ),
                ModelParameter(
                    name="volatility",
                    label="波动保留",
                    type="number",
                    default=0.35,
                    min=0,
                    max=1.5,
                    step=0.05,
                    meaning="把历史短期波动模式带入未来预测，0 表示完全平滑，数值越大波动越明显。",
                    recommendation="趋势模型建议从 0.25 到 0.6 开始，避免盖过主趋势。",
                ),
            ],
        ),
        ModelInfo(
            id=ForecastModel.ets,
            name="ETS 指数平滑",
            description="使用水平、趋势和可选季节项递推预测，适合带趋势或周期波动的序列。",
            best_for="日销售、访问量、库存等有平滑趋势和固定周期的数据。",
            parameters=[
                ModelParameter(
                    name="alpha",
                    label="水平平滑",
                    type="number",
                    default=0.35,
                    min=0.01,
                    max=1,
                    step=0.01,
                    meaning="控制最新观测值对水平项的影响。",
                    recommendation="波动快的数据可提高到 0.5 以上；稳定数据建议 0.2 到 0.4。",
                ),
                ModelParameter(
                    name="beta",
                    label="趋势平滑",
                    type="number",
                    default=0.12,
                    min=0,
                    max=1,
                    step=0.01,
                    meaning="控制趋势项更新速度。",
                    recommendation="趋势稳定时使用 0.05 到 0.2；趋势变化快时可提高。",
                ),
                ModelParameter(
                    name="gamma",
                    label="季节平滑",
                    type="number",
                    default=0.1,
                    min=0,
                    max=1,
                    step=0.01,
                    meaning="控制季节项更新速度；季节长度为 1 时不启用季节项。",
                    recommendation="有明显周周期的日数据可配合季节长度 7 使用。",
                ),
                ModelParameter(
                    name="season_length",
                    label="季节长度",
                    type="integer",
                    default=1,
                    min=1,
                    max=365,
                    step=1,
                    meaning="一个完整季节周期包含的数据点数量。",
                    recommendation="无季节性用 1；日数据周周期用 7；月度年度周期用 12。",
                ),
                ModelParameter(
                    name="volatility",
                    label="波动保留",
                    type="number",
                    default=0.3,
                    min=0,
                    max=1.5,
                    step=0.05,
                    meaning="把季节项之外的短期波动带入未来预测。",
                    recommendation="ETS 已有季节项，通常保留 0.2 到 0.5 即可。",
                ),
            ],
        ),
        ModelInfo(
            id=ForecastModel.elm,
            name="ELM 极限学习机",
            description="把历史窗口映射到随机隐藏层，并用岭回归训练输出层进行递推预测。",
            best_for="中小规模单变量序列，需要快速非线性基线模型的场景。",
            parameters=[
                ModelParameter(
                    name="lookback",
                    label="回看窗口",
                    type="integer",
                    default=7,
                    min=2,
                    max=60,
                    step=1,
                    meaning="每次预测使用最近多少个历史点作为输入。",
                    recommendation="日数据可从 7 或 14 开始；数据很短时降低窗口。",
                ),
                ModelParameter(
                    name="hidden_units",
                    label="隐藏节点",
                    type="integer",
                    default=16,
                    min=4,
                    max=128,
                    step=1,
                    meaning="随机隐藏层节点数量，越大表达能力越强但更容易过拟合。",
                    recommendation="普通业务数据建议 12 到 32；数据量较小时不要过大。",
                ),
                ModelParameter(
                    name="regularization",
                    label="正则强度",
                    type="number",
                    default=0.05,
                    min=0.0001,
                    max=10,
                    step=0.01,
                    meaning="岭回归正则项，控制模型稳定性。",
                    recommendation="结果抖动时提高；欠拟合时降低。",
                ),
                ModelParameter(
                    name="volatility",
                    label="波动保留",
                    type="number",
                    default=0.3,
                    min=0,
                    max=1.5,
                    step=0.05,
                    meaning="把历史短期波动模式带入未来预测，弥补递推模型逐步变平的问题。",
                    recommendation="ELM 建议从 0.2 到 0.6 开始。",
                ),
            ],
        ),
        ModelInfo(
            id=ForecastModel.lstm,
            name="LSTM 循环网络",
            description="使用轻量 LSTM 风格门控循环单元提取时序状态，并训练输出层递推预测。",
            best_for="存在非线性变化、近期状态对未来影响较强的单变量序列。",
            parameters=[
                ModelParameter(
                    name="lookback",
                    label="回看窗口",
                    type="integer",
                    default=10,
                    min=3,
                    max=90,
                    step=1,
                    meaning="模型每次编码最近多少个历史点。",
                    recommendation="日数据建议从 10 到 30；历史数据较少时降低窗口。",
                ),
                ModelParameter(
                    name="hidden_units",
                    label="隐藏单元",
                    type="integer",
                    default=12,
                    min=4,
                    max=64,
                    step=1,
                    meaning="循环状态维度，越大能表示更复杂模式。",
                    recommendation="中小数据建议 8 到 24；过大可能不稳定。",
                ),
                ModelParameter(
                    name="regularization",
                    label="正则强度",
                    type="number",
                    default=0.08,
                    min=0.0001,
                    max=10,
                    step=0.01,
                    meaning="输出层岭回归正则项，控制预测稳定性。",
                    recommendation="预测过度波动时提高；响应太慢时降低。",
                ),
                ModelParameter(
                    name="volatility",
                    label="波动保留",
                    type="number",
                    default=0.25,
                    min=0,
                    max=1.5,
                    step=0.05,
                    meaning="把历史短期波动模式带入未来预测，弥补递推模型逐步变平的问题。",
                    recommendation="LSTM 风格模型建议从 0.2 到 0.5 开始。",
                ),
            ],
        ),
    ]


@app.get("/api/decompositions", response_model=list[DecompositionInfo])
def list_decompositions() -> list[DecompositionInfo]:
    return [
        DecompositionInfo(
            id=DecompositionModel.none,
            name="不分解",
            description="直接使用原始价格序列进行预测。",
            best_for="数据量较少、需要快速得到基线结果，或不希望引入额外参数的场景。",
            parameters=[],
        ),
        DecompositionInfo(
            id=DecompositionModel.vmd,
            name="VMD 变分模态分解",
            description="把价格序列拆成若干带宽受限的 IMF 分量，再分别预测并相加组合。",
            best_for="混合了低频趋势和高频波动的价格序列，尤其适合波动形态较复杂的数据。",
            parameters=[
                ModelParameter(
                    name="modes",
                    label="模态数量",
                    type="integer",
                    default=5,
                    min=2,
                    max=8,
                    step=1,
                    meaning="VMD 拆出的 IMF 分量数量。数量越多，分解越细，但计算量和过拟合风险也更高。",
                    recommendation="日价格序列可从 5 或 6 开始；数据较短时使用 3 到 5。",
                ),
                ModelParameter(
                    name="alpha",
                    label="带宽约束",
                    type="number",
                    default=3000,
                    min=100,
                    max=10000,
                    step=100,
                    meaning="控制每个模态的带宽。数值越大，分量越平滑、频带越窄。",
                    recommendation="常用 1000 到 3000；波动很强时可尝试 3000 以上。",
                ),
            ],
        ),
        DecompositionInfo(
            id=DecompositionModel.ssa,
            name="SSA 奇异谱分析",
            description="使用轨迹矩阵和 SVD 将序列拆成趋势、振荡和残差分量。",
            best_for="存在趋势或缓慢周期变化，且希望得到稳定、可解释分量的价格序列。",
            parameters=[
                ModelParameter(
                    name="window",
                    label="窗口长度",
                    type="integer",
                    default=6,
                    min=3,
                    max=30,
                    step=1,
                    meaning="构造 SSA 轨迹矩阵时使用的嵌入窗口长度。",
                    recommendation="可从 6 开始；周期明显时可设置为周期长度或其倍数。",
                ),
                ModelParameter(
                    name="components",
                    label="保留分量",
                    type="integer",
                    default=6,
                    min=2,
                    max=30,
                    step=1,
                    meaning="保留并分别预测的 SSA 分量数量，剩余部分会进入残差分量。",
                    recommendation="通常不超过窗口长度；想要简单结果时使用 3 到 6。",
                ),
            ],
        ),
        DecompositionInfo(
            id=DecompositionModel.ewt,
            name="EWT 经验小波分解",
            description="根据频谱局部峰值自动划分频带，再用 Meyer 滤波器提取子序列。",
            best_for="频谱结构较清晰、含多个振荡频段的价格序列。",
            parameters=[
                ModelParameter(
                    name="bands",
                    label="频带数量",
                    type="integer",
                    default=5,
                    min=2,
                    max=8,
                    step=1,
                    meaning="EWT 划分出的频带数量，每个频带会形成一个待预测子序列。",
                    recommendation="一般从 5 开始；数据短或噪声大时降低到 3。",
                ),
            ],
        ),
    ]


@app.post("/api/forecast", response_model=JobResult)
async def create_forecast(request: str = Form(...), file: UploadFile = File(...)) -> JobResult:
    filename = file.filename or ""
    if not filename.lower().endswith(SUPPORTED_FILE_EXTENSIONS):
        raise HTTPException(status_code=400, detail="请上传 CSV 或 XLSX 文件。")
    try:
        forecast_request = ForecastRequest.model_validate(json.loads(request))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="预测参数格式不正确。") from exc
    content = await file.read()
    job_id = str(uuid.uuid4())
    submitted_at = utc_now()
    jobs[job_id] = JobResult(
        job_id=job_id,
        status="queued",
        stage="已提交",
        progress=5,
        model=forecast_request.model,
        parameters=forecast_request.parameters,
        auto_tune=forecast_request.auto_tune,
        submitted_at=submitted_at,
        usage=JobUsage(input_bytes=len(content), forecast_steps=forecast_request.horizon),
    )
    asyncio.create_task(run_forecast(job_id, content, filename, forecast_request))
    return jobs[job_id]


@app.get("/api/forecast/{job_id}", response_model=JobResult)
def get_forecast(job_id: str) -> JobResult:
    result = jobs.get(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="任务不存在。")
    if result.started_at and result.status in {"queued", "running"}:
        result.elapsed_ms = elapsed_ms(result.started_at)
    return result


async def run_forecast(job_id: str, content: bytes, filename: str, request: ForecastRequest) -> None:
    update_job(job_id, status="running", stage="准备计算", progress=10, started_at=utc_now())
    try:
        await asyncio.to_thread(run_forecast_sync, job_id, content, filename, request)
    except Exception as exc:
        fail_job(job_id, request, exc)


def run_forecast_sync(job_id: str, content: bytes, filename: str, request: ForecastRequest) -> None:
    update_job(job_id, stage="解析数据", progress=18)
    history = parse_input_file(content, filename)
    usage = jobs[job_id].usage
    usage.history_points = len(history)
    usage.forecast_steps = request.horizon
    update_job(job_id, stage="数据解析完成", progress=28, usage=usage)

    if request.decomposition == DecompositionModel.none:
        parameters, tuning_score, tuning_test_size, forecast, decomposition_result = run_direct_pipeline(
            job_id, history, request, usage
        )
    else:
        parameters, tuning_score, tuning_test_size, forecast, decomposition_result = run_decomposed_pipeline(
            job_id, history, request, usage
        )

    usage.generated_points = len(forecast)
    update_job(job_id, stage="汇总结果", progress=90, usage=usage)
    summary = summarize_forecast(history, forecast)

    finished_at = utc_now()
    started_at = jobs[job_id].started_at
    jobs[job_id] = JobResult(
        job_id=job_id,
        status="completed",
        stage="计算完成",
        progress=100,
        model=request.model,
        parameters=parameters,
        auto_tune=request.auto_tune,
        tuning_score=tuning_score,
        tuning_test_size=tuning_test_size,
        submitted_at=jobs[job_id].submitted_at,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_ms=elapsed_ms(started_at, finished_at) if started_at else None,
        usage=usage,
        history=history,
        forecast=forecast,
        decomposition_result=decomposition_result,
        summary=summary,
    )


def run_direct_pipeline(
    job_id: str,
    history: list[SeriesPoint],
    request: ForecastRequest,
    usage: JobUsage,
) -> tuple[Dict[str, float], Optional[float], Optional[int], list[ForecastPoint], None]:
    parameters = request.parameters
    tuning_score = None
    tuning_test_size = None
    if request.auto_tune:
        parameters, tuning_score, tuning_test_size = tune_forecast_parameters(
            job_id,
            "预测模型自动寻参",
            history,
            request,
            usage,
            progress_start=35,
            progress_span=35,
        )
    else:
        update_job(job_id, stage="准备生成预测", progress=62)

    update_job(job_id, stage="生成预测", progress=78)
    forecast = forecast_series(history, request.model, request.horizon, parameters)
    usage.model_runs += 1
    return parameters, tuning_score, tuning_test_size, forecast, None


def run_decomposed_pipeline(
    job_id: str,
    history: list[SeriesPoint],
    request: ForecastRequest,
    usage: JobUsage,
) -> tuple[Dict[str, float], Optional[float], Optional[int], list[ForecastPoint], DecompositionRun]:
    decomposition_parameters = normalized_decomposition_parameters(
        request.decomposition,
        request.decomposition_parameters,
    )
    decomposition_score = None
    decomposition_test_size = None
    if request.auto_tune_decomposition:
        decomposition_parameters, decomposition_score, decomposition_test_size = tune_decomposition_parameters(
            job_id,
            history,
            request,
            decomposition_parameters,
            usage,
        )
    else:
        update_job(job_id, stage="准备分解数据", progress=38)

    update_job(job_id, stage=f"执行{request.decomposition.value.upper()}分解", progress=48)
    values = [point.value for point in history]
    decomposed = decompose_values(values, request.decomposition, decomposition_parameters)
    usage.decomposition_components = len(decomposed.components)
    update_job(
        job_id,
        stage=f"分解完成：{len(decomposed.components)} 个子序列",
        progress=55,
        usage=usage,
    )

    component_forecasts = []
    component_parameters = []
    component_scores = []
    component_test_sizes = []
    dates = [point.date for point in history]
    component_count = len(decomposed.components)
    for index, (name, component_values) in enumerate(zip(decomposed.names, decomposed.components), start=1):
        component_history = series_from_values(dates, component_values)
        progress_start = 58 + round((index - 1) / max(component_count, 1) * 24)
        progress_span = max(3, round(24 / max(component_count, 1)))
        if request.auto_tune:
            parameters, score, test_size = tune_forecast_parameters(
                job_id,
                f"{name} 自动寻参",
                component_history,
                request,
                usage,
                progress_start=progress_start,
                progress_span=progress_span,
            )
            if score is not None:
                component_scores.append(score)
            if test_size:
                component_test_sizes.append(test_size)
        else:
            parameters = request.parameters
        update_job(job_id, stage=f"预测子序列 {index}/{component_count}", progress=min(progress_start + progress_span, 84))
        forecast = forecast_series(component_history, request.model, request.horizon, parameters)
        usage.model_runs += 1
        component_forecasts.append(forecast)
        component_parameters.append(parameters)

    if request.combination == CombinationModel.pso_cs and component_count > 1:
        update_job(job_id, stage="PSO-CS优化权重", progress=86)
        weights = optimize_pso_cs_component_weights(
            history,
            request,
            decomposition_parameters,
            component_parameters,
            component_count,
            usage,
        )
        combined_forecast = combine_with_weights(component_forecasts, weights)
    else:
        combined_forecast = combine_component_forecasts(component_forecasts)
    decomposition_run = DecompositionRun(
        model=request.decomposition,
        parameters=decomposition_parameters,
        auto_tune=request.auto_tune_decomposition,
        tuning_score=decomposition_score,
        tuning_test_size=decomposition_test_size,
        component_count=component_count,
        component_names=decomposed.names,
        components=[
            component_summary(name, values, forecast, parameters)
            for name, values, forecast, parameters in zip(
                decomposed.names,
                decomposed.components,
                component_forecasts,
                component_parameters,
            )
        ],
    )
    model_tuning_score = round(mean(component_scores), 4) if component_scores else None
    model_tuning_test_size = component_test_sizes[0] if component_test_sizes else None
    return request.parameters, model_tuning_score, model_tuning_test_size, combined_forecast, decomposition_run


def tune_forecast_parameters(
    job_id: str,
    label: str,
    history: list[SeriesPoint],
    request: ForecastRequest,
    usage: JobUsage,
    progress_start: int,
    progress_span: int,
) -> tuple[Dict[str, float], float, int]:
    candidates = parameter_candidates(request.model, request.parameters, len(history))
    usage.parameter_candidates += len(candidates)
    base_runs = usage.model_runs
    update_job(job_id, stage=f"{label} 0/{len(candidates)}", progress=progress_start, usage=usage)

    def track_tuning(attempted: int, total: int) -> None:
        usage.model_runs = base_runs + attempted
        progress = progress_start + round((attempted / max(total, 1)) * progress_span)
        update_job(job_id, stage=f"{label} {attempted}/{total}", progress=min(progress, 88), usage=usage)

    return auto_tune_parameters(
        history,
        request.model,
        request.parameters,
        progress_callback=track_tuning,
    )


def tune_decomposition_parameters(
    job_id: str,
    history: list[SeriesPoint],
    request: ForecastRequest,
    current_parameters: Dict[str, float],
    usage: JobUsage,
) -> tuple[Dict[str, float], float, int]:
    candidates = decomposition_candidates(request.decomposition, current_parameters, len(history))
    usage.decomposition_candidates = len(candidates)
    update_job(job_id, stage=f"分解参数寻优 0/{len(candidates)}", progress=32, usage=usage)

    test_size = choose_test_size(len(history))
    if test_size < 1 or len(history) - test_size < 6:
        raise ValueError("分解参数自动寻优至少需要 9 个历史数据点。")

    train = history[:-test_size]
    actual = [point.value for point in history[-test_size:]]
    best_parameters = current_parameters
    best_score = float("inf")
    evaluated = 0

    for attempted, candidate in enumerate(candidates, start=1):
        try:
            predicted = backtest_decomposition_candidate(train, request, candidate, test_size, usage)
            score = rmse(actual, predicted)
        except ValueError:
            pass
        else:
            if math.isfinite(score) and score < best_score:
                best_score = score
                best_parameters = candidate
            evaluated += 1
        progress = 32 + round((attempted / max(len(candidates), 1)) * 14)
        update_job(
            job_id,
            stage=f"分解参数寻优 {attempted}/{len(candidates)}",
            progress=progress,
            usage=usage,
        )

    if evaluated == 0 or not math.isfinite(best_score):
        raise ValueError("分解参数自动寻优没有找到可用组合，请减少分解数量或关闭分解寻优。")

    return best_parameters, round(best_score, 4), test_size


def backtest_decomposition_candidate(
    train: list[SeriesPoint],
    request: ForecastRequest,
    decomposition_parameters: Dict[str, float],
    horizon: int,
    usage: JobUsage,
) -> list[float]:
    values = [point.value for point in train]
    decomposed = decompose_values(values, request.decomposition, decomposition_parameters)
    dates = [point.date for point in train]
    component_forecasts = []
    for component_values in decomposed.components:
        component_history = series_from_values(dates, component_values)
        component_forecasts.append(forecast_series(component_history, request.model, horizon, request.parameters))
        usage.model_runs += 1
    return [point.value for point in combine_component_forecasts(component_forecasts)]


def optimize_pso_cs_component_weights(
    history: list[SeriesPoint],
    request: ForecastRequest,
    decomposition_parameters: Dict[str, float],
    component_parameters: list[Dict[str, float]],
    component_count: int,
    usage: JobUsage,
) -> list[float]:
    baseline_weights = [1.0] * component_count
    test_size = choose_test_size(len(history))
    if test_size < 1 or len(history) - test_size < 6:
        return baseline_weights

    train = history[:-test_size]
    actual = [point.value for point in history[-test_size:]]
    train_values = [point.value for point in train]
    try:
        decomposed = decompose_values(train_values, request.decomposition, decomposition_parameters)
    except ValueError:
        return baseline_weights
    if len(decomposed.components) != component_count:
        return baseline_weights

    dates = [point.date for point in train]
    validation_component_predictions = []
    for index, component_values in enumerate(decomposed.components):
        component_history = series_from_values(dates, component_values)
        parameters = component_parameters[index] if index < len(component_parameters) else request.parameters
        forecast = forecast_series(component_history, request.model, test_size, parameters)
        validation_component_predictions.append([point.value for point in forecast])
        usage.model_runs += 1

    weights = pso_cs_optimize_weights(
        validation_component_predictions,
        actual,
        baseline_weights=baseline_weights,
    )
    if not pso_cs_beats_additive(validation_component_predictions, actual, weights):
        return baseline_weights
    return weights


def pso_cs_beats_additive(
    component_predictions: list[list[float]],
    actual: list[float],
    weights: list[float],
) -> bool:
    if not component_predictions or len(weights) != len(component_predictions):
        return False
    validation_count = min(len(actual), *(len(prediction) for prediction in component_predictions))
    if validation_count < 1:
        return False

    additive = [
        sum(prediction[index] for prediction in component_predictions)
        for index in range(validation_count)
    ]
    weighted = [
        sum(
            prediction[index] * weights[component_index]
            for component_index, prediction in enumerate(component_predictions)
        )
        for index in range(validation_count)
    ]
    actual_slice = actual[:validation_count]
    return rmse(actual_slice, weighted) <= rmse(actual_slice, additive)


def normalized_decomposition_parameters(
    model: DecompositionModel,
    current_parameters: Dict[str, float],
) -> Dict[str, float]:
    defaults = default_decomposition_parameters(model)
    return {**defaults, **current_parameters}


def series_from_values(dates: list[str], values: list[float]) -> list[SeriesPoint]:
    return [SeriesPoint(date=item_date, value=float(value)) for item_date, value in zip(dates, values)]


def combine_component_forecasts(component_forecasts: list[list[ForecastPoint]]) -> list[ForecastPoint]:
    if not component_forecasts:
        raise ValueError("没有可组合的子序列预测结果。")
    forecast_count = len(component_forecasts[0])
    combined = []
    for index in range(forecast_count):
        combined.append(
            ForecastPoint(
                date=component_forecasts[0][index].date,
                value=round(sum(forecast[index].value for forecast in component_forecasts), 4),
            )
        )
    return combined


def combine_with_weights(component_forecasts: list[list[ForecastPoint]], weights: list[float]) -> list[ForecastPoint]:
    if not component_forecasts or len(weights) != len(component_forecasts):
        raise ValueError("没有可组合的子序列预测结果或权重。")
    forecast_count = len(component_forecasts[0])
    combined = []
    for index in range(forecast_count):
        combined_value = sum(forecast[index].value * weights[i] for i, forecast in enumerate(component_forecasts))
        combined.append(
            ForecastPoint(
                date=component_forecasts[0][index].date,
                value=round(combined_value, 4),
            )
        )
    return combined


def component_summary(
    name: str,
    values: list[float],
    forecast: list[ForecastPoint],
    parameters: Dict[str, float],
) -> DecompositionComponentSummary:
    forecast_values = [point.value for point in forecast]
    return DecompositionComponentSummary(
        name=name,
        history_min=round(min(values), 4),
        history_max=round(max(values), 4),
        history_average=round(mean(values), 4),
        forecast_final=round(forecast_values[-1], 4),
        forecast_average=round(mean(forecast_values), 4),
        parameters=parameters,
    )


def fail_job(job_id: str, request: ForecastRequest, exc: Exception) -> None:
    current = jobs[job_id]
    finished_at = utc_now()
    current_finished_progress = max(current.progress, 10)
    jobs[job_id] = JobResult(
        job_id=job_id,
        status="failed",
        stage="计算失败",
        progress=current_finished_progress,
        model=request.model,
        parameters=request.parameters,
        auto_tune=request.auto_tune,
        submitted_at=current.submitted_at,
        started_at=current.started_at,
        finished_at=finished_at,
        elapsed_ms=elapsed_ms(current.started_at, finished_at) if current.started_at else None,
        usage=current.usage,
        error=str(exc),
    )

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ForecastModel(str, Enum):
    naive = "naive"
    moving_average = "moving_average"
    exponential_smoothing = "exponential_smoothing"
    linear_trend = "linear_trend"
    ets = "ets"
    elm = "elm"
    lstm = "lstm"


class DecompositionModel(str, Enum):
    none = "none"
    vmd = "vmd"
    ssa = "ssa"
    ewt = "ewt"


class ForecastRequest(BaseModel):
    model: ForecastModel
    horizon: int = Field(default=7, ge=1, le=365)
    parameters: Dict[str, float] = Field(default_factory=dict)
    auto_tune: bool = False
    decomposition: DecompositionModel = DecompositionModel.none
    decomposition_parameters: Dict[str, float] = Field(default_factory=dict)
    auto_tune_decomposition: bool = False


class SeriesPoint(BaseModel):
    date: str
    value: float


class ForecastPoint(BaseModel):
    date: str
    value: float


class ForecastSummary(BaseModel):
    history_count: int
    forecast_count: int
    last_actual: float
    first_forecast: float
    final_forecast: float
    forecast_average: float
    forecast_min: float
    forecast_max: float
    absolute_change: float
    percentage_change: Optional[float] = None
    trend: str


class JobUsage(BaseModel):
    input_bytes: int = 0
    history_points: int = 0
    forecast_steps: int = 0
    parameter_candidates: int = 0
    decomposition_candidates: int = 0
    decomposition_components: int = 0
    model_runs: int = 0
    generated_points: int = 0


class DecompositionComponentSummary(BaseModel):
    name: str
    history_min: float
    history_max: float
    history_average: float
    forecast_final: float
    forecast_average: float
    parameters: Dict[str, float] = Field(default_factory=dict)


class DecompositionRun(BaseModel):
    model: DecompositionModel
    parameters: Dict[str, float] = Field(default_factory=dict)
    auto_tune: bool = False
    tuning_score: Optional[float] = None
    tuning_test_size: Optional[int] = None
    component_count: int = 0
    component_names: List[str] = Field(default_factory=list)
    components: List[DecompositionComponentSummary] = Field(default_factory=list)


class JobResult(BaseModel):
    job_id: str
    status: str
    stage: str = "等待开始"
    progress: int = Field(default=0, ge=0, le=100)
    model: Optional[ForecastModel] = None
    parameters: Dict[str, float] = Field(default_factory=dict)
    auto_tune: bool = False
    tuning_score: Optional[float] = None
    tuning_test_size: Optional[int] = None
    submitted_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    elapsed_ms: Optional[int] = None
    usage: JobUsage = Field(default_factory=JobUsage)
    history: List[SeriesPoint] = Field(default_factory=list)
    forecast: List[ForecastPoint] = Field(default_factory=list)
    decomposition_result: Optional[DecompositionRun] = None
    summary: Optional[ForecastSummary] = None
    error: Optional[str] = None


class ModelParameter(BaseModel):
    name: str
    label: str
    type: str
    default: float
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    meaning: str
    recommendation: str


class ModelInfo(BaseModel):
    id: ForecastModel
    name: str
    description: str
    best_for: str
    parameters: List[ModelParameter]


class DecompositionInfo(BaseModel):
    id: DecompositionModel
    name: str
    description: str
    best_for: str
    parameters: List[ModelParameter]

import csv
import io
import math
import posixpath
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from statistics import mean
from typing import Callable, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np

from .models import ForecastModel, ForecastPoint, ForecastSummary, SeriesPoint


def sigmoid_vectorized(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S")
SUPPORTED_FILE_EXTENSIONS = (".csv", ".xlsx")
XML_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def parse_input_file(content: bytes, filename: str) -> List[SeriesPoint]:
    lower_name = (filename or "").lower()
    if lower_name.endswith(".csv"):
        return parse_csv(content)
    if lower_name.endswith(".xlsx"):
        return parse_xlsx(content)
    raise ValueError("请上传 CSV 或 XLSX 文件。")


def parse_csv(content: bytes) -> List[SeriesPoint]:
    text = decode_text(content)
    rows = list(csv.reader(io.StringIO(text)))
    return parse_tabular_rows(rows, "CSV")


def decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("文件编码无法解析，请使用 UTF-8 或 GB18030 编码。")


def parse_xlsx(content: bytes) -> List[SeriesPoint]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as workbook:
            shared_strings = read_shared_strings(workbook)
            sheet_path = first_sheet_path(workbook)
            rows = read_sheet_rows(workbook, sheet_path, shared_strings)
    except zipfile.BadZipFile as exc:
        raise ValueError("XLSX 文件无法打开，请确认文件没有损坏。") from exc
    except KeyError as exc:
        raise ValueError("XLSX 文件缺少工作表数据，请重新导出后再上传。") from exc
    except ET.ParseError as exc:
        raise ValueError("XLSX 文件结构无法解析，请重新导出后再上传。") from exc
    return parse_tabular_rows(rows, "XLSX")


def parse_tabular_rows(rows: List[List[str]], source_label: str) -> List[SeriesPoint]:
    non_empty_rows = [row for row in rows if any((cell or "").strip() for cell in row)]
    if not non_empty_rows:
        raise ValueError(f"{source_label} 文件为空，或缺少表头。")

    headers = [cell.strip() for cell in non_empty_rows[0]]
    normalized = {field.lower(): index for index, field in enumerate(headers) if field}
    date_index = first_present_index(normalized, ("date", "time", "ds"))
    value_index = first_present_index(normalized, ("value", "y", "target"))
    if date_index is None or value_index is None:
        raise ValueError(f"{source_label} 必须包含 date/time/ds 时间列和 value/y/target 数值列。")

    points_by_date: Dict[date, List[float]] = defaultdict(list)
    for index, row in enumerate(non_empty_rows[1:], start=2):
        if not any((cell or "").strip() for cell in row):
            continue
        raw_date = cell_at(row, date_index)
        raw_value = cell_at(row, value_index)
        if not raw_date or not raw_value:
            raise ValueError(f"第 {index} 行缺少日期或数值。")
        try:
            parsed_date = parse_date(raw_date)
        except ValueError as exc:
            raise ValueError(f"第 {index} 行的日期无法解析：{raw_date}") from exc
        try:
            parsed_value = float(raw_value.replace(",", ""))
        except ValueError as exc:
            raise ValueError(f"第 {index} 行的数值无法解析：{raw_value}") from exc
        if math.isnan(parsed_value) or math.isinf(parsed_value):
            raise ValueError(f"第 {index} 行的数值不是有效有限数字。")
        points_by_date[parsed_date].append(parsed_value)

    if len(points_by_date) < 2:
        raise ValueError("至少需要 2 个不同日期的有效时间序列数据。")

    points: List[Tuple[date, float]] = sorted(
        (item_date, mean(item_values)) for item_date, item_values in points_by_date.items()
    )
    return [SeriesPoint(date=item_date.isoformat(), value=item_value) for item_date, item_value in points]


def cell_at(row: List[str], index: int) -> str:
    if index >= len(row):
        return ""
    return (row[index] or "").strip()


def first_present_index(indexes: Dict[str, int], names: Tuple[str, ...]) -> Optional[int]:
    for name in names:
        if name in indexes:
            return indexes[name]
    return None


def read_shared_strings(workbook: zipfile.ZipFile) -> List[str]:
    try:
        payload = workbook.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(payload)
    strings = []
    for item in root.findall("main:si", XML_NS):
        strings.append("".join(text_node.text or "" for text_node in item.findall(".//main:t", XML_NS)))
    return strings


def first_sheet_path(workbook: zipfile.ZipFile) -> str:
    try:
        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationship_id = workbook_root.find("main:sheets/main:sheet", XML_NS).attrib.get(
            f"{{{XML_NS['rel']}}}id"
        )
        relationships = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    except (AttributeError, KeyError):
        return "xl/worksheets/sheet1.xml"

    for relationship in relationships.findall("pkg:Relationship", XML_NS):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib.get("Target", "worksheets/sheet1.xml")
            if target.startswith("/"):
                return target.lstrip("/")
            if target.startswith("xl/"):
                return target
            return posixpath.normpath(posixpath.join("xl", target))
    return "xl/worksheets/sheet1.xml"


def read_sheet_rows(workbook: zipfile.ZipFile, sheet_path: str, shared_strings: List[str]) -> List[List[str]]:
    root = ET.fromstring(workbook.read(sheet_path))
    rows: List[List[str]] = []
    for sheet_row in root.findall(".//main:sheetData/main:row", XML_NS):
        cells: Dict[int, str] = {}
        for fallback_index, cell in enumerate(sheet_row.findall("main:c", XML_NS), start=1):
            column_index = column_index_from_reference(cell.attrib.get("r")) or fallback_index
            cells[column_index] = read_cell_value(cell, shared_strings)
        if cells:
            rows.append([cells.get(index, "") for index in range(1, max(cells) + 1)])
    return rows


def read_cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text_node.text or "" for text_node in cell.findall(".//main:t", XML_NS)).strip()

    value_node = cell.find("main:v", XML_NS)
    raw_value = "" if value_node is None or value_node.text is None else value_node.text.strip()
    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)].strip()
        except (IndexError, ValueError):
            return raw_value
    if cell_type == "b":
        return "TRUE" if raw_value == "1" else "FALSE"
    return raw_value


def column_index_from_reference(reference: Optional[str]) -> Optional[int]:
    if not reference:
        return None
    column = 0
    for char in reference:
        if not char.isalpha():
            break
        column = column * 26 + ord(char.upper()) - ord("A") + 1
    return column or None


def parse_date(value: str) -> date:
    stripped = value.strip()
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(stripped, date_format).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(stripped).date()
    except ValueError:
        pass
    try:
        return parse_excel_serial_date(float(stripped))
    except ValueError as exc:
        raise ValueError(f"日期无法解析：{value}") from exc


def parse_excel_serial_date(serial: float) -> date:
    if not math.isfinite(serial) or serial < 1 or serial > 100000:
        raise ValueError("Excel 日期序列号不在支持范围内。")
    return date(1899, 12, 30) + timedelta(days=int(serial))


def forecast_series(
    history: List[SeriesPoint],
    model: ForecastModel,
    horizon: int,
    parameters: Dict[str, float],
) -> List[ForecastPoint]:
    if horizon < 1 or horizon > 365:
        raise ValueError("预测步数 horizon 必须在 1 到 365 之间。")

    values = [point.value for point in history]
    future_dates = next_dates(history, horizon)

    if model == ForecastModel.naive:
        predicted = naive(values, horizon)
    elif model == ForecastModel.moving_average:
        window = int(parameters.get("window", 7))
        predicted = moving_average(values, horizon, window)
    elif model == ForecastModel.exponential_smoothing:
        alpha = float(parameters.get("alpha", 0.35))
        predicted = exponential_smoothing(values, horizon, alpha)
    elif model == ForecastModel.linear_trend:
        damping = float(parameters.get("damping", 1.0))
        predicted = linear_trend(values, horizon, damping)
    elif model == ForecastModel.ets:
        alpha = float(parameters.get("alpha", 0.35))
        beta = float(parameters.get("beta", 0.12))
        gamma = float(parameters.get("gamma", 0.1))
        season_length = int(parameters.get("season_length", 1))
        predicted = ets(values, horizon, alpha, beta, gamma, season_length)
    elif model == ForecastModel.elm:
        lookback = int(parameters.get("lookback", 7))
        hidden_units = int(parameters.get("hidden_units", 16))
        regularization = float(parameters.get("regularization", 0.05))
        predicted = elm(values, horizon, lookback, hidden_units, regularization)
    elif model == ForecastModel.lstm:
        lookback = int(parameters.get("lookback", 10))
        hidden_units = int(parameters.get("hidden_units", 12))
        regularization = float(parameters.get("regularization", 0.08))
        predicted = lstm_forecast(values, horizon, lookback, hidden_units, regularization)
    else:
        raise ValueError("不支持的模型。")

    if model != ForecastModel.naive:
        volatility = float(parameters.get("volatility", default_volatility(model)))
        predicted = apply_volatility_pattern(values, predicted, volatility)

    return [
        ForecastPoint(date=item_date.isoformat(), value=round(value, 4))
        for item_date, value in zip(future_dates, predicted)
    ]


def auto_tune_parameters(
    history: List[SeriesPoint],
    model: ForecastModel,
    current_parameters: Dict[str, float],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Dict[str, float], float, int]:
    candidates = parameter_candidates(model, current_parameters, len(history))
    if not candidates:
        return current_parameters, 0.0, 0

    test_size = choose_test_size(len(history))
    if test_size < 1 or len(history) - test_size < 2:
        raise ValueError("自动寻参至少需要 5 个历史数据点。")

    train = history[:-test_size]
    actual = [point.value for point in history[-test_size:]]
    best_parameters = current_parameters
    best_score = float("inf")
    evaluated = 0

    for attempted, candidate in enumerate(candidates, start=1):
        try:
            forecast = forecast_series(train, model, test_size, candidate)
        except ValueError:
            pass
        else:
            predicted = [point.value for point in forecast]
            score = rmse(actual, predicted)
            if math.isfinite(score) and score < best_score:
                best_score = score
                best_parameters = candidate
            evaluated += 1
        if progress_callback:
            progress_callback(attempted, len(candidates))

    if evaluated == 0 or not math.isfinite(best_score):
        raise ValueError("自动寻参没有找到可用的参数组合，请减少窗口长度或关闭自动寻参。")

    return best_parameters, round_metric(best_score), test_size


def choose_test_size(history_count: int) -> int:
    if history_count < 5:
        return 0
    return min(max(2, history_count // 5), max(1, history_count - 3), 30)


def rmse(actual: List[float], predicted: List[float]) -> float:
    return math.sqrt(mean((actual_value - predicted_value) ** 2 for actual_value, predicted_value in zip(actual, predicted)))


def parameter_candidates(
    model: ForecastModel,
    current_parameters: Dict[str, float],
    history_count: int,
) -> List[Dict[str, float]]:
    if model == ForecastModel.naive:
        return [current_parameters]
    if model == ForecastModel.moving_average:
        windows = bounded_unique_ints([3, 5, 7, 14, 21, int(current_parameters.get("window", 7))], 1, history_count - 2)
        volatilities = volatility_candidates(current_parameters, 0.45)
        return [{"window": window, "volatility": volatility} for window in windows for volatility in volatilities]
    if model == ForecastModel.exponential_smoothing:
        volatilities = volatility_candidates(current_parameters, 0.4)
        return [
            {"alpha": alpha, "volatility": volatility}
            for alpha in unique_floats([0.15, 0.25, 0.35, 0.5, 0.7, current_parameters.get("alpha", 0.35)])
            for volatility in volatilities
        ]
    if model == ForecastModel.linear_trend:
        volatilities = volatility_candidates(current_parameters, 0.35)
        return [
            {"damping": damping, "volatility": volatility}
            for damping in unique_floats([0.0, 0.5, 0.75, 1.0, 1.2, current_parameters.get("damping", 1.0)])
            for volatility in volatilities
        ]
    if model == ForecastModel.ets:
        alphas = unique_floats([0.2, 0.35, 0.55, current_parameters.get("alpha", 0.35)])
        betas = unique_floats([0.05, 0.12, 0.25, current_parameters.get("beta", 0.12)])
        gammas = unique_floats([0.0, 0.1, 0.25, current_parameters.get("gamma", 0.1)])
        season_lengths = bounded_unique_ints(
            [1, 7, 12, int(current_parameters.get("season_length", 1))],
            1,
            max(1, history_count // 2),
        )
        volatilities = volatility_candidates(current_parameters, 0.3)
        return [
            {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "season_length": season_length,
                "volatility": volatility,
            }
            for alpha in alphas
            for beta in betas
            for gamma in gammas
            for season_length in season_lengths
            for volatility in volatilities
        ]
    if model == ForecastModel.elm:
        lookbacks = bounded_unique_ints(
            [3, 5, 7, 14, int(current_parameters.get("lookback", 7))],
            2,
            history_count - 2,
        )
        hidden_units = bounded_unique_ints([8, 16, 24, int(current_parameters.get("hidden_units", 16))], 4, 64)
        regularizations = unique_floats([0.01, 0.05, 0.2, current_parameters.get("regularization", 0.05)])
        volatilities = volatility_candidates(current_parameters, 0.3)
        return [
            {
                "lookback": lookback,
                "hidden_units": hidden_unit,
                "regularization": regularization,
                "volatility": volatility,
            }
            for lookback in lookbacks
            for hidden_unit in hidden_units
            for regularization in regularizations
            for volatility in volatilities
        ]
    if model == ForecastModel.lstm:
        lookbacks = bounded_unique_ints(
            [5, 10, 20, int(current_parameters.get("lookback", 10))],
            3,
            history_count - 2,
        )
        hidden_units = bounded_unique_ints([8, 12, 20, int(current_parameters.get("hidden_units", 12))], 4, 48)
        regularizations = unique_floats([0.02, 0.08, 0.25, current_parameters.get("regularization", 0.08)])
        volatilities = volatility_candidates(current_parameters, 0.25)
        return [
            {
                "lookback": lookback,
                "hidden_units": hidden_unit,
                "regularization": regularization,
                "volatility": volatility,
            }
            for lookback in lookbacks
            for hidden_unit in hidden_units
            for regularization in regularizations
            for volatility in volatilities
        ]
    return [current_parameters]


def bounded_unique_ints(values: List[int], minimum: int, maximum: int) -> List[int]:
    return sorted({value for value in values if minimum <= value <= maximum})


def unique_floats(values: List[float]) -> List[float]:
    return sorted({round(float(value), 6) for value in values})


def volatility_candidates(current_parameters: Dict[str, float], fallback: float) -> List[float]:
    return unique_floats([0.0, fallback, 0.7, 1.0, current_parameters.get("volatility", fallback)])


def default_volatility(model: ForecastModel) -> float:
    defaults = {
        ForecastModel.moving_average: 0.45,
        ForecastModel.exponential_smoothing: 0.4,
        ForecastModel.linear_trend: 0.35,
        ForecastModel.ets: 0.3,
        ForecastModel.elm: 0.3,
        ForecastModel.lstm: 0.25,
    }
    return defaults.get(model, 0.0)


def next_dates(history: List[SeriesPoint], horizon: int) -> List[date]:
    dates = [parse_date(point.date) for point in history]
    if len(dates) >= 2:
        intervals = [(right - left).days for left, right in zip(dates, dates[1:]) if (right - left).days > 0]
        step_days = Counter(intervals).most_common(1)[0][0] if intervals else 1
    else:
        step_days = 1
    step = timedelta(days=max(step_days, 1))
    start = dates[-1]
    return [start + step * index for index in range(1, horizon + 1)]


def naive(values: List[float], horizon: int) -> List[float]:
    return [values[-1]] * horizon


def moving_average(values: List[float], horizon: int, window: int) -> List[float]:
    if window < 1:
        raise ValueError("移动平均窗口 window 必须大于等于 1。")
    if window > len(values):
        raise ValueError("移动平均窗口 window 不能大于历史数据条数。")
    forecast_values = values[:]
    predictions = []
    for _ in range(horizon):
        prediction = mean(forecast_values[-window:])
        predictions.append(prediction)
        forecast_values.append(prediction)
    return predictions


def exponential_smoothing(values: List[float], horizon: int, alpha: float) -> List[float]:
    if alpha <= 0 or alpha > 1:
        raise ValueError("平滑系数 alpha 必须在 0 到 1 之间，且不能为 0。")
    level = values[0]
    for value in values[1:]:
        level = alpha * value + (1 - alpha) * level
    return [level] * horizon


def linear_trend(values: List[float], horizon: int, damping: float) -> List[float]:
    if damping < 0 or damping > 1.5:
        raise ValueError("趋势阻尼 damping 建议在 0 到 1.5 之间。")
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = mean(values)
    numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
    denominator = sum((index - x_mean) ** 2 for index in range(n))
    slope = numerator / denominator if denominator else 0
    return [values[-1] + slope * damping * (step + 1) for step in range(horizon)]


def apply_volatility_pattern(values: List[float], predictions: List[float], volatility: float) -> List[float]:
    if volatility <= 0 or len(values) < 4 or not predictions:
        return predictions
    if volatility > 1.5:
        raise ValueError("波动保留 volatility 必须在 0 到 1.5 之间。")

    window = min(7, max(2, len(values) // 4))
    residuals = []
    for index, value in enumerate(values):
        left = max(0, index - window)
        right = min(len(values), index + window + 1)
        local_level = mean(values[left:right])
        residuals.append(value - local_level)

    pattern_length = min(max(3, window * 2), len(residuals))
    pattern = residuals[-pattern_length:]
    centered_pattern = center_pattern(pattern)
    recent_scale = mean(abs(item) for item in residuals[-pattern_length:]) or 0.0
    pattern_scale = mean(abs(item) for item in centered_pattern) or 1.0
    scale = min(recent_scale / pattern_scale, recent_scale * 2 if recent_scale else 0.0)

    if scale <= 0:
        return predictions

    adjusted = []
    for index, prediction in enumerate(predictions):
        residual = centered_pattern[index % len(centered_pattern)] * scale * volatility
        adjusted.append(prediction + residual)
    return adjusted


def center_pattern(values: List[float]) -> List[float]:
    pattern_mean = mean(values)
    centered = [value - pattern_mean for value in values]
    if any(abs(value) > 1e-9 for value in centered):
        return centered
    return values


def ets(values: List[float], horizon: int, alpha: float, beta: float, gamma: float, season_length: int) -> List[float]:
    if alpha <= 0 or alpha > 1:
        raise ValueError("ETS 水平平滑 alpha 必须在 0 到 1 之间，且不能为 0。")
    if beta < 0 or beta > 1:
        raise ValueError("ETS 趋势平滑 beta 必须在 0 到 1 之间。")
    if gamma < 0 or gamma > 1:
        raise ValueError("ETS 季节平滑 gamma 必须在 0 到 1 之间。")
    if season_length < 1:
        raise ValueError("ETS 季节长度 season_length 必须大于等于 1。")

    trend = (values[-1] - values[0]) / max(len(values) - 1, 1)
    if season_length == 1 or len(values) < season_length * 2:
        level = values[0]
        for value in values[1:]:
            previous_level = level
            level = alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - previous_level) + (1 - beta) * trend
        return [level + trend * step for step in range(1, horizon + 1)]

    seasonals = initial_seasonals(values, season_length)
    level = values[0] - seasonals[0]
    for index, value in enumerate(values):
        seasonal = seasonals[index % season_length]
        previous_level = level
        level = alpha * (value - seasonal) + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend
        seasonals[index % season_length] = gamma * (value - level) + (1 - gamma) * seasonal

    return [
        level + trend * step + seasonals[(len(values) + step - 1) % season_length]
        for step in range(1, horizon + 1)
    ]


def initial_seasonals(values: List[float], season_length: int) -> List[float]:
    seasons = len(values) // season_length
    if seasons < 2:
        return [0.0] * season_length
    season_averages = [
        mean(values[index * season_length : (index + 1) * season_length])
        for index in range(seasons)
    ]
    return [
        mean(values[season * season_length + offset] - season_averages[season] for season in range(seasons))
        for offset in range(season_length)
    ]


def elm(values: List[float], horizon: int, lookback: int, hidden_units: int, regularization: float) -> List[float]:
    validate_window_model(values, lookback, hidden_units, regularization, "ELM")
    normalized, center, scale = normalize_values(values)
    samples = make_lag_samples(normalized, lookback)
    weights = random_matrix(hidden_units, lookback, 0.85)
    biases = random_vector(hidden_units, 0.35)

    weights_arr = np.array(weights, dtype=float)
    biases_arr = np.array(biases, dtype=float)
    inputs = np.array([s[0] for s in samples], dtype=float)
    targets = np.array([s[1] for s in samples], dtype=float)

    hidden = sigmoid_vectorized(inputs @ weights_arr.T + biases_arr)
    output_weights = ridge_solve(hidden.tolist(), targets.tolist(), regularization)

    window = np.array(normalized[-lookback:])
    predictions = []
    for _ in range(horizon):
        features = sigmoid_vectorized(window @ weights_arr.T + biases_arr)
        prediction = dot(output_weights, features.tolist())
        prediction = clamp_float(prediction, -8, 8)
        predictions.append(prediction * scale + center)
        window = np.append(window[1:], prediction)
    return predictions


def lstm_forecast(
    values: List[float],
    horizon: int,
    lookback: int,
    hidden_units: int,
    regularization: float,
) -> List[float]:
    validate_window_model(values, lookback, hidden_units, regularization, "LSTM")
    normalized, center, scale = normalize_values(values)
    samples = make_lag_samples(normalized, lookback)
    cell = LstmCell(hidden_units)

    inputs_arr = np.array([s[0] for s in samples], dtype=float)
    targets_arr = np.array([s[1] for s in samples], dtype=float)

    cell_np = np.array(cell.hidden_weights, dtype=float)
    input_np = np.array(cell.input_weights, dtype=float)
    bias_np = np.array(cell.biases, dtype=float)

    states = np.array([cell.encode_vectorized(seq, cell_np, input_np, bias_np) for seq in inputs_arr])
    output_weights = ridge_solve(states.tolist(), targets_arr.tolist(), regularization)

    window = np.array(normalized[-lookback:])
    predictions = []
    for _ in range(horizon):
        state = cell.encode_vectorized(window, cell_np, input_np, bias_np)
        prediction = clamp_float(dot(output_weights, state.tolist()), -8, 8)
        predictions.append(prediction * scale + center)
        window = np.append(window[1:], prediction)
    return predictions


class LstmCell:
    def __init__(self, hidden_units: int):
        self.hidden_units = hidden_units
        self.input_weights = random_matrix(hidden_units * 4, 1, 0.9)
        self.hidden_weights = random_matrix(hidden_units * 4, hidden_units, 0.35)
        self.biases = random_vector(hidden_units * 4, 0.15)

    def encode(self, sequence: List[float]) -> List[float]:
        hidden = [0.0] * self.hidden_units
        cell = [0.0] * self.hidden_units
        for value in sequence:
            next_hidden = []
            next_cell = []
            for index in range(self.hidden_units):
                base = index * 4
                hidden_mix = dot(self.hidden_weights[base], hidden)
                forget_gate = sigmoid(self.input_weights[base][0] * value + hidden_mix + self.biases[base] + 0.6)
                input_gate = sigmoid(
                    self.input_weights[base + 1][0] * value
                    + dot(self.hidden_weights[base + 1], hidden)
                    + self.biases[base + 1]
                )
                output_gate = sigmoid(
                    self.input_weights[base + 2][0] * value
                    + dot(self.hidden_weights[base + 2], hidden)
                    + self.biases[base + 2]
                )
                candidate = math.tanh(
                    self.input_weights[base + 3][0] * value
                    + dot(self.hidden_weights[base + 3], hidden)
                    + self.biases[base + 3]
                )
                cell_value = forget_gate * cell[index] + input_gate * candidate
                next_cell.append(cell_value)
                next_hidden.append(output_gate * math.tanh(cell_value))
            hidden = next_hidden
            cell = next_cell
        return hidden

    def encode_vectorized(self, sequence, hidden_weights, input_weights, biases):
        hidden = np.zeros(self.hidden_units)
        cell_state = np.zeros(self.hidden_units)
        for value in sequence:
            hidden_mix = hidden_weights[:, :self.hidden_units] @ hidden
            gates = input_weights[:, 0] * value + hidden_mix + biases
            gates = gates.reshape(4, self.hidden_units)
            forget_gate = sigmoid_vectorized(gates[0])
            input_gate = sigmoid_vectorized(gates[1])
            output_gate = sigmoid_vectorized(gates[2])
            candidate = np.tanh(gates[3])
            cell_state = forget_gate * cell_state + input_gate * candidate
            hidden = output_gate * np.tanh(cell_state)
        return hidden


def validate_window_model(
    values: List[float],
    lookback: int,
    hidden_units: int,
    regularization: float,
    label: str,
) -> None:
    if lookback < 2:
        raise ValueError(f"{label} 回看窗口 lookback 必须大于等于 2。")
    if lookback >= len(values):
        raise ValueError(f"{label} 回看窗口 lookback 必须小于历史数据条数。")
    if hidden_units < 1 or hidden_units > 128:
        raise ValueError(f"{label} 隐藏单元 hidden_units 必须在 1 到 128 之间。")
    if regularization <= 0:
        raise ValueError(f"{label} 正则强度 regularization 必须大于 0。")


def normalize_values(values: List[float]) -> Tuple[List[float], float, float]:
    center = mean(values)
    variance = mean((value - center) ** 2 for value in values)
    scale = math.sqrt(variance) or 1.0
    return [(value - center) / scale for value in values], center, scale


def make_lag_samples(values: List[float], lookback: int) -> List[Tuple[List[float], float]]:
    return [(values[index - lookback : index], values[index]) for index in range(lookback, len(values))]


def ridge_solve(features: List[List[float]], targets: List[float], regularization: float) -> List[float]:
    if not features:
        raise ValueError("训练样本为空。")
    X = np.array(features, dtype=float)
    y = np.array(targets, dtype=float)
    columns = X.shape[1]
    if columns == 0:
        raise ValueError("特征维度为0。")
    regularization_matrix = regularization * np.eye(columns)
    try:
        return np.linalg.solve(X.T @ X + regularization_matrix, X.T @ y).tolist()
    except np.linalg.LinAlgError:
        return solve_linear_system_pure_python(features, targets, regularization)


def solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[pivot][column] = 1e-12
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * base for current, base in zip(augmented[row], augmented[column])
            ]
    return [row[-1] for row in augmented]


def solve_linear_system_pure_python(matrix: List[List[float]], vector: List[float]) -> List[float]:
    return solve_linear_system(matrix, vector)


def random_matrix(rows: int, columns: int, scale: float) -> List[List[float]]:
    return [[deterministic_weight(row, column, scale) for column in range(columns)] for row in range(rows)]


def random_vector(length: int, scale: float) -> List[float]:
    return [deterministic_weight(index, 17, scale) for index in range(length)]


def deterministic_weight(row: int, column: int, scale: float) -> float:
    raw = math.sin((row + 1) * 12.9898 + (column + 1) * 78.233) * 43758.5453
    return ((raw - math.floor(raw)) * 2 - 1) * scale


def dot(left: List[float], right: List[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def sigmoid(value: float) -> float:
    value = clamp_float(value, -50, 50)
    return 1 / (1 + math.exp(-value))


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def summarize_forecast(history: List[SeriesPoint], forecast: List[ForecastPoint]) -> ForecastSummary:
    if not history or not forecast:
        raise ValueError("无法生成结果摘要：历史数据或预测数据为空。")

    forecast_values = [point.value for point in forecast]
    last_actual = history[-1].value
    final_forecast = forecast_values[-1]
    absolute_change = final_forecast - last_actual
    percentage_change = None if last_actual == 0 else absolute_change / abs(last_actual) * 100

    if absolute_change > 0:
        trend = "上升"
    elif absolute_change < 0:
        trend = "下降"
    else:
        trend = "持平"

    return ForecastSummary(
        history_count=len(history),
        forecast_count=len(forecast),
        last_actual=round_metric(last_actual),
        first_forecast=round_metric(forecast_values[0]),
        final_forecast=round_metric(final_forecast),
        forecast_average=round_metric(mean(forecast_values)),
        forecast_min=round_metric(min(forecast_values)),
        forecast_max=round_metric(max(forecast_values)),
        absolute_change=round_metric(absolute_change),
        percentage_change=None if percentage_change is None else round_metric(percentage_change),
        trend=trend,
    )


def round_metric(value: float) -> float:
    return round(value, 4)

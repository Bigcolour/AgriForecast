import math
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import DecompositionModel


def require_numpy():
    return np


@dataclass
class DecomposedSeries:
    names: List[str]
    components: List[List[float]]


def decompose_values(
    values: List[float],
    model: DecompositionModel,
    parameters: Dict[str, float],
) -> DecomposedSeries:
    if model == DecompositionModel.none:
        return DecomposedSeries(names=["原始序列"], components=[values[:]])
    if len(values) < 6:
        raise ValueError("数据分解至少需要 6 个历史数据点。")

    if model == DecompositionModel.vmd:
        return vmd_decompose(values, parameters)
    if model == DecompositionModel.ssa:
        return ssa_decompose(values, parameters)
    if model == DecompositionModel.ewt:
        return ewt_decompose(values, parameters)
    raise ValueError("不支持的分解模型。")


def decomposition_candidates(
    model: DecompositionModel,
    current_parameters: Dict[str, float],
    history_count: int,
) -> List[Dict[str, float]]:
    if model == DecompositionModel.none:
        return [current_parameters]
    if model == DecompositionModel.vmd:
        max_modes = min(8, max(2, history_count // 4))
        modes = bounded_unique_ints([3, 5, 6, int(current_parameters.get("modes", 5))], 2, max_modes)
        alphas = unique_floats([1000, 2000, 3000, current_parameters.get("alpha", 3000)])
        return [{"modes": mode, "alpha": alpha} for mode in modes for alpha in alphas]
    if model == DecompositionModel.ssa:
        max_window = min(30, max(3, history_count // 2))
        windows = bounded_unique_ints([6, 8, 12, int(current_parameters.get("window", 6))], 3, max_window)
        candidates = []
        for window in windows:
            components = bounded_unique_ints(
                [3, 5, 6, int(current_parameters.get("components", min(6, window)))],
                2,
                window,
            )
            candidates.extend({"window": window, "components": component_count} for component_count in components)
        return candidates
    if model == DecompositionModel.ewt:
        max_bands = min(8, max(2, history_count // 4))
        bands = bounded_unique_ints([3, 5, 6, int(current_parameters.get("bands", 5))], 2, max_bands)
        return [{"bands": band} for band in bands]
    return [current_parameters]


def default_decomposition_parameters(model: DecompositionModel) -> Dict[str, float]:
    defaults = {
        DecompositionModel.none: {},
        DecompositionModel.vmd: {"modes": 5, "alpha": 3000},
        DecompositionModel.ssa: {"window": 6, "components": 6},
        DecompositionModel.ewt: {"bands": 5},
    }
    return defaults[model]


def bounded_unique_ints(values: List[int], minimum: int, maximum: int) -> List[int]:
    return sorted({value for value in values if minimum <= value <= maximum})


def unique_floats(values: List[float]) -> List[float]:
    return sorted({round(float(value), 6) for value in values})


def require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError("数据分解需要安装 numpy，请先执行 pip install -r backend/requirements.txt。") from exc
    return np


def vmd_decompose(values: List[float], parameters: Dict[str, float]) -> DecomposedSeries:
    np = require_numpy()
    original = np.asarray(values, dtype=float)
    modes = int(parameters.get("modes", 5))
    alpha = float(parameters.get("alpha", 3000))
    if modes < 2 or modes > min(12, max(2, len(values) // 2)):
        raise ValueError("VMD 模态数 modes 必须合理小于历史数据长度。")
    if alpha <= 0:
        raise ValueError("VMD alpha 必须大于 0。")

    working = original
    if working.size % 2:
        working = np.append(working, working[-1])
    decomposed, _, _ = vmd(working, alpha=alpha, tau=0, mode_count=modes, dc=0, init=1, tolerance=1e-6)
    components = [component[: original.size].astype(float).tolist() for component in decomposed]
    names = [f"VMD-IMF{i + 1}" for i in range(len(components))]
    return with_residual(original, names, components, "VMD-残差")


def vmd(f, alpha, tau, mode_count, dc, init, tolerance):
    np = require_numpy()
    fs = 1.0 / len(f)
    half_len = len(f) // 2
    mirrored = np.append(np.flip(f[:half_len], axis=0), f)
    mirrored = np.append(mirrored, np.flip(f[-half_len:], axis=0))
    signal_len = len(mirrored)
    time = np.arange(1, signal_len + 1) / signal_len
    freqs = time - 0.5 - (1 / signal_len)
    iteration_limit = 300
    alpha_vector = alpha * np.ones(mode_count)

    f_hat = np.fft.fftshift(np.fft.fft(mirrored))
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[: signal_len // 2] = 0

    omega_plus = np.zeros([iteration_limit, mode_count])
    if init == 1:
        for index in range(mode_count):
            omega_plus[0, index] = (0.5 / mode_count) * index
    elif init == 2:
        omega_plus[0, :] = np.sort(
            np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.default_rng(0).random(mode_count))
        )
    if dc:
        omega_plus[0, 0] = 0

    lambda_hat = np.zeros([iteration_limit, len(freqs)], dtype=complex)
    u_hat_plus = np.zeros([iteration_limit, len(freqs), mode_count], dtype=complex)
    u_diff = tolerance + np.spacing(1)
    iteration = 0
    sum_uk = 0

    while u_diff > tolerance and iteration < iteration_limit - 1:
        mode_index = 0
        sum_uk = u_hat_plus[iteration, :, mode_count - 1] + sum_uk - u_hat_plus[iteration, :, 0]
        u_hat_plus[iteration + 1, :, mode_index] = (
            f_hat_plus - sum_uk - lambda_hat[iteration, :] / 2
        ) / (1.0 + alpha_vector[mode_index] * (freqs - omega_plus[iteration, mode_index]) ** 2)

        if not dc:
            omega_plus[iteration + 1, mode_index] = update_omega(
                np, freqs, signal_len, u_hat_plus[iteration + 1, :, mode_index], omega_plus[iteration, mode_index]
            )

        for mode_index in range(1, mode_count):
            sum_uk = u_hat_plus[iteration + 1, :, mode_index - 1] + sum_uk - u_hat_plus[iteration, :, mode_index]
            u_hat_plus[iteration + 1, :, mode_index] = (
                f_hat_plus - sum_uk - lambda_hat[iteration, :] / 2
            ) / (1 + alpha_vector[mode_index] * (freqs - omega_plus[iteration, mode_index]) ** 2)
            omega_plus[iteration + 1, mode_index] = update_omega(
                np, freqs, signal_len, u_hat_plus[iteration + 1, :, mode_index], omega_plus[iteration, mode_index]
            )

        lambda_hat[iteration + 1, :] = lambda_hat[iteration, :] + tau * (
            np.sum(u_hat_plus[iteration + 1, :, :], axis=1) - f_hat_plus
        )

        iteration += 1
        u_diff = np.spacing(1)
        for mode_index in range(mode_count):
            delta = u_hat_plus[iteration, :, mode_index] - u_hat_plus[iteration - 1, :, mode_index]
            u_diff += (1 / signal_len) * np.dot(delta, np.conj(delta))
        u_diff = np.abs(u_diff)

    used_iteration = min(iteration, iteration_limit - 1)
    omega = omega_plus[:used_iteration, :]
    idxs = np.flip(np.arange(1, signal_len // 2 + 1), axis=0)
    u_hat = np.zeros([signal_len, mode_count], dtype=complex)
    u_hat[signal_len // 2 : signal_len, :] = u_hat_plus[used_iteration, signal_len // 2 : signal_len, :]
    u_hat[idxs, :] = np.conj(u_hat_plus[used_iteration, signal_len // 2 : signal_len, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])

    u = np.zeros([mode_count, len(time)])
    for mode_index in range(mode_count):
        u[mode_index, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, mode_index])))
    u = u[:, signal_len // 4 : 3 * signal_len // 4]

    recomputed_hat = np.zeros([u.shape[1], mode_count], dtype=complex)
    for mode_index in range(mode_count):
        recomputed_hat[:, mode_index] = np.fft.fftshift(np.fft.fft(u[mode_index, :]))
    return u, recomputed_hat, omega


def update_omega(np, freqs, signal_len, spectrum, fallback):
    weights = np.abs(spectrum[signal_len // 2 : signal_len]) ** 2
    denominator = np.sum(weights)
    if denominator <= 1e-12 or not np.isfinite(denominator):
        return fallback
    return np.dot(freqs[signal_len // 2 : signal_len], weights) / denominator


def ssa_decompose(values: List[float], parameters: Dict[str, float]) -> DecomposedSeries:
    np = require_numpy()
    series = np.asarray(values, dtype=float)
    window = int(parameters.get("window", 6))
    component_count = int(parameters.get("components", min(6, window)))
    if window < 3 or window >= len(series):
        raise ValueError("SSA 窗口长度 window 必须大于等于 3，且小于历史数据长度。")

    trajectory_columns = len(series) - window + 1
    trajectory = np.column_stack([series[index : index + window] for index in range(trajectory_columns)])
    left, singular_values, right = np.linalg.svd(trajectory, full_matrices=False)
    rank = min(component_count, len(singular_values))
    components = []
    for index in range(rank):
        matrix = singular_values[index] * np.outer(left[:, index], right[index, :])
        components.append(diagonal_average(np, matrix, len(series)).tolist())
    names = [f"SSA-分量{i + 1}" for i in range(len(components))]
    return with_residual(series, names, components, "SSA-残差")


def diagonal_average(np, matrix, output_length):
    rows, columns = matrix.shape
    reconstructed = np.zeros(output_length)
    counts = np.zeros(output_length)
    for row in range(rows):
        for column in range(columns):
            reconstructed[row + column] += matrix[row, column]
            counts[row + column] += 1
    return reconstructed / np.maximum(counts, 1)


def ewt_decompose(values: List[float], parameters: Dict[str, float]) -> DecomposedSeries:
    np = require_numpy()
    series = np.asarray(values, dtype=float)
    bands = int(parameters.get("bands", 5))
    if bands < 2 or bands > min(12, max(2, len(series) // 2)):
        raise ValueError("EWT 频带数 bands 必须合理小于历史数据长度。")

    modes = ewt_1d(np, series, bands)
    components = [modes[:, index].astype(float).tolist() for index in range(modes.shape[1])]
    names = [f"EWT-频带{i + 1}" for i in range(len(components))]
    return with_residual(series, names, components, "EWT-残差")


def ewt_1d(np, series, bands):
    spectrum = np.abs(np.fft.fft(series)[: int(np.ceil(series.size / 2))])
    boundaries = local_max_boundaries(np, smooth_spectrum(np, spectrum), bands)
    boundaries = boundaries * np.pi / round(spectrum.size)
    if len(boundaries) < bands - 1:
        boundaries = complete_boundaries(np, boundaries, bands - 1)

    mirror_len = int(np.ceil(series.size / 2))
    mirrored = np.append(np.flip(series[0 : mirror_len - 1], axis=0), series)
    mirrored = np.append(mirrored, np.flip(series[-mirror_len - 1 : -1], axis=0))
    mirrored_fft = np.fft.fft(mirrored)
    filters = meyer_filter_bank(np, boundaries, mirrored_fft.size)

    modes = np.zeros(filters.shape)
    for index in range(filters.shape[1]):
        modes[:, index] = np.real(np.fft.ifft(np.conjugate(filters[:, index]) * mirrored_fft))
    return modes[mirror_len - 1 : -mirror_len, :]


def smooth_spectrum(np, spectrum):
    if spectrum.size < 7:
        return spectrum
    window = min(9, spectrum.size if spectrum.size % 2 else spectrum.size - 1)
    kernel = np.ones(window) / window
    return np.convolve(spectrum, kernel, mode="same")


def local_max_boundaries(np, spectrum, bands):
    target_maxima = bands - 1
    local_maxima = np.zeros(spectrum.size)
    for index in range(1, spectrum.size - 1):
        if spectrum[index - 1] < spectrum[index] and spectrum[index] > spectrum[index + 1]:
            local_maxima[index] = spectrum[index]
    selected = np.sort(local_maxima.argsort()[::-1][:target_maxima])
    boundaries = np.zeros(len(selected))
    for index, maximum_index in enumerate(selected):
        left = 0 if index == 0 else selected[index - 1]
        boundaries[index] = (left + maximum_index) / 2
    return boundaries + 1


def complete_boundaries(np, boundaries, target_count):
    if len(boundaries) == 0:
        return np.linspace(np.pi / (target_count + 1), np.pi - np.pi / (target_count + 1), target_count)
    completed = boundaries.copy()
    missing = target_count - len(completed)
    step = (np.pi - completed[-1]) / (missing + 1)
    for _ in range(missing):
        completed = np.append(completed, completed[-1] + step)
    return completed


def meyer_filter_bank(np, boundaries, signal_len):
    boundary_count = len(boundaries)
    gamma = 1
    for index in range(boundary_count - 1):
        ratio = (boundaries[index + 1] - boundaries[index]) / (boundaries[index + 1] + boundaries[index])
        gamma = min(gamma, ratio)
    gamma = min(gamma, (np.pi - boundaries[-1]) / (np.pi + boundaries[-1]))
    gamma = max((1 - 1 / signal_len) * gamma, 1e-6)

    filters = np.zeros([signal_len, boundary_count + 1])
    middle = int(np.floor(signal_len / 2))
    frequencies = np.fft.fftshift(np.linspace(0, 2 * np.pi - 2 * np.pi / signal_len, num=signal_len))
    frequencies[0:middle] = -2 * np.pi + frequencies[0:middle]
    absolute_frequencies = np.abs(frequencies)

    filters[:, 0] = meyer_scaling(np, absolute_frequencies, boundaries[0], gamma)
    for index in range(boundary_count - 1):
        filters[:, index + 1] = meyer_wavelet(np, absolute_frequencies, boundaries[index], boundaries[index + 1], gamma)
    filters[:, boundary_count] = meyer_wavelet(np, absolute_frequencies, boundaries[-1], np.pi, gamma)
    return np.fft.ifftshift(filters, axes=0)


def meyer_scaling(np, absolute_frequencies, boundary, gamma):
    values = np.zeros(absolute_frequencies.size)
    lower = (1 - gamma) * boundary
    upper = (1 + gamma) * boundary
    values[absolute_frequencies <= lower] = 1
    transition = (absolute_frequencies >= lower) & (absolute_frequencies <= upper)
    values[transition] = np.cos(np.pi * ewt_beta((absolute_frequencies[transition] - lower) / (2 * gamma * boundary)) / 2)
    return values


def meyer_wavelet(np, absolute_frequencies, lower_boundary, upper_boundary, gamma):
    values = np.zeros(absolute_frequencies.size)
    lower_pass = (1 + gamma) * lower_boundary
    upper_pass = (1 - gamma) * upper_boundary
    left_transition = ((1 - gamma) * lower_boundary, (1 + gamma) * lower_boundary)
    right_transition = ((1 - gamma) * upper_boundary, (1 + gamma) * upper_boundary)

    values[(absolute_frequencies >= lower_pass) & (absolute_frequencies <= upper_pass)] = 1
    left = (absolute_frequencies >= left_transition[0]) & (absolute_frequencies <= left_transition[1])
    right = (absolute_frequencies >= right_transition[0]) & (absolute_frequencies <= right_transition[1])
    values[left] = np.sin(
        np.pi * ewt_beta((absolute_frequencies[left] - left_transition[0]) / (2 * gamma * lower_boundary)) / 2
    )
    values[right] = np.cos(
        np.pi * ewt_beta((absolute_frequencies[right] - right_transition[0]) / (2 * gamma * upper_boundary)) / 2
    )
    return values


def ewt_beta(value):
    return (value**4) * (35 - 84 * value + 70 * (value**2) - 20 * (value**3))


def with_residual(series, names: List[str], components: List[List[float]], residual_name: str) -> DecomposedSeries:
    np = require_numpy()
    component_matrix = np.asarray(components, dtype=float)
    residual = series - np.sum(component_matrix, axis=0)
    residual_scale = math.sqrt(mean(float(item) ** 2 for item in residual))
    series_scale = math.sqrt(mean(float(item) ** 2 for item in series)) or 1.0
    if residual_scale / series_scale > 1e-6:
        names = [*names, residual_name]
        components = [*components, residual.astype(float).tolist()]
    return DecomposedSeries(names=names, components=components)


def pso_cs_optimize_weights(
    component_predictions: List[List[float]],
    target_values: List[float],
    n_iter: int = 120,
    sizepop: int = 40,
    baseline_weights: Optional[List[float]] = None,
) -> List[float]:
    n_components = len(component_predictions)
    if n_components == 0:
        return []
    if n_components == 1:
        return [1.0]

    min_len = min(len(prediction) for prediction in component_predictions)
    min_len = min(min_len, len(target_values))
    if min_len < 1:
        return [1.0] * n_components

    train_true = np.asarray(target_values[:min_len], dtype=float)
    train_preds = np.asarray([prediction[:min_len] for prediction in component_predictions], dtype=float)
    if train_preds.shape != (n_components, min_len):
        return [1.0] * n_components

    baseline = np.asarray(baseline_weights if baseline_weights else [1.0] * n_components, dtype=float)
    if baseline.shape != (n_components,):
        baseline = np.ones(n_components, dtype=float)

    c1 = 1.4999
    c2 = 1.4999
    wmax = 0.9
    wmin = 0.4
    pa = 0.25
    popmin = 0.0
    popmax = max(2.0, float(np.max(baseline)) * 1.5)
    Vmin = -(popmax - popmin) * 0.25
    Vmax = (popmax - popmin) * 0.25
    rng = np.random.default_rng(20240517 + n_components * 31 + min_len)

    baseline = np.clip(baseline, popmin, popmax)
    pop = rng.uniform(popmin, popmax, size=(sizepop, n_components))
    pop[0] = baseline
    for index in range(1, min(sizepop, n_components + 1)):
        pop[index] = np.clip(baseline + rng.normal(0, 0.2, size=n_components), popmin, popmax)

    V = np.zeros((sizepop, n_components))
    target_scale = max(float(np.var(train_true)), 1.0)

    def prediction_error(weights: np.ndarray) -> float:
        combined = np.dot(weights, train_preds)
        return float(np.mean((train_true - combined) ** 2))

    def fitness(weights: np.ndarray) -> float:
        stability_penalty = 0.002 * target_scale * float(np.mean((weights - baseline) ** 2))
        return prediction_error(weights) + stability_penalty

    fitness_vals = np.array([fitness(pop[i]) for i in range(sizepop)])

    best_idx = np.argmin(fitness_vals)
    zbest = pop[best_idx].copy()
    gbest = pop.copy()
    fitness_gbest = fitness_vals.copy()
    fitness_zbest = fitness_vals[best_idx]

    for iteration in range(n_iter):
        for i in range(sizepop):
            w = wmax - (wmax - wmin) * (iteration / max(n_iter - 1, 1))
            V[i] = w * V[i] + c1 * rng.random() * (gbest[i] - pop[i]) + c2 * rng.random() * (zbest - pop[i])
            V[i] = np.clip(V[i], Vmin, Vmax)
            pop[i] = pop[i] + 0.5 * V[i]
            pop[i] = np.clip(pop[i], popmin, popmax)

            new_fitness = fitness(pop[i])
            fitness_vals[i] = new_fitness
            if new_fitness < fitness_gbest[i]:
                fitness_gbest[i] = new_fitness
                gbest[i] = pop[i].copy()
            if new_fitness < fitness_zbest:
                fitness_zbest = new_fitness
                zbest = pop[i].copy()

        for i in range(sizepop):
            new_nest = levy_flight(zbest, pop[i], popmin, popmax, rng)
            new_fitness = fitness(new_nest)
            if new_fitness <= fitness_vals[i]:
                fitness_vals[i] = new_fitness
                pop[i] = new_nest

        for i in range(sizepop):
            if rng.random() < pa:
                new_nest = empty_nest(pop, popmin, popmax, rng)
                new_fitness = fitness(new_nest)
                if new_fitness <= fitness_vals[i]:
                    fitness_vals[i] = new_fitness
                    pop[i] = new_nest

        for i in range(sizepop):
            if fitness_vals[i] < fitness_zbest:
                fitness_zbest = fitness_vals[i]
                zbest = pop[i].copy()

    final_weights = np.clip(zbest, popmin, popmax)
    if prediction_error(final_weights) > prediction_error(baseline):
        final_weights = baseline
    return final_weights.tolist()


def levy_flight(best, nest, popmin, popmax, rng):
    beta = 1.8
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
        math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    )) ** (1 / beta)
    n = len(nest)
    u = rng.normal(size=n) * sigma
    v = rng.normal(size=n)
    step = u / (abs(v) ** (1 / beta))
    stepsize = 0.01 * step * (nest - best)
    return np.clip(nest + stepsize * rng.normal(size=n), popmin, popmax)


def empty_nest(pop, popmin, popmax, rng):
    n, m = pop.shape
    j = rng.integers(n)
    new_nest = pop[j].copy()
    if rng.random() < 0.25:
        new_nest = new_nest + rng.random() * (pop[rng.integers(n)] - pop[rng.integers(n)])
    return np.clip(new_nest, popmin, popmax)

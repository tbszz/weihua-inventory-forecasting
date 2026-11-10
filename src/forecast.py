from __future__ import annotations

from typing import Iterable

import numpy as np


def _history(history: Iterable[float], window: int) -> np.ndarray:
    values = np.asarray(list(history), dtype=float)
    values = values[np.isfinite(values)]
    values = np.clip(values, 0, None)
    return values[-max(1, int(window)) :]


def _croston(history: np.ndarray, alpha: float = 0.1) -> float:
    nonzero = np.flatnonzero(history > 0)
    if len(nonzero) == 0:
        return 0.0
    demand = float(history[nonzero[0]])
    interval = float(nonzero[0] + 1)
    last = int(nonzero[0])
    for index in nonzero[1:]:
        demand = demand + alpha * (history[index] - demand)
        interval = interval + alpha * (float(index - last) - interval)
        last = int(index)
    return max(0.0, demand / max(interval, 1.0))


def forecast(
    history: Iterable[float],
    horizon: int,
    *,
    model: str = "mean",
    window: int = 28,
    alpha: float = 0.2,
    history_dates: Iterable[object] | None = None,
    future_dates: Iterable[object] | None = None,
    dow_shrink: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Return a deterministic, non-negative daily forecast and metadata."""
    history_values = list(history)
    values = _history(history_values, window)
    horizon = max(1, int(horizon))
    if len(values) == 0:
        level = 0.0
    elif model == "mean":
        level = float(values.mean())
    elif model == "wma":
        weights = np.arange(1, len(values) + 1, dtype=float)
        level = float(np.average(values, weights=weights))
    elif model == "median":
        level = float(np.median(values))
    elif model == "ewma":
        level = float(values[0])
        for item in values[1:]:
            level = alpha * float(item) + (1 - alpha) * level
    elif model == "croston":
        level = _croston(values, alpha=min(max(alpha, 0.01), 0.99))
    elif model == "dow_mean":
        if history_dates is None or future_dates is None:
            raise ValueError("dow_mean requires history_dates and future_dates")
        all_history_dates = np.asarray(list(history_dates))
        if len(all_history_dates) != len(history_values):
            raise ValueError("history_dates must align with history")
        dates = np.asarray(all_history_dates[-len(values) :], dtype="datetime64[D]")
        target_dates = np.asarray(list(future_dates), dtype="datetime64[D]")[:horizon]
        if len(target_dates) < horizon:
            raise ValueError("future_dates must cover horizon")
        weekdays = pd_weekday(dates)
        target_weekdays = pd_weekday(target_dates)
        overall = float(values.mean()) if len(values) else 0.0
        predictions = []
        for weekday in target_weekdays:
            same_day = values[weekdays == weekday]
            if len(same_day) == 0:
                predictions.append(overall)
            else:
                predictions.append((float(same_day.sum()) + dow_shrink * overall) / (len(same_day) + dow_shrink))
        result = np.clip(np.asarray(predictions, dtype=float), 0, None)
        return result, {
            "model": model,
            "window": int(window),
            "alpha": float(alpha),
            "dow_shrink": float(dow_shrink),
            "history_points": int(len(values)),
            "level": float(result.mean()) if len(result) else 0.0,
        }
    else:
        raise ValueError(f"unknown forecast model: {model}")
    level = max(0.0, float(level))
    return np.full(horizon, level, dtype=float), {
        "model": model,
        "window": int(window),
        "alpha": float(alpha),
        "history_points": int(len(values)),
        "level": level,
    }


def pd_weekday(values: np.ndarray) -> np.ndarray:
    """Return Monday=0..Sunday=6 without requiring pandas in the forecast core."""
    return (values.astype("datetime64[D]").astype("int64") + 3) % 7


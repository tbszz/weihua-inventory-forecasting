from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .forecast import forecast


DEFAULT_MODELS = ("mean", "wma", "ewma", "croston", "median", "dow_mean")


def _metrics(actual: np.ndarray, predicted: np.ndarray, train: np.ndarray) -> dict:
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    scale = float(np.abs(np.diff(train)).mean()) if len(train) > 1 else 0.0
    return {
        "wape": float(np.abs(error).sum() / denominator) if denominator else float(np.abs(error).mean()),
        "abs_error_sum": float(np.abs(error).sum()),
        "actual_sum": float(np.abs(actual).sum()),
        "forecast_sum": float(np.abs(predicted).sum()),
        "signed_error_sum": float(error.sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mase": float(np.abs(error).mean() / scale) if scale else np.nan,
    }


def evaluate_models(
    demand: pd.DataFrame,
    *,
    horizons: Iterable[int] = (7, 14),
    min_train: int = 28,
    step: int = 7,
    models: Iterable[str] = DEFAULT_MODELS,
    window: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for (material, unit), group in demand.groupby(["material", "unit"], sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        values = group["demand_qty"].to_numpy(dtype=float)
        dates = pd.to_datetime(group["date"]).to_numpy()
        for horizon in horizons:
            if len(values) <= min_train + horizon:
                continue
            for origin_idx in range(min_train, len(values) - horizon + 1, step):
                train = values[:origin_idx]
                actual = values[origin_idx : origin_idx + horizon]
                origin = pd.Timestamp(dates[origin_idx - 1])
                target_dates = pd.to_datetime(dates[origin_idx : origin_idx + horizon])
                for model in models:
                    predicted, meta = forecast(
                        train,
                        horizon,
                        model=model,
                        window=window,
                        history_dates=dates[:origin_idx],
                        future_dates=dates[origin_idx : origin_idx + horizon],
                    )
                    metric_rows.append({
                        "material": material,
                        "unit": unit,
                        "model": model,
                        "horizon": int(horizon),
                        "origin": origin,
                        **_metrics(actual, predicted, train),
                        "history_points": meta["history_points"],
                        "window": int(window),
                    })
                    for date, actual_value, forecast_value in zip(target_dates, actual, predicted):
                        prediction_rows.append({
                            "material": material,
                            "unit": unit,
                            "model": model,
                            "horizon": int(horizon),
                            "origin": origin,
                            "date": pd.Timestamp(date),
                            "actual_qty": float(actual_value),
                            "forecast_qty": float(forecast_value),
                            "window": int(window),
                        })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def select_models(
    metrics: pd.DataFrame,
    *,
    horizon: int = 7,
    bias_penalty: float = 0.25,
    coverage_floor: float = 0.65,
) -> pd.DataFrame:
    subset = metrics[metrics["horizon"] == horizon].copy()
    if subset.empty:
        return pd.DataFrame(columns=["material", "unit", "model", "horizon", "wape", "mae"])
    group_keys = ["material", "unit", "model", "horizon"]
    if "window" in subset.columns:
        group_keys.append("window")
    aggregations = {"wape": ("wape", "mean"), "mae": ("mae", "mean")}
    if {"abs_error_sum", "actual_sum"}.issubset(subset.columns):
        aggregations["abs_error_sum"] = ("abs_error_sum", "sum")
        aggregations["actual_sum"] = ("actual_sum", "sum")
    if "forecast_sum" in subset:
        aggregations["forecast_sum"] = ("forecast_sum", "sum")
    if "signed_error_sum" in subset:
        aggregations["signed_error_sum"] = ("signed_error_sum", "sum")
    if "rmse" in subset:
        aggregations["rmse"] = ("rmse", "mean")
    if "mase" in subset:
        aggregations["mase"] = ("mase", "mean")
    summary = subset.groupby(group_keys, as_index=False).agg(**aggregations).sort_values(["material", "unit", "wape", "mae"])
    if {"abs_error_sum", "actual_sum"}.issubset(summary.columns):
        summary["wape"] = summary["abs_error_sum"] / summary["actual_sum"].replace(0, np.nan)
        summary["wape"] = summary["wape"].fillna(summary["wape"].mean())
    if "signed_error_sum" in subset:
        summary["bias_ratio"] = summary["signed_error_sum"].abs() / summary["actual_sum"].replace(0, np.nan)
        summary["bias_ratio"] = summary["bias_ratio"].fillna(0.0)
        summary["selection_score"] = summary["wape"] + float(bias_penalty) * summary["bias_ratio"]
    else:
        summary["bias_ratio"] = 0.0
        summary["selection_score"] = summary["wape"]
    if "forecast_sum" in summary and "actual_sum" in summary:
        summary["coverage_ratio"] = summary["forecast_sum"] / summary["actual_sum"].replace(0, np.nan)
        eligible = summary[summary["coverage_ratio"] >= float(coverage_floor)]
        if not eligible.empty:
            summary = eligible
    else:
        summary["coverage_ratio"] = np.nan
    summary = summary.sort_values(["material", "unit", "wape", "selection_score", "mae"])
    return summary.groupby(["material", "unit"], as_index=False).head(1).reset_index(drop=True)


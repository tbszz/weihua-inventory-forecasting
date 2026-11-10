import numpy as np
import pandas as pd

from src.backtest import evaluate_models, select_models


def test_backtest_uses_only_past_data_and_returns_metrics():
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    demand = pd.DataFrame({"material": "A", "unit": "PC", "date": dates, "demand_qty": np.arange(60, dtype=float)})
    metrics, predictions = evaluate_models(demand, horizons=(7,), min_train=28, step=7)
    assert not metrics.empty
    assert {"model", "horizon", "wape", "mae", "origin"}.issubset(metrics.columns)
    assert (predictions["origin"] < predictions["date"]).all()
    assert (predictions["forecast_qty"] >= 0).all()


def test_select_models_returns_one_model_per_series():
    metrics = pd.DataFrame(
        [
            {"material": "A", "unit": "PC", "model": "mean", "horizon": 7, "window": 28, "wape": 0.4, "mae": 2, "abs_error_sum": 40, "actual_sum": 100, "signed_error_sum": 0, "forecast_sum": 100},
            {"material": "A", "unit": "PC", "model": "ewma", "horizon": 7, "window": 28, "wape": 0.2, "mae": 3, "abs_error_sum": 20, "actual_sum": 100, "signed_error_sum": -90, "forecast_sum": 0},
        ]
    )
    chosen = select_models(metrics, horizon=7)
    assert chosen.iloc[0]["model"] == "mean"


def test_selection_penalizes_systematic_underforecast():
    metrics = pd.DataFrame(
        [
            {"material": "A", "unit": "PC", "model": "median", "horizon": 7, "window": 28, "wape": 0.2, "mae": 1, "abs_error_sum": 20, "actual_sum": 100, "signed_error_sum": -100, "forecast_sum": 100},
            {"material": "A", "unit": "PC", "model": "dow_mean", "horizon": 7, "window": 28, "wape": 0.2, "mae": 2, "abs_error_sum": 20, "actual_sum": 100, "signed_error_sum": 0, "forecast_sum": 100},
        ]
    )
    chosen = select_models(metrics, horizon=7)
    assert chosen.iloc[0]["model"] == "dow_mean"


def test_selection_rejects_zero_forecast_when_coverage_is_available():
    metrics = pd.DataFrame(
        [
            {"material": "A", "unit": "PC", "model": "median", "horizon": 7, "window": 28, "wape": 0.1, "mae": 1, "abs_error_sum": 10, "actual_sum": 100, "signed_error_sum": -10, "forecast_sum": 0},
            {"material": "A", "unit": "PC", "model": "mean", "horizon": 7, "window": 28, "wape": 0.35, "mae": 3, "abs_error_sum": 35, "actual_sum": 100, "signed_error_sum": 0, "forecast_sum": 80},
        ]
    )
    chosen = select_models(metrics, horizon=7)
    assert chosen.iloc[0]["model"] == "mean"


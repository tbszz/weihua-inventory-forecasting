import numpy as np
import pandas as pd

from src.forecast import forecast


def test_forecasters_are_non_negative_and_have_requested_horizon():
    history = np.array([0, 0, 4, 0, 0, 8, 0, 0, 0, 6], dtype=float)
    for model in ("mean", "wma", "ewma", "croston"):
        values, meta = forecast(history, 7, model=model, window=5)
        assert len(values) == 7
        assert np.isfinite(values).all()
        assert (values >= 0).all()
        assert meta["model"] == model


def test_mean_forecast_constant_series():
    values, _ = forecast(np.full(10, 3.0), 4, model="mean", window=5)
    assert np.allclose(values, 3.0)


def test_short_history_falls_back_to_available_history():
    values, _ = forecast(np.array([2.0, 4.0]), 3, model="wma", window=28)
    assert np.allclose(values, 10 / 3)


def test_dow_mean_uses_future_weekday_profile():
    dates = pd.date_range("2026-01-05", periods=14, freq="D")
    history = np.array([10 if d.dayofweek == 0 else 2 if d.dayofweek == 1 else 0 for d in dates], dtype=float)
    future_dates = pd.date_range("2026-01-19", periods=2, freq="D")
    values, meta = forecast(
        history,
        2,
        model="dow_mean",
        window=14,
        history_dates=dates,
        future_dates=future_dates,
    )
    assert np.allclose(values, [10, 2])
    assert meta["model"] == "dow_mean"


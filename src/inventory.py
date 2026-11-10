from __future__ import annotations

from statistics import NormalDist
from typing import Iterable, Mapping

import numpy as np


def calculate_safety_stock(
    mean_demand: float,
    demand_std: float,
    mean_lead_days: float,
    lead_std: float,
    *,
    service_level: float = 0.95,
) -> float:
    z = NormalDist().inv_cdf(min(max(service_level, 0.5), 0.9999))
    variance = max(0.0, mean_lead_days) * max(0.0, demand_std) ** 2 + max(0.0, mean_demand) ** 2 * max(0.0, lead_std) ** 2
    return max(0.0, float(z * np.sqrt(variance)))


def simulate_inventory(
    current_inventory: float,
    forecast_demand: Iterable[float],
    *,
    receipts: Mapping[int, float] | None = None,
) -> dict:
    receipts = receipts or {}
    balance = float(current_inventory)
    trajectory: list[float] = []
    stockout_day: int | None = None
    for day, demand in enumerate(forecast_demand, start=1):
        balance += float(receipts.get(day, 0.0))
        balance -= max(0.0, float(demand))
        trajectory.append(balance)
        if balance < 0 and stockout_day is None:
            stockout_day = day
    minimum = min(trajectory) if trajectory else float(current_inventory)
    return {
        "trajectory": trajectory,
        "stockout_day": stockout_day,
        "minimum_inventory": float(minimum),
        "gap_qty": float(max(0.0, -minimum)),
    }


def risk_level(stockout_day: int | None, reorder_day: int | None) -> str:
    if stockout_day is not None:
        return "red"
    if reorder_day is not None:
        return "orange"
    return "green"


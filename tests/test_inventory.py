import numpy as np

from src.inventory import calculate_safety_stock, simulate_inventory


def test_safety_stock_increases_with_demand_or_lead_time_volatility():
    base = calculate_safety_stock(3, 1, 5, 1)
    more_demand = calculate_safety_stock(3, 2, 5, 1)
    more_lead = calculate_safety_stock(3, 1, 8, 1)
    assert more_demand > base
    assert more_lead > base


def test_simulation_finds_stockout_and_gap():
    result = simulate_inventory(10, np.array([4, 4, 4, 4], dtype=float), receipts={2: 5})
    assert result["stockout_day"] == 4
    assert result["minimum_inventory"] < 0
    assert result["gap_qty"] > 0


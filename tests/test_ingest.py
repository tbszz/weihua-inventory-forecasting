from pathlib import Path

import pytest

from src.ingest import load_latest_inventory, load_workbooks


RAW = Path(__file__).parents[1] / "data" / "raw"
PRIVATE_DATA_AVAILABLE = (RAW / "出入库数据.XLSX").exists() and (RAW / "工字钢-出入库数据(1).XLSX").exists()

pytestmark = pytest.mark.skipif(not PRIVATE_DATA_AVAILABLE, reason="private source workbooks are not checked into the public repository")


def test_load_workbooks_normalizes_materials_and_units():
    demand, lead = load_workbooks(RAW)
    assert {"7005002850", "1009000034", "16T0000022", "7002000342"}.issubset(
        set(demand["material"].astype(str))
    )
    assert {"KG", "M"}.issubset(set(demand.loc[demand.material == "7002000342", "unit"]))
    assert (demand["demand_qty"] >= 0).all()
    assert demand[["material", "unit", "date"]].duplicated().sum() == 0
    assert {"material", "lead_days"}.issubset(lead.columns)
    assert (lead["lead_days"] > 0).all()


def test_latest_inventory_preserves_steel_units():
    inventory = load_latest_inventory(RAW)
    steel = inventory[inventory.material == "7002000342"]
    assert set(steel.unit) == {"KG", "M"}
    assert (steel.inventory_qty >= 0).all()


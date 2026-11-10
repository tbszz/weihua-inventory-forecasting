from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openpyxl import load_workbook


def _rows(path: Path, sheet: str) -> tuple[list[str], list[tuple]]:
    ws = load_workbook(path, read_only=True, data_only=True)[sheet]
    values = list(ws.iter_rows(values_only=True))
    if not values:
        return [], []
    header = [str(v).strip() if v is not None else "" for v in values[0]]
    return header, values[1:]


def _index(header: list[str], *names: str, occurrence: int = 0) -> int | None:
    for name in names:
        target = "".join(str(name).split())
        matches = [i for i, value in enumerate(header) if "".join(str(value).split()) == target]
        if len(matches) > occurrence:
            return matches[occurrence]
    return None


def _material_code(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def _as_number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _parse_outbound(path: Path, sheet: str) -> list[dict]:
    header, rows = _rows(path, sheet)
    date_i = _index(header, "过账日期")
    material_i = _index(header, "物料")
    qty_i = _index(header, "数量", "出库数量")
    unit_i = _index(header, "基本计量单位", "单位", "条目单位")
    movement_i = _index(header, "移动类型文本")
    if None in (date_i, material_i, qty_i):
        return []
    result: list[dict] = []
    for row in rows:
        date = row[date_i] if date_i < len(row) else None
        material = _material_code(row[material_i] if material_i < len(row) else None)
        if not material or not hasattr(date, "date"):
            continue
        movement = str(row[movement_i]).strip() if movement_i is not None and movement_i < len(row) else ""
        if "初始条目" in movement or "余额" in movement:
            continue
        qty = abs(_as_number(row[qty_i] if qty_i < len(row) else 0))
        if qty <= 0:
            continue
        unit = str(row[unit_i]).strip() if unit_i is not None and unit_i < len(row) and row[unit_i] else "UNKNOWN"
        result.append({"material": material, "unit": unit, "date": pd.Timestamp(date).normalize(), "demand_qty": qty})
    return result


def _parse_steel_parallel(path: Path, sheet: str) -> list[dict]:
    header, rows = _rows(path, sheet)
    date_i = _index(header, "过账日期")
    material_i = _index(header, "物料")
    qty_kg_i = _index(header, "数量")
    qty_m_i = _index(header, "以并行UoM计的数量")
    unit_i = _index(header, "基本计量单位")
    movement_i = _index(header, "移动类型文本")
    if None in (date_i, material_i, qty_kg_i):
        return []
    result: list[dict] = []
    for row in rows:
        date = row[date_i] if date_i < len(row) else None
        material = _material_code(row[material_i] if material_i < len(row) else None)
        if not material or not hasattr(date, "date"):
            continue
        movement = str(row[movement_i]).strip() if movement_i is not None and movement_i < len(row) else ""
        if "初始条目" in movement or "余额" in movement:
            continue
        kg = abs(_as_number(row[qty_kg_i] if qty_kg_i < len(row) else 0))
        if kg > 0:
            result.append({"material": material, "unit": "KG", "date": pd.Timestamp(date).normalize(), "demand_qty": kg})
        if qty_m_i is not None and qty_m_i < len(row):
            meters = abs(_as_number(row[qty_m_i]))
            if meters > 0:
                result.append({"material": material, "unit": "M", "date": pd.Timestamp(date).normalize(), "demand_qty": meters})
    return result


def _parse_lead(path: Path, sheet: str) -> list[dict]:
    header, rows = _rows(path, sheet)
    material_i = _index(header, "物料")
    lead_i = _index(header, "实际交货天数")
    apply_i = _index(header, "申报时间")
    arrival_i = _index(header, "过账日期")
    supplier_i = _index(header, "供应商")
    if material_i is None:
        return []
    result: list[dict] = []
    for row in rows:
        material = _material_code(row[material_i] if material_i < len(row) else None)
        if not material:
            continue
        lead = _as_number(row[lead_i]) if lead_i is not None and lead_i < len(row) else 0.0
        if lead <= 0 and apply_i is not None and arrival_i is not None and apply_i < len(row) and arrival_i < len(row):
            start, end = row[apply_i], row[arrival_i]
            if hasattr(start, "date") and hasattr(end, "date"):
                lead = (pd.Timestamp(end).normalize() - pd.Timestamp(start).normalize()).days
        if lead <= 0:
            continue
        supplier = str(row[supplier_i]).strip() if supplier_i is not None and supplier_i < len(row) and row[supplier_i] else "UNKNOWN"
        result.append({"material": material, "supplier": supplier, "lead_days": float(lead)})
    return result


def _daily_frame(records: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["material", "unit", "date", "demand_qty"])
    frame = frame.groupby(["material", "unit", "date"], as_index=False)["demand_qty"].sum()
    expanded: list[pd.DataFrame] = []
    for (material, unit), group in frame.groupby(["material", "unit"], sort=True):
        dates = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        indexed = group.set_index("date")["demand_qty"].reindex(dates, fill_value=0.0)
        expanded.append(pd.DataFrame({"material": material, "unit": unit, "date": dates, "demand_qty": indexed.to_numpy()}))
    return pd.concat(expanded, ignore_index=True).sort_values(["material", "unit", "date"]).reset_index(drop=True)


def load_workbooks(input_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = Path(input_dir)
    ordinary = input_dir / "出入库数据.XLSX"
    steel = input_dir / "工字钢-出入库数据(1).XLSX"
    outbound: list[dict] = []
    leads: list[dict] = []
    if ordinary.exists():
        outbound.extend(_parse_outbound(ordinary, "出库明细"))
        leads.extend(_parse_lead(ordinary, "入库明细"))
    if steel.exists():
        outbound.extend(_parse_steel_parallel(steel, "出库明细"))
        leads.extend(_parse_lead(steel, "入库明细"))
    demand = _daily_frame(outbound)
    lead = pd.DataFrame(leads, columns=["material", "supplier", "lead_days"])
    if not lead.empty:
        lead = lead.drop_duplicates().reset_index(drop=True)
    return demand, lead


def load_latest_inventory(input_dir: str | Path) -> pd.DataFrame:
    """Read the latest cached stock values from the material-level sheets."""
    input_dir = Path(input_dir)
    records: list[dict] = []
    specs = [
        (input_dir / "出入库数据.XLSX", ["7005002850", "1009000034", "16T0000022"]),
        (input_dir / "工字钢-出入库数据(1).XLSX", ["7002000342"]),
    ]
    for path, sheets in specs:
        if not path.exists():
            continue
        for sheet in sheets:
            header, rows = _rows(path, sheet)
            date_i = _index(header, "过账日期")
            material_i = _index(header, "物料")
            unit_i = _index(header, "单位", "基本计量单位")
            stock_i = _index(header, "每日库存量")
            stock_kg_i = _index(header, "每日库存量KG")
            stock_m_i = _index(header, "每日库存量M")
            if date_i is None:
                continue
            candidates: list[tuple[pd.Timestamp, str, float]] = []
            for row in rows:
                date = row[date_i] if date_i < len(row) else None
                if not hasattr(date, "date"):
                    continue
                material = _material_code(row[material_i] if material_i is not None and material_i < len(row) else sheet)
                if not material:
                    continue
                if stock_i is not None and stock_i < len(row):
                    unit = str(row[unit_i]).strip() if unit_i is not None and unit_i < len(row) and row[unit_i] else "PC"
                    candidates.append((pd.Timestamp(date), unit, _as_number(row[stock_i])))
                else:
                    if stock_kg_i is not None and stock_kg_i < len(row):
                        candidates.append((pd.Timestamp(date), "KG", _as_number(row[stock_kg_i])))
                    if stock_m_i is not None and stock_m_i < len(row):
                        candidates.append((pd.Timestamp(date), "M", _as_number(row[stock_m_i])))
            latest_by_unit: dict[str, tuple[pd.Timestamp, float]] = {}
            for date, unit, value in candidates:
                if unit not in latest_by_unit or date > latest_by_unit[unit][0]:
                    latest_by_unit[unit] = (date, value)
            for unit, (date, value) in latest_by_unit.items():
                records.append({"material": material, "unit": unit, "date": date, "inventory_qty": value})
    return pd.DataFrame(records, columns=["material", "unit", "date", "inventory_qty"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()
    demand, lead = load_workbooks(args.input_dir)
    print(demand.head().to_string(index=False))
    print(f"demand rows={len(demand):,}; lead rows={len(lead):,}")


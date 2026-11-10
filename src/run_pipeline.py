from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.backtest import evaluate_models, select_models
    from src.forecast import forecast
    from src.ingest import load_latest_inventory, load_workbooks
    from src.inventory import calculate_safety_stock, simulate_inventory
    from src.report import write_report
else:
    from .backtest import evaluate_models, select_models
    from .forecast import forecast
    from .ingest import load_latest_inventory, load_workbooks
    from .inventory import calculate_safety_stock, simulate_inventory
    from .report import write_report


def _lead_summary(lead: pd.DataFrame, material: str) -> tuple[float, float, float]:
    subset = lead[lead.material.astype(str) == str(material)] if not lead.empty else pd.DataFrame()
    if subset.empty:
        return 7.0, 7.0, 0.0
    values = subset.lead_days.astype(float)
    return float(values.median()), float(values.quantile(0.9)), float(values.std(ddof=0) or 0.0)


def run_pipeline(input_dir: str | Path, output_dir: str | Path) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    demand, lead = load_workbooks(input_dir)
    inventory = load_latest_inventory(input_dir)
    evaluations = [evaluate_models(demand, horizons=(7, 14), min_train=56, step=7, window=window) for window in (14, 21, 28, 42, 56, 84, 180)]
    metrics = pd.concat([result[0] for result in evaluations], ignore_index=True)
    predictions = pd.concat([result[1] for result in evaluations], ignore_index=True)
    selected = select_models(metrics, horizon=7)
    selected_map = {(str(row.material), str(row.unit)): (str(row.model), int(row.window)) for row in selected.itertuples()}
    risk_rows: list[dict] = []
    for (material, unit), group in demand.groupby(["material", "unit"], sort=True):
        group = group.sort_values("date")
        values = group.demand_qty.to_numpy(dtype=float)
        history_dates = pd.to_datetime(group.date).to_numpy()
        model, window = selected_map.get((str(material), str(unit)), ("ewma", 28))
        horizon = 14
        future_dates = pd.date_range(pd.Timestamp(history_dates[-1]) + pd.Timedelta(days=1), periods=horizon, freq="D")
        future, meta = forecast(
            values,
            horizon,
            model=model,
            window=window,
            history_dates=history_dates,
            future_dates=future_dates,
        )
        recent = values[-28:] if len(values) else np.array([0.0])
        mean_demand = float(recent.mean())
        demand_std = float(recent.std(ddof=0))
        mean_lead, p90_lead, lead_std = _lead_summary(lead, str(material))
        safety = calculate_safety_stock(mean_demand, demand_std, p90_lead, lead_std, service_level=0.95)
        rop = mean_demand * p90_lead + safety
        inv_row = inventory[(inventory.material.astype(str) == str(material)) & (inventory.unit == unit)]
        current_inventory = float(inv_row.iloc[0].inventory_qty) if not inv_row.empty else 0.0
        simulation = simulate_inventory(current_inventory, future)
        reorder_day = next((i + 1 for i, value in enumerate(simulation["trajectory"]) if value < rop), None)
        risk = "red" if simulation["stockout_day"] is not None else ("orange" if reorder_day is not None else "green")
        risk_rows.append({
            "material": material,
            "unit": unit,
            "model": model,
            "window": window,
            "model_level": meta["level"],
            "current_inventory": current_inventory,
            "mean_daily_demand": mean_demand,
            "demand_std": demand_std,
            "lead_median_days": mean_lead,
            "lead_p90_days": p90_lead,
            "safety_stock": safety,
            "reorder_point": rop,
            "stockout_day": simulation["stockout_day"],
            "reorder_day": reorder_day,
            "gap_qty": simulation["gap_qty"],
            "recommended_order_qty": max(0.0, rop + float(future.sum()) - current_inventory),
            "risk_level": risk,
        })
    risk = pd.DataFrame(risk_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output_dir / "selected_models.csv", index=False, encoding="utf-8-sig")
    risk.to_csv(output_dir / "risk.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, demand=demand, metrics=metrics, selected=selected, risk=risk)
    manifest = {"materials": int(demand.material.nunique()), "series": int(demand[["material", "unit"]].drop_duplicates().shape[0]), "metrics": int(len(metrics)), "risk_rows": int(len(risk))}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.input_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


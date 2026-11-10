# Inventory Forecasting MVP

This project turns warehouse transaction workbooks into daily demand series, compares forecasting baselines with rolling-origin backtests, and calculates dynamic safety stock and stockout risk.

The public repository intentionally excludes the source warehouse workbooks and generated reports. Put the two private files under `data/raw/` before running the full pipeline:

```text
data/raw/出入库数据.XLSX
data/raw/工字钢-出入库数据(1).XLSX
```

The current candidate set includes historical mean, weighted mean, EWMA, Croston, robust median, and weekday demand profile. Selection uses rolling-origin WAPE with a 65% forecast-coverage guard, so a zero forecast cannot win merely by avoiding spikes.

Run from inside `inventory-forecasting/`:

```powershell
python -m pytest tests -q
python src/run_pipeline.py --input-dir data/raw --output-dir reports/latest
```

The pipeline writes local-only CSV and Markdown reports under `reports/latest/`.

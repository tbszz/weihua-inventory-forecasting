from pathlib import Path

import pandas as pd
import pytest

from src.run_pipeline import run_pipeline


RAW = Path(__file__).parents[1] / "data" / "raw"
pytestmark = pytest.mark.skipif(
    not ((RAW / "出入库数据.XLSX").exists() and (RAW / "工字钢-出入库数据(1).XLSX").exists()),
    reason="private source workbooks are not checked into the public repository",
)


def test_pipeline_writes_metrics_predictions_and_report(tmp_path):
    output = run_pipeline(RAW, tmp_path)
    for name in ("metrics.csv", "predictions.csv", "selected_models.csv", "risk.csv", "report.md", "accuracy-review.md"):
        assert (tmp_path / name).exists(), name
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert not metrics.empty
    assert output["materials"] >= 4
    selected = pd.read_csv(tmp_path / "selected_models.csv")
    assert {"coverage_ratio", "selection_score"}.issubset(selected.columns)
    for row in selected.itertuples():
        baseline = metrics[
            (metrics.material.astype(str) == str(row.material))
            & (metrics.unit == row.unit)
            & (metrics.model == "mean")
            & (metrics.window == 180)
            & (metrics.horizon == row.horizon)
        ]
        assert row.wape <= baseline.abs_error_sum.sum() / baseline.actual_sum.sum() + 1e-9


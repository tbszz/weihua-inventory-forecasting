from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_report(
    output_dir: str | Path,
    *,
    demand: pd.DataFrame,
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    risk: pd.DataFrame,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = metrics[metrics.model == "mean"] if not metrics.empty else metrics
    selected_columns = [column for column in ["material", "unit", "model", "horizon", "window", "wape", "mae", "bias_ratio", "coverage_ratio", "selection_score"] if column in selected.columns]
    selected_summary = selected[selected_columns].to_string(index=False) if not selected.empty else "无可用模型"
    risk_summary = risk.sort_values(["risk_level", "stockout_day"], na_position="last").head(20).to_string(index=False) if not risk.empty else "无风险结果"
    comparison_rows = []
    for row in selected.itertuples():
        selected_wape = float(row.wape)
        baseline = metrics[(metrics.material.astype(str) == str(row.material)) & (metrics.unit == row.unit) & (metrics.model == "mean") & (metrics.window == 180) & (metrics.horizon == row.horizon)]
        baseline_wape = float(baseline.abs_error_sum.sum() / baseline.actual_sum.sum()) if not baseline.empty and baseline.actual_sum.sum() else float("nan")
        improvement = (baseline_wape - selected_wape) / baseline_wape if baseline_wape else float("nan")
        comparison_rows.append({"material": row.material, "unit": row.unit, "baseline_wape": baseline_wape, "selected_model": row.model, "selected_window": row.window, "selected_wape": selected_wape, "improvement": improvement})
    comparison = pd.DataFrame(comparison_rows)
    comparison_text = comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}") if not comparison.empty else "无可用对比"
    text = f"""# 库存预测准确率回测报告

生成时间：{pd.Timestamp.utcnow().isoformat()}

## 数据范围

- 物料/单位序列：{demand[['material', 'unit']].drop_duplicates().shape[0]} 条
- 日需求记录：{len(demand):,} 条
- 日期范围：{demand.date.min().date()} 至 {demand.date.max().date()}

## 模型选择

模型按物料/单位和预测窗口进行滚动起点回测，以 WAPE 加系统性偏差惩罚的 `selection_score` 选择。结果如下：

```text
{selected_summary}
```

## 基线对比

`mean` 是当前“历史日均用量”基线。详见 `metrics.csv`，其中同一物料/单位/窗口的模型可直接比较 WAPE、MAE 和 RMSE。

- 回测记录数：{len(metrics):,}
- 基线记录数：{len(baseline):,}

相对于“全历史日均值”（180 天窗口）基线，当前调参结果如下。正的 improvement 表示 WAPE 降低：

```text
{comparison_text}
```

## 风险结果

```text
{risk_summary}
```

## 使用限制

- 样例数据覆盖时间较短，模型结果用于 MVP 验证，不代表长期稳定性。
- 当前未来 BOM/订单需求和在途订单未接入，风险模拟只使用已知库存与历史交期统计。
- KG/M 作为独立单位序列计算，未进行隐式换算。
"""
    report_path = output_dir / "report.md"
    report_path.write_text(text, encoding="utf-8")
    (output_dir / "accuracy-review.md").write_text(text, encoding="utf-8")
    return report_path


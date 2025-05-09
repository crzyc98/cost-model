# cost_model.reporting.metrics

The metrics module takes your simulation output and reduces it to high-level KPIs: headcounts, participation rates, total and average contributions, ages, and more. Use this to produce summary tables for reporting and analysis.

Example usage:

```python
from cost_model.reporting.metrics import calculate_summary_metrics
summary = calculate_summary_metrics(results_df, config)
print(summary.head())
```

::: cost_model.reporting.metrics

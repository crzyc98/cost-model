# Phase 4: Onboarding & Visualization

This document outlines how to configure and extend the model for onboarding dynamics and how to visualize survival curves.

## 1. Onboarding Configuration

In `config.yaml`, under the top level add:
```yaml
onboarding:
  enabled: true                  # Turn on onboarding hazard & productivity effects
  early_tenure_months: 6        # Months after hire with elevated hazard
  hazard_multiplier: 1.5        # Multiply base hazard by this factor for onboarding
  productivity_curve:           # Productivity ramp (fraction of full productivity)
    - 0.80  # Month 1
    - 0.85  # Month 2
    - 0.90  # Month 3
    - 0.95  # Month 4
    - 1.00  # Month 5
    - 1.00  # Month 6+
```

- **hazard_multiplier** applies only for agents within their first `early_tenure_months` months.
- **productivity_curve** defines the fraction of a full-year compensation or contributions to apply for new hires in their first months.

## 2. Agent Behavior Adjustments

- In `RetirementPlanModel`, monthly hazards are already adjusted by `hazard_multiplier`.
- To apply the productivity curve, integrate `self.productivity_curve` in proration logic. For example:
  ```python
  # inside year-step, per-agent proration
  tenure = agent._calculate_tenure_months(current_date)
  if self.onboarding_enabled and tenure <= len(self.productivity_curve):
      factor = self.productivity_curve[tenure-1]
  else:
      factor = 1.0
  agent.prorated_compensation_for_reporting *= Decimal(str(factor))
  ```
- You can similarly scale initial contributions: multiply `deferral_rate * gross_compensation * factor`.

## 3. Visualization Guide

### 3.1 Survival Curves by Cohort

1. Install Jupyter and `lifelines`:
   ```bash
   pip install jupyterlab lifelines
   ```
2. Create a notebook (e.g., `notebooks/survival_plots.ipynb`) that:
   - Loads `data/historical_turnover.csv`.
   - Fits a `KaplanMeierFitter()` on durations and event flags by cohort.
   - Plots KM curves for each tenure bucket (0–1yr, 1–3yr, 3+yr).
   ```python
   from lifelines import KaplanMeierFitter
   import pandas as pd

   df = pd.read_csv('data/historical_turnover.csv', parse_dates=['hire_date','termination_date'])
   df['duration'] = (df['termination_date'].fillna(pd.Timestamp.today()) - df['hire_date']).dt.days/365
   df['cohort'] = df['duration'].apply(lambda x: '0-1yr' if x<=1 else '1-3yr' if x<=3 else '3+yr')

   kmf = KaplanMeierFitter()
   for cohort, data in df.groupby('cohort'):
       kmf.fit(data['duration'], data['event_observed'], label=cohort)
       kmf.plot_survival_function()
   ```

### 3.2 Hazard Function Plots

You can also plot estimated hazard functions using CoxPHFitter or non-parametric estimators.

## 4. Best Practices

- Validate that `hazard_multiplier` produces plausible separation curves.
- Check that `productivity_curve` factors sum to a realistic fraction of annual activity.
- Version control your notebooks in `notebooks/` and link them in `README.md`.

---

*Next Steps:* integrate the productivity ramp code snippet into your model, and scaffold `notebooks/survival_plots.ipynb` for exploratory analysis.

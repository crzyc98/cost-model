#!/usr/bin/env python3
"""
Standalone script to plot Kaplan–Meier survival curves by tenure cohort.
Run from project root: python3 scripts/plot_survival.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Ensure working directory is project root
os.chdir(os.path.dirname(os.path.dirname(__file__)))

# Load historical data
data_path = os.path.join('data', 'historical_turnover.csv')
print(f"Loading data from {data_path}")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")
df = pd.read_csv(data_path, parse_dates=['hire_date','termination_date'])

# Compute duration and event flag
df['duration'] = (df['termination_date'].fillna(pd.Timestamp.today()) - df['hire_date']).dt.days / 365
df['event_observed'] = df['termination_date'].notna().astype(int)

# Assign cohorts
def assign_cohort(x):
    if x <= 1:
        return '0-1yr'
    elif x <= 3:
        return '1-3yr'
    else:
        return '3+yr'

df['cohort'] = df['duration'].apply(assign_cohort)

# Plot survival curves
kmf = KaplanMeierFitter()
plt.figure(figsize=(8,6))
for cohort, group in df.groupby('cohort'):
    kmf.fit(group['duration'], group['event_observed'], label=cohort)
    kmf.plot_survival_function(ci_show=False)

plt.title('Survival Curves by Tenure Cohort')
plt.xlabel('Years Since Hire')
plt.ylabel('Survival Probability')
plt.legend()
plt.tight_layout()
# Save figure
out_fig = os.path.join('output', 'survival_curves.png')
os.makedirs('output', exist_ok=True)
plt.savefig(out_fig)
print(f"Survival curves saved to {out_fig}")
plt.show()

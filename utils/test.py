import pandas as pd
df2 = pd.read_parquet("plan_outputs/Baseline_year1.parquet")
print(df2.columns)
print(df2[['employee_plan_year_compensation', 'deferral_rate', 'is_participating']].head())
## QuickStart

To calculate employee tenure and tenure bands programmatically:

```python
import pandas as pd
from pathlib import Path
from cost_model.state.tenure import assign_tenure_band, apply_tenure

# Create or load a DataFrame with employee hire dates
employees_df = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
    'employee_hire_date': [
        '2020-01-15',  # ~5 years tenure
        '2023-06-01',  # ~2 years tenure
        '2024-11-15',  # <1 year tenure
        '2019-03-10',  # >6 years tenure
        '2021-08-20',  # ~4 years tenure
    ],
    'employee_role': ['Engineer', 'Analyst', 'Manager', 'Director', 'Engineer'],
    'employee_gross_compensation': [85000, 65000, 110000, 150000, 75000]
})

# Set the reference date for tenure calculation
as_of_date = pd.Timestamp('2025-12-31')  # End of 2025

# Apply tenure calculations to the DataFrame
employees_df = apply_tenure(
    df=employees_df,
    hire_col='employee_hire_date',
    as_of=as_of_date,
    out_tenure_col='employee_tenure',
    out_band_col='tenure_band'
)

# Examine the results
print(employees_df[['employee_id', 'employee_hire_date', 'employee_tenure', 'tenure_band']])

# Calculate tenure distribution
tenure_distribution = employees_df['tenure_band'].value_counts()
print("\nTenure distribution:")
print(tenure_distribution)

# Filter employees by tenure band
senior_employees = employees_df[employees_df['tenure_band'] == '5+']
print(f"\nEmployees with 5+ years tenure: {len(senior_employees)}")

# Calculate average compensation by tenure band
avg_comp_by_tenure = employees_df.groupby('tenure_band')['employee_gross_compensation'].mean()
print("\nAverage compensation by tenure band:")
print(avg_comp_by_tenure)

# Manual tenure band assignment
new_employee_tenure = 0.75  # 9 months
tenure_band = assign_tenure_band(new_employee_tenure)
print(f"\nAn employee with {new_employee_tenure} years tenure is in band: {tenure_band}")
```

This demonstrates how to calculate employee tenure in years, assign tenure bands, and analyze tenure distributions.
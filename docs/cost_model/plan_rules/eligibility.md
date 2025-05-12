## QuickStart

To determine employee eligibility programmatically:

```python
import pandas as pd
from datetime import date
from pathlib import Path
from cost_model.plan_rules.eligibility import run
from cost_model.config.plan_rules import EligibilityConfig

# Create a sample snapshot with employee data
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
    'employee_hire_date': [
        pd.Timestamp('2023-01-15'),  # 2+ years of service
        pd.Timestamp('2024-06-01'),  # 1 year of service
        pd.Timestamp('2025-01-01'),  # 6 months of service
        pd.Timestamp('2025-05-01')   # 1 month of service
    ],
    'employee_birth_date': [
        pd.Timestamp('1990-05-12'),  # 35 years old
        pd.Timestamp('1995-08-23'),  # 30 years old
        pd.Timestamp('2000-11-30'),  # 25 years old
        pd.Timestamp('2005-03-15')   # 20 years old
    ],
    'active': [True, True, True, True]
}).set_index('employee_id')

# Set the reference date
as_of = date(2025, 6, 15)

# Configure eligibility rules
eligibility_config = EligibilityConfig(
    min_age=21,                # Minimum age requirement
    min_service_months=12      # Minimum service requirement in months
)

# Generate eligibility events
eligibility_events = run(snapshot, as_of, eligibility_config)

# Examine the generated events
if eligibility_events:
    events_df = pd.concat(eligibility_events)
    print(f"Generated {len(events_df)} eligibility events")
    print(events_df[['employee_id', 'event_type']].head())
    
    # Identify which employees are eligible
    eligible_employees = events_df['employee_id'].unique()
    print(f"\nEligible employees: {eligible_employees}")
    
    # Identify which employees are not eligible
    all_employees = snapshot.index.tolist()
    ineligible_employees = [emp for emp in all_employees if emp not in eligible_employees]
    print(f"Ineligible employees: {ineligible_employees}")
    
    # Save the events
    output_dir = Path('output/eligibility')
    output_dir.mkdir(parents=True, exist_ok=True)
    events_df.to_parquet(output_dir / 'eligibility_events_2025.parquet')
else:
    print("No eligibility events generated")
```

This demonstrates how to determine employee eligibility based on age and service requirements.
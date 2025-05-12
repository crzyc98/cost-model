## QuickStart

To generate new hires programmatically:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from cost_model.dynamics.hiring import generate_new_hires

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(seed=42)

# Define the number of new hires and the year
num_hires = 5
hire_year = 2025

# Define the existing employee IDs (to avoid duplicates)
existing_ids = ['EMP001', 'EMP002', 'EMP003', 'EMP004']

# Create a configuration for the hiring process
scenario_config = {
    # Role distribution
    'new_hire_roles': {
        'Engineer': 0.6,    # 60% Engineers
        'Analyst': 0.3,     # 30% Analysts
        'Manager': 0.1      # 10% Managers
    },
    
    # Age distribution parameters
    'new_hire_age': {
        'mean': 30,         # Mean age of 30
        'std_dev': 5,       # Standard deviation of 5 years
        'min': 22,          # Minimum age of 22
        'max': 45           # Maximum age of 45
    },
    
    # Compensation parameters by role
    'new_hire_comp': {
        'Engineer': {
            'base': 70000,   # Base salary for Engineers
            'age_factor': 1000,  # Increase per year of age
            'std_dev': 5000      # Standard deviation
        },
        'Analyst': {
            'base': 55000,   # Base salary for Analysts
            'age_factor': 800,   # Increase per year of age
            'std_dev': 4000      # Standard deviation
        },
        'Manager': {
            'base': 90000,   # Base salary for Managers
            'age_factor': 1500,  # Increase per year of age
            'std_dev': 7000      # Standard deviation
        }
    },
    
    # Default compensation parameters (used if role-specific not available)
    'default_comp': {
        'base': 60000,      # Default base salary
        'age_factor': 1000, # Default increase per year of age
        'std_dev': 5000     # Default standard deviation
    }
}

# Generate the new hires
new_hires = generate_new_hires(
    num_hires=num_hires,
    hire_year=hire_year,
    scenario_config=scenario_config,
    existing_ids=existing_ids,
    rng=rng
)

# Examine the new hires
print(f"Generated {len(new_hires)} new hires for {hire_year}:")
print(new_hires[['employee_hire_date', 'employee_birth_date', 'employee_role', 'employee_gross_compensation']])

# Analyze the distribution of roles
role_counts = new_hires['employee_role'].value_counts()
print("\nRole distribution:")
for role, count in role_counts.items():
    print(f"  {role}: {count} ({count/len(new_hires)*100:.1f}%)")

# Analyze the age distribution
hire_dates = pd.to_datetime(new_hires['employee_hire_date'])
birth_dates = pd.to_datetime(new_hires['employee_birth_date'])

# Calculate ages at hire date
ages = []
for hire_date, birth_date in zip(hire_dates, birth_dates):
    age = hire_date.year - birth_date.year - ((hire_date.month, hire_date.day) < (birth_date.month, birth_date.day))
    ages.append(age)

new_hires['age_at_hire'] = ages

print(f"\nAge distribution:")
print(f"  Average age: {new_hires['age_at_hire'].mean():.1f} years")
print(f"  Minimum age: {new_hires['age_at_hire'].min()} years")
print(f"  Maximum age: {new_hires['age_at_hire'].max()} years")

# Analyze compensation by role
print("\nCompensation by role:")
for role in new_hires['employee_role'].unique():
    role_df = new_hires[new_hires['employee_role'] == role]
    avg_comp = role_df['employee_gross_compensation'].mean()
    print(f"  {role}: ${avg_comp:,.2f} average")

# Save the new hires
output_dir = Path('output/hiring')
output_dir.mkdir(parents=True, exist_ok=True)
new_hires.to_parquet(output_dir / f'new_hires_{hire_year}.parquet')

# Generate new hires for multiple years
multi_year_hires = {}
for year in range(2025, 2028):
    # Adjust number of hires each year
    year_hires = generate_new_hires(
        num_hires=num_hires + (year - 2025) * 2,  # Increase hires each year
        hire_year=year,
        scenario_config=scenario_config,
        existing_ids=existing_ids,
        rng=rng
    )
    multi_year_hires[year] = year_hires
    existing_ids.extend(year_hires.index.tolist())  # Update existing IDs
    
    print(f"\nGenerated {len(year_hires)} new hires for {year}")
    print(f"Average compensation: ${year_hires['employee_gross_compensation'].mean():,.2f}")

# Combine all years of new hires
all_hires = pd.concat(multi_year_hires.values())
all_hires.to_parquet(output_dir / 'all_new_hires.parquet')
```

This demonstrates how to generate new hires with configurable role distributions, age distributions, and compensation parameters.
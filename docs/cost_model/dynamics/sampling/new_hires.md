# New Hires Sampling Engine

The new hires sampling engine generates new employee hires for workforce simulations, including demographic and compensation details.

## QuickStart

To generate new hires in your workforce simulation:

```python
import pandas as pd
from datetime import datetime
from numpy.random import default_rng
from cost_model.dynamics.sampling.new_hires import sample_new_hires
from cost_model.utils.columns import EMP_HIRE_DATE, EMP_AGE, EMP_ROLE

# Set up the random number generator
rng = default_rng()

# Set the simulation parameters
year_end = pd.Timestamp('2025-12-31')

# Generate new hires
new_hires = sample_new_hires(
    year_end=year_end,
    rng=rng,
    n_hires=100,  # Number of hires to generate
    role_distribution={
        'Engineer': 0.4,
        'Manager': 0.2,
        'Analyst': 0.3,
        'Other': 0.1
    },
    age_distribution={
        'min_age': 22,
        'max_age': 65,
        'mean_age': 35,
        'std_age': 7
    },
    salary_distribution={
        'Engineer': {'min': 80000, 'max': 150000},
        'Manager': {'min': 120000, 'max': 200000},
        'Analyst': {'min': 60000, 'max': 120000},
        'Other': {'min': 50000, 'max': 180000}
    }
)

# Examine the generated hires
print(f"Generated {len(new_hires)} new hires")

# Show some sample hires
print("\nSample new hires:")
for _, row in new_hires.head().iterrows():
    print(f"  {row.name}: {row[EMP_ROLE]} hired on {row[EMP_HIRE_DATE].date()}, age {row[EMP_AGE]}")

# Save the results
output_dir = Path('output/new_hires')
output_dir.mkdir(parents=True, exist_ok=True)
new_hires.to_parquet(output_dir / 'new_hires_2025.parquet')
```

## Key Features

1. **Role-based Hiring**
   - Configurable role distribution percentages
   - Supports any number of role categories
   - Ensures total distribution sums to 1

2. **Age Distribution**
   - Configurable minimum and maximum ages
   - Normal distribution with mean and standard deviation
   - Automatic bounds checking

3. **Salary Distribution**
   - Role-specific salary ranges
   - Uniform distribution within role ranges
   - Automatic validation of salary ranges

4. **Date Management**
   - Random hiring dates within simulation year
   - Proper timestamp handling
   - Consistent date formatting

## Configuration Options

The engine can be configured with:

- `year_end`: End date of the simulation year
- `rng`: Random number generator (numpy.random.Generator)
- `n_hires`: Number of hires to generate
- `role_distribution`: Dictionary of role percentages
- `age_distribution`: Dictionary of age parameters
- `salary_distribution`: Dictionary of role-specific salary ranges

## Error Handling

The engine includes several safety checks:

1. Validates role distribution sums to 1
2. Ensures age parameters are within reasonable bounds
3. Verifies salary ranges are valid (min < max)
4. Handles missing configuration parameters with defaults
5. Logs warnings for out-of-bound values

## Performance Considerations

- Uses vectorized operations where possible for efficiency
- Implements efficient random sampling for large numbers of hires
- Uses numpy's random number generator for consistent results
- Handles edge cases efficiently (e.g., minimum hires)

## Example Scenarios

### Scenario 1: Basic New Hires Generation
```python
# Generate 50 new hires with default distributions
new_hires = sample_new_hires(
    year_end=pd.Timestamp('2025-12-31'),
    rng=rng,
    n_hires=50
)
```

### Scenario 2: Custom Role Distribution
```python
# Generate hires with custom role distribution
new_hires = sample_new_hires(
    year_end=pd.Timestamp('2025-12-31'),
    rng=rng,
    n_hires=100,
    role_distribution={
        'Developer': 0.5,
        'Designer': 0.2,
        'Product': 0.2,
        'Support': 0.1
    }
)
```

### Scenario 3: Custom Age and Salary Ranges
```python
# Generate hires with specific age and salary parameters
new_hires = sample_new_hires(
    year_end=pd.Timestamp('2025-12-31'),
    rng=rng,
    n_hires=75,
    age_distribution={
        'min_age': 25,
        'max_age': 60,
        'mean_age': 32,
        'std_age': 5
    },
    salary_distribution={
        'Developer': {'min': 90000, 'max': 160000},
        'Designer': {'min': 75000, 'max': 130000},
        'Product': {'min': 100000, 'max': 180000},
        'Support': {'min': 50000, 'max': 90000}
    }
)
```

## Troubleshooting

If you're not seeing expected hiring patterns:

1. Verify your role distribution sums to 1:
   ```python
   assert abs(sum(role_distribution.values()) - 1.0) < 1e-6
   ```

2. Check age parameters are reasonable:
   - Minimum age should be >= 18
   - Maximum age should be <= 70
   - Mean age should be between min and max

3. Validate salary ranges:
   - Minimum should be less than maximum for each role
   - Ranges should be consistent with industry standards

4. Review the random number generator seed for reproducibility
5. Check the debug logging for detailed sampling information

## Output Format

The engine returns a DataFrame with the following columns:

- `employee_id`: Unique identifier for each new hire
- `employee_hire_date`: Timestamp of hire date
- `employee_role`: Role assigned to the employee
- `employee_age`: Age of the employee
- `employee_salary`: Starting salary
- Any additional columns required by your simulation

## Best Practices

1. Use realistic role distributions based on your workforce data
2. Set age parameters based on your industry's typical hiring patterns
3. Use salary ranges that reflect your compensation structure
4. Consider seasonality in hiring dates
5. Validate distributions against historical hiring patterns
6. Monitor the output for any unexpected patterns
7. Use consistent random number generator seeds for reproducibility

# Salary Sampling Engine

The salary sampling engine generates realistic salary distributions for new hires and salary adjustments, using configurable distributions and role-based parameters.

## QuickStart

To generate salary samples for your workforce:

```python
import pandas as pd
from numpy.random import default_rng
from cost_model.dynamics.sampling.salary import sample_salary
from cost_model.utils.columns import EMP_ROLE, EMP_SALARY

# Set up the random number generator
rng = default_rng()

# Create a sample DataFrame with employee roles
employees = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003'],
    'employee_role': ['Engineer', 'Manager', 'Analyst']
}).set_index('employee_id')

# Sample salaries using role-based distributions
salaries = sample_salary(
    df=employees,
    rng=rng,
    salary_distribution={
        'Engineer': {
            'distribution': 'lognormal',
            'mean': 100000,  # Mean salary
            'sigma': 0.2     # Standard deviation (as fraction of mean)
        },
        'Manager': {
            'distribution': 'normal',
            'mean': 150000,
            'std': 20000
        },
        'Analyst': {
            'distribution': 'uniform',
            'min': 70000,
            'max': 120000
        }
    },
    salary_bounds={
        'min': 50000,    # Minimum salary
        'max': 200000,   # Maximum salary
        'role_min': {    # Optional role-specific minimums
            'Manager': 130000
        }
    }
)

# Examine the generated salaries
print(f"Generated {len(salaries)} salary samples")

# Show salary distribution by role
print("\nSalary distribution by role:")
for role in ['Engineer', 'Manager', 'Analyst']:
    role_salaries = salaries[employees[EMP_ROLE] == role]
    print(f"\n{role} salaries:")
    print(f"  Mean: ${role_salaries.mean():,.2f}")
    print(f"  Min: ${role_salaries.min():,.2f}")
    print(f"  Max: ${role_salaries.max():,.2f}")

# Save the results
output_dir = Path('output/salaries')
output_dir.mkdir(parents=True, exist_ok=True)
employees[EMP_SALARY] = salaries
employees.to_parquet(output_dir / 'salaries_2025.parquet')
```

## Key Features

1. **Distribution Types**
   - Lognormal distribution for realistic salary spread
   - Normal distribution for role-based salary ranges
   - Uniform distribution for flat salary ranges
   - Custom distribution support

2. **Role-based Parameters**
   - Configurable mean and standard deviation per role
   - Role-specific minimum and maximum salary bounds
   - Supports any number of role categories

3. **Bounds Management**
   - Global minimum and maximum salary bounds
   - Role-specific minimum salary overrides
   - Automatic clamping of generated values

4. **Quality Control**
   - Ensures generated salaries are positive
   - Maintains role hierarchy in salary ranges
   - Handles missing role parameters gracefully

## Configuration Options

The engine can be configured with:

- `df`: Input DataFrame containing employee data
- `rng`: Random number generator (numpy.random.Generator)
- `salary_distribution`: Dictionary of role-specific distribution parameters
- `salary_bounds`: Dictionary of salary bounds

## Distribution Types

1. **Lognormal Distribution**
   ```python
   'Engineer': {
       'distribution': 'lognormal',
       'mean': 100000,  # Mean salary
       'sigma': 0.2     # Standard deviation (as fraction of mean)
   }
   ```

2. **Normal Distribution**
   ```python
   'Manager': {
       'distribution': 'normal',
       'mean': 150000,
       'std': 20000
   }
   ```

3. **Uniform Distribution**
   ```python
   'Analyst': {
       'distribution': 'uniform',
       'min': 70000,
       'max': 120000
   }
   ```

## Error Handling

The engine includes several safety checks:

1. Validates distribution parameters
2. Ensures salary bounds are reasonable
3. Handles missing role parameters
4. Logs warnings for out-of-bound values
5. Maintains positive salary values

## Performance Considerations

- Uses vectorized operations where possible for efficiency
- Implements efficient random sampling for large datasets
- Uses numpy's random number generator for consistent results
- Handles edge cases efficiently (e.g., small datasets)

## Example Scenarios

### Scenario 1: Basic Salary Distribution
```python
# Generate salaries with default distributions
salaries = sample_salary(
    df=employees,
    rng=rng,
    salary_distribution={
        'Engineer': {'mean': 100000, 'std': 20000},
        'Manager': {'mean': 150000, 'std': 30000},
        'Analyst': {'mean': 80000, 'std': 15000}
    }
)
```

### Scenario 2: Custom Distribution Parameters
```python
# Generate salaries with custom distributions
salaries = sample_salary(
    df=employees,
    rng=rng,
    salary_distribution={
        'Developer': {
            'distribution': 'lognormal',
            'mean': 120000,
            'sigma': 0.15
        },
        'Designer': {
            'distribution': 'normal',
            'mean': 90000,
            'std': 15000
        }
    },
    salary_bounds={
        'min': 60000,
        'max': 180000,
        'role_min': {
            'Developer': 80000
        }
    }
)
```

## Troubleshooting

If you're not seeing expected salary distributions:

1. Verify your distribution parameters:
   - Lognormal: mean > 0, sigma > 0
   - Normal: mean > 0, std > 0
   - Uniform: min < max

2. Check salary bounds:
   - Global min should be > 0
   - Role-specific min should be > global min
   - All max values should be reasonable

3. Review the random number generator seed for reproducibility
4. Check the debug logging for detailed sampling information
5. Validate the generated values against your industry standards

## Output Format

The engine returns a Series with the generated salaries, which can be easily merged with your employee DataFrame. The output maintains:

- Positive salary values
- Role hierarchy in salary ranges
- Global and role-specific bounds
- Consistent data types

## Best Practices

1. Use realistic distribution parameters based on your industry data
2. Set appropriate salary bounds to maintain role hierarchy
3. Consider using lognormal distribution for realistic spread
4. Validate distributions against historical salary data
5. Monitor the output for any unexpected patterns
6. Use consistent random number generator seeds for reproducibility
7. Regularly update parameters based on market data

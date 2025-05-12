# Termination Sampling Engine

The termination sampling engine handles the probabilistic sampling of employee terminations during workforce simulations. It supports both rate-based and hazard-based termination sampling with special handling for annual rates.

## QuickStart

To sample terminations in your workforce simulation:

```python
import pandas as pd
from datetime import datetime
from numpy.random import default_rng
from cost_model.dynamics.sampling.terminations import sample_terminations
from cost_model.utils.columns import EMP_TERM_DATE, STATUS_COL

# Create a sample snapshot with employee data
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003'],
    'employee_hire_date': [
        pd.Timestamp('2023-01-15'),
        pd.Timestamp('2024-06-01'),
        pd.Timestamp('2025-01-01')
    ],
    'term_rate': [0.05, 0.10, 0.02],  # Hazard rates for each employee
    'active': [True, True, True]
}).set_index('employee_id')

# Set up the random number generator
rng = default_rng()

# Set the simulation parameters
year_end = pd.Timestamp('2025-12-31')
termination_rate = 0.05  # Annual termination rate

# Sample terminations
result = sample_terminations(
    df_out=snapshot.copy(),
    year_end=year_end,
    termination_rate=termination_rate,
    rng=rng
)

# Examine the results
if result:
    terminated_df = result[0]
    print(f"Generated {len(terminated_df)} terminations")
    
    # Show termination details
    print("\nTermination details:")
    for _, row in terminated_df.iterrows():
        if pd.notna(row[EMP_TERM_DATE]):
            print(f"  {row.name}: Terminated on {row[EMP_TERM_DATE].date()}")
    
    # Save the results
    output_dir = Path('output/terminations')
    output_dir.mkdir(parents=True, exist_ok=True)
    terminated_df.to_parquet(output_dir / 'terminations_2025.parquet')
else:
    print("No terminations generated")
```

## Key Features

1. **Flexible Rate Handling**
   - Supports both float/int annual rates and per-row hazard rates
   - For annual rates, implements exact sampling with weighted probabilities
   - For hazard rates, uses per-row probability sampling

2. **Weighted Sampling**
   - Can use per-row 'term_rate' column for weighted termination probabilities
   - Normalizes weights to ensure proper probability distribution
   - Handles cases with missing or zero hazard rates

3. **Date Validation**
   - Ensures termination dates are within the simulation year
   - Prevents terminations before hire dates
   - Clips dates to valid simulation window

4. **Status Management**
   - Automatically updates employee status to "Terminated"
   - Handles cases where termination dates are invalid
   - Maintains proper status transitions

## Configuration Options

The engine can be configured with:

- `termination_rate`: Annual termination rate (float or int)
- `year_end`: End date of the simulation year
- `rng`: Random number generator (numpy.random.Generator)
- `df_out`: Input DataFrame containing employee data

## Special Handling for Annual Rates

When using an annual rate:

1. Calculates exact number of terminations needed: `ceil(rate × active_employees)`
2. Implements weighted sampling if per-row hazard rates are available
3. Uses uniform sampling if no hazard rates are provided
4. Logs the exact number of terminations being generated

## Error Handling

The engine includes several safety checks:

1. Prevents terminations before hire dates
2. Clips termination dates to valid simulation window
3. Handles missing status columns by creating them
4. Logs warnings for invalid termination dates
5. Ensures proper data types for termination dates

## Performance Considerations

- Uses vectorized operations where possible for efficiency
- Implements efficient weighted sampling for large datasets
- Uses numpy's random number generator for consistent results
- Handles edge cases efficiently (e.g., no valid employees)

## Example Scenarios

### Scenario 1: Basic Annual Rate Sampling
```python
# Sample terminations with a 5% annual rate
result = sample_terminations(
    df_out=snapshot,
    year_end=pd.Timestamp('2025-12-31'),
    termination_rate=0.05,
    rng=rng
)
```

### Scenario 2: Hazard-based Sampling with Weights
```python
# Sample terminations using per-row hazard rates
result = sample_terminations(
    df_out=snapshot,
    year_end=pd.Timestamp('2025-12-31'),
    termination_rate=None,  # Use per-row term_rate
    rng=rng
)
```

## Troubleshooting

If you're not seeing expected termination behavior:

1. Verify that your snapshot contains the required columns:
   - `employee_id` (as index)
   - `employee_hire_date`
   - `term_rate` (if using hazard-based sampling)
   - `active` status

2. Check that your termination rate is appropriate:
   - For annual rates: Ensure it's between 0 and 1
   - For hazard rates: Ensure they're non-negative

3. Verify that your simulation year end date is correct
4. Check the random number generator seed for reproducibility
5. Review the debug logging for detailed sampling information

## Output Format

The engine modifies the input DataFrame in place and returns it. For each terminated employee, it:

1. Sets the termination date (`EMP_TERM_DATE`)
2. Updates the status to "Terminated" (`STATUS_COL`)
3. Ensures proper data types for all columns

## Best Practices

1. Always provide a consistent random number generator for reproducibility
2. Use appropriate termination rates based on your workforce characteristics
3. Consider using hazard-based sampling for more realistic termination patterns
4. Monitor the logging output for warnings about invalid termination dates
5. Validate the results against expected termination patterns

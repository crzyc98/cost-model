# Promotions and Compensation Adjustments

This document outlines how promotions and compensation adjustments are handled in the workforce simulation.

## Compensation Validation

### Handling Missing Compensation Data

As of May 2024, the promotion logic includes robust handling of missing compensation data:

1. **Pre-Filtering**: At the start of the promotion process, employees with missing `employee_gross_compensation` are identified and excluded from promotion consideration.

2. **Logging**: Detailed warnings are logged for employees with missing compensation, including:
   - Employee ID
   - Hire date
   - Total count of affected employees

3. **Safeguards**: 
   - The code includes multiple layers of validation to prevent processing employees with missing compensation
   - Even if an employee with missing compensation somehow passes initial checks, additional safeguards in the event creation logic will skip them

### Example Log Output

```
WARNING: Found 3 employees missing employee_gross_comp in promotion processing. These employees will be excluded from promotion consideration.
WARNING: Employee EMP123 (hired 2023-01-15) is missing compensation data; skipping promotion consideration.
WARNING: Employee EMP456 (hired 2023-03-22) is missing compensation data; skipping promotion consideration.
... and 1 more employees with missing compensation.
```

## Promotion Process

1. **Eligibility Check**: 
   - Only active employees with valid compensation are considered for promotions
   - Employees with missing or invalid compensation are excluded

2. **Markov Chain Model**:
   - Promotions are determined using a Markov chain model based on current level and tenure
   - Promotion probabilities are defined in the promotion matrix

3. **Compensation Adjustment**:
   - Successful promotions trigger compensation increases based on predefined rules
   - The raise percentage depends on the source and destination levels (e.g., 5% for 1→2, 8% for 2→3)

## Error Handling

The system includes comprehensive error handling:

- **Data Integrity**: Checks for data consistency before processing
- **Logging**: Detailed logs for debugging and auditing
- **Graceful Degradation**: If critical issues are detected, the system will skip affected employees rather than failing

## Testing

Unit tests verify the promotion logic, including handling of missing compensation data. See `tests/engines/test_markov_promotion.py` for details.

## Configuration

Promotion parameters can be configured in the simulation settings, including:

- Promotion matrix (transition probabilities)
- Raise percentages by level
- Eligibility rules

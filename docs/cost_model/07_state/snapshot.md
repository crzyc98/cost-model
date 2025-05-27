# Snapshot Management

## Employee Snapshot
- **Location**: `state.snapshot`
- **Description**: Represents the state of all employees at a specific point in time
- **Key Attributes**:
  - `employee_id`: Unique employee identifier (string)
  - `employee_hire_date`: Date employee was hired
  - `employee_birth_date`: Employee's date of birth
  - `employee_gross_compensation`: Annual compensation
  - `employee_termination_date`: Date employee was terminated (if applicable)
  - `active`: Boolean indicating if employee is currently active
  - `employee_deferral_rate`: Current retirement plan deferral rate
  - `employee_tenure`: Years of service (float)
  - `employee_tenure_band`: Categorical tenure band (e.g., "0-1", "1-3", "3-5", "5+")
  - `employee_level`: Job level/grade
  - `job_level_source`: Source of job level assignment
  - `exited`: Boolean indicating if employee has exited the company
  - `simulation_year`: Current simulation year
- **Search Tags**: `class:EmployeeSnapshot`, `state:snapshot`

## Snapshot Operations

### create_initial_snapshot()
- **Location**: `state.snapshot.create_initial_snapshot`
- **Description**: Creates the initial employee snapshot from census data
- **Parameters**:
  - `census_data`: DataFrame with employee census data
  - `simulation_year`: Starting simulation year
- **Returns**: Initial snapshot DataFrame with validated compensation data
- **Compensation Handling**:
  - Validates all compensation values in the input data
  - Replaces missing or invalid compensation with role-based defaults
  - Logs warnings for any compensation validation issues
- **Search Tags**: `function:create_initial_snapshot`, `state:initialization`, `compensation:validation`

### update_snapshot()
- **Location**: `state.snapshot.update_snapshot`
- **Description**: Updates the snapshot with new employee data
- **Compensation Validation**:
  - Validates compensation for all new and updated employee records
  - Ensures compensation values are within expected ranges for job levels
  - Applies default compensation when data is missing or invalid
  - Logs detailed information about any compensation adjustments
- **Search Tags**: `function:update_snapshot`, `state:update`, `compensation:validation`

### validate_compensation()
- **Location**: `state.snapshot.validate_compensation`
- **Description**: Validates and normalizes compensation data
- **Parameters**:
  - `snapshot`: The employee snapshot DataFrame to validate
  - `role_levels`: Optional mapping of job roles to expected compensation ranges
- **Returns**: Validated snapshot with normalized compensation values
- **Validation Rules**:
  - Checks for missing or null compensation values
  - Validates against role-based compensation ranges
  - Applies default values when necessary
  - Logs all validation issues for auditing

## Compensation Data Management

### Compensation Validation Process
1. **Initial Load**:
   - Validate all compensation values from source data
   - Replace missing values with role-based defaults
   - Log any data quality issues

2. **During Simulation**:
   - Validate compensation for new hires and promotions
   - Ensure compensation changes comply with business rules
   - Log all compensation adjustments

3. **Final Validation**:
   - Verify all employees have valid compensation
   - Check for outliers and anomalies
   - Generate validation report

## Snapshot Schema

```python
class EmployeeSnapshot(BaseModel):
    employee_id: str
    employee_hire_date: date
    employee_birth_date: date
    employee_gross_compensation: float
    employee_termination_date: Optional[date] = None
    active: bool = True
    employee_deferral_rate: float = 0.0
    employee_tenure: float
    employee_tenure_band: str
    employee_level: int
    job_level_source: str
    exited: bool = False
    simulation_year: int
```

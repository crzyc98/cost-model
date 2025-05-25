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
  - `employee_tenure_band`: Categorical tenure band (e.g., "<1", "1-3", "3-5", "5+")
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
- **Returns**: Initial snapshot DataFrame
- **Search Tags**: `function:create_initial_snapshot`, `state:initialization`

### update_snapshot()
- **Location**: `state.snapshot.update_snapshot`
- **Description**: Updates the snapshot with new employee data
- **Search Tags**: `function:update_snapshot`, `state:update`

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

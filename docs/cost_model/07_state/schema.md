# Data Schema

## Core Schemas

### Employee Schema
```python
class Employee(BaseModel):
    employee_id: str
    hire_date: date
    birth_date: date
    termination_date: Optional[date] = None
    active: bool
    compensation: float
    department: str
    job_level: int
    job_title: str
    manager_id: Optional[str] = None
    location: str
    status: Literal["active", "terminated", "on_leave"]
    created_at: datetime
    updated_at: datetime
```

### Department Schema
```python
class Department(BaseModel):
    department_id: str
    name: str
    cost_center: str
    manager_id: str
    headcount: int
    budget: float
    created_at: datetime
    updated_at: datetime
```

## Schema Validation

### validate_employee()
- **Location**: `state.schema.validate_employee`
- **Description**: Validates employee data against schema
- **Search Tags**: `function:validate_employee`, `schema:validation`

### validate_department()
- **Location**: `state.schema.validate_department`
- **Description**: Validates department data against schema
- **Search Tags**: `function:validate_department`, `schema:validation`

## Schema Migrations

### Migration Guide
1. **Version 1.0.0**: Initial schema
2. **Version 1.1.0**: Added `manager_id` to Employee
3. **Version 1.2.0**: Added `location` field
4. **Version 2.0.0**: Major schema refactor

### Migration Utilities
- `migrate_employee()`: Migrates employee records between versions
- `migrate_department()`: Migrates department records
- `get_schema_version()`: Gets current schema version

## Schema Import Guidelines

### Required Constants
All modules **must** import column names directly from `cost_model.state.schema`. Missing constants will raise an `ImportError` at startup.

**Example:**
```python
# Correct
from cost_model.state.schema import EMP_ID, SIMULATION_YEAR

# Incorrect - will raise ImportError
try:
    from .schema import EMP_ID, SIMULATION_YEAR
except ImportError:
    EMP_ID = "employee_id"  # Fallback not allowed
```

### Available Constants
- `EMP_ID`: Primary employee identifier column name
- `SIMULATION_YEAR`: Column name for simulation year tracking
- `EVENT_COLS`: List of standard event columns
- `SNAPSHOT_COLS`: List of columns in the employee snapshot
- `SNAPSHOT_DTYPES`: Pandas dtypes for snapshot columns

## Data Types

### Enums
```python
class EmploymentStatus(str, Enum):
    ACTIVE = "active"
    TERMINATED = "terminated"
    ON_LEAVE = "on_leave"

class JobLevel(int, Enum):
    ENTRY = 1
    INTERMEDIATE = 2
    SENIOR = 3
    MANAGER = 4
    DIRECTOR = 5
    EXECUTIVE = 6
```

### Custom Types
- `EmailStr`: Valid email address
- `PhoneNumber`: Standardized phone number
- `Currency`: Monetary amount with currency code
- `Percentage`: Float between 0 and 100

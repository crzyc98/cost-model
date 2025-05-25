# Census Data Parser

## Parser Functions

### parse_census_data
- **Location**: `cost_model.data.parsers.parse_census_data`
- **Description**: Parses raw census data into structured format
- **Parameters**:
  - `df`: Input DataFrame with raw census data
  - `config`: Configuration dictionary with parsing rules
- **Returns**: Processed DataFrame
- **Search Tags**: `function:parse_census_data`, `data:parsing`

### validate_census_data
- **Location**: `cost_model.data.validators.validate_census_data`
- **Description**: Validates census data against schema
- **Parameters**:
  - `df`: DataFrame to validate
  - `schema`: Expected schema definition
- **Raises**: `ValidationError` if data is invalid
- **Search Tags**: `function:validate_census_data`, `data:validation`

## Data Models

### EmployeeRecord
- **Location**: `cost_model.data.models.EmployeeRecord`
- **Description**: Represents an employee's census data
- **Attributes**:
  - `employee_id`: Unique employee identifier
  - `hire_date`: Date of hire
  - `termination_date`: Date of termination (if applicable)
  - `compensation`: Annual compensation
  - `department`: Department code
- **Search Tags**: `class:EmployeeRecord`, `model:employee`

### Department
- **Location**: `cost_model.data.models.Department`
- **Description**: Department information
- **Attributes**:
  - `code`: Department code
  - `name`: Department name
  - `cost_center`: Cost center code
- **Search Tags**: `class:Department`, `model:department`

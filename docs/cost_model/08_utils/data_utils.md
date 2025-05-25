# Data Processing Utilities

## Data Transformation

### DataFrame Helpers
- **Location**: `cost_model.utils.dataframe`
- **Description**: Pandas DataFrame utilities
- **Key Functions**:
  - `safe_merge()`: Merge with duplicate handling
  - `filter_by_dates()`: Filter DataFrame by date range
  - `add_missing_columns()`: Ensure consistent columns
  - `explode_dict()`: Expand dictionary columns
- **Search Tags**: `module:dataframe`, `utils:dataframes`

### Data Validation
- **Location**: `cost_model.utils.validation`
- **Description**: Data quality checks
- **Features**:
  - Schema validation
  - Data type checking
  - Constraint validation
  - Custom validators
- **Search Tags**: `module:validation`, `utils:validation`

## Data I/O

### File Operations
- **Location**: `cost_model.utils.file_io`
- **Description**: File handling utilities
- **Key Functions**:
  - `read_csv_safe()`: Read CSV with error handling
  - `write_parquet()`: Write DataFrame to Parquet
  - `read_config()`: Read YAML/JSON configs
- **Search Tags**: `module:file_io`, `utils:files`

### Database Helpers
- **Location**: `cost_model.utils.database`
- **Description**: Database utilities
- **Features**:
  - Connection pooling
  - Query building
  - Batch operations
  - Transaction management
- **Search Tags**: `module:database`, `utils:db`

## Usage Examples

### Data Transformation
```python
import pandas as pd
from cost_model.utils.dataframe import safe_merge, add_missing_columns

# Safe merge with duplicate handling
df1 = pd.DataFrame({'id': [1, 2], 'value': ['A', 'B']})
df2 = pd.DataFrame({'id': [2, 3], 'value': ['C', 'D']})
result = safe_merge(df1, df2, on='id', how='left')

# Ensure consistent columns
columns = ['id', 'name', 'salary']
df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
df = add_missing_columns(df, columns)  # Adds 'salary' with NaN
```

### Data Validation
```python
from pydantic import BaseModel
from cost_model.utils.validation import validate_dataframe

class EmployeeSchema(BaseModel):
    id: int
    name: str
    salary: float

# Validate DataFrame against schema
df = pd.DataFrame({
    'id': [1, 2],
    'name': ['Alice', 'Bob'],
    'salary': [75000, 85000]
})
validate_dataframe(df, EmployeeSchema)
```

### File Operations
```python
from pathlib import Path
from cost_model.utils.file_io import read_csv_safe, write_parquet

# Read CSV with error handling
try:
    df = read_csv_safe('data/employees.csv')
except FileNotFoundError:
    print("File not found")

# Write to Parquet
output_path = Path('output/employees.parquet')
write_parquet(df, output_path)
```

## Performance Tips

### Memory Management
- Use categoricals for low-cardinality strings
- Specify dtypes when reading files
- Process in chunks for large datasets

### Optimization
- Vectorize operations
- Avoid apply() with Python functions
- Use query() for filtering

## Related Documentation
- [Data Management](../04_data/index.md)
- [Configuration](../03_config/data_settings.md)

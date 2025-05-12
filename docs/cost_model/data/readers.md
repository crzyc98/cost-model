# Data Readers Module

The data readers module provides utilities for reading simulation data from various formats, including Parquet, CSV, and custom formats.

## QuickStart

To read simulation data from Parquet format:

```python
import pandas as pd
from cost_model.data.readers import read_parquet
from cost_model.utils.columns import EMP_ID, EMP_HIRE_DATE, EMP_ROLE

# Read a single Parquet file
employees = read_parquet(
    input_path='output/simulation/employees_2025.parquet',
    columns=['employee_id', 'employee_role', 'employee_hire_date'],
    filters=[('employee_role', '==', 'Engineer')]
)

# Read partitioned Parquet data
partitioned_data = read_parquet(
    input_path='output/simulation',
    columns=['employee_id', 'employee_role', 'employee_salary'],
    filters=[
        ('employee_role', '==', 'Engineer'),
        ('employee_salary', '>', 100000)
    ],
    partition_cols=['employee_role']
)

# Read multiple related files
simulation_data = read_multiple(
    input_dir='output/simulation',
    formats={
        'employees': 'parquet',
        'salaries': 'csv',
        'events': 'parquet'
    },
    base_filename='simulation_2025',
    metadata=True
)

# Examine the read data
print(f"Read {len(employees)} employees")
print("\nEmployee distribution by role:")
print(employees[EMP_ROLE].value_counts())
```

## Key Features

1. **File Formats**
   - Parquet (with partition support)
   - CSV (with configurable options)
   - Custom formats via extension points

2. **Partition Reading**
   - Support for reading specific partitions
   - Filter-based partition selection
   - Efficient querying of partitioned data

3. **Column Selection**
   - Select specific columns to read
   - Column name validation
   - Type conversion support

4. **Data Filtering**
   - Pre-filtering using column conditions
   - Support for multiple filter conditions
   - Efficient predicate pushdown

5. **Metadata Management**
   - Read file metadata
   - Track simulation parameters
   - Version information
   - Generation timestamp

## Configuration Options

The readers module can be configured with:

- `input_path`: Path to input file or directory
- `columns`: List of columns to read
- `filters`: List of filter conditions
- `partition_cols`: Columns to partition by
- `metadata`: Whether to read metadata
- `format`: Input format (parquet, csv, etc.)

## Partition Reading Strategies

1. **Role-based Partition Reading**
   ```python
   read_parquet(
       input_path='output/simulation',
       partition_cols=['employee_role'],
       filters=[('employee_role', '==', 'Engineer')]
   )
   ```

2. **Year-based Partition Reading**
   ```python
   read_parquet(
       input_path='output/simulation',
       partition_cols=['simulation_year'],
       filters=[('simulation_year', '==', 2025)]
   )
   ```

3. **Complex Partition Queries**
   ```python
   read_parquet(
       input_path='output/simulation',
       partition_cols=['year', 'quarter', 'region'],
       filters=[
           ('year', '>=', 2025),
           ('region', 'in', ['North', 'South'])
       ]
   )
   ```

## Error Handling

The readers module includes several safety checks:

1. File existence and accessibility
2. Format validation
3. Column existence validation
4. Partition structure validation
5. Filter condition validation
6. Data type validation

## Performance Considerations

- Uses efficient predicate pushdown for filtering
- Implements columnar reads for large datasets
- Supports parallel reading where possible
- Optimizes memory usage for large files
- Handles partitioned data efficiently
- Caches metadata to avoid redundant reads

## Example Scenarios

### Scenario 1: Basic Parquet Reading
```python
# Read a simple DataFrame from Parquet
employees = read_parquet(
    input_path='output/simulation/employees_2025.parquet'
)
```

### Scenario 2: Efficient Partition Reading
```python
# Read specific partitions with filtering
engineers = read_parquet(
    input_path='output/simulation',
    partition_cols=['employee_role', 'location'],
    filters=[
        ('employee_role', '==', 'Engineer'),
        ('location', 'in', ['San Francisco', 'New York'])
    ]
)
```

### Scenario 3: Multiple Files with Metadata
```python
# Read multiple related files with their metadata
simulation_data = read_multiple(
    input_dir='output/simulation',
    formats={
        'employees': 'parquet',
        'salaries': 'csv',
        'events': 'parquet'
    },
    base_filename='simulation_2025',
    metadata=True
)
```

## Troubleshooting

If you're having issues with data reading:

1. Verify input path exists and is readable:
   ```python
   assert Path(input_path).exists()
   ```

2. Check partition structure matches your data:
   ```python
   assert all(col in df.columns for col in partition_cols)
   ```

3. Validate filter conditions:
   ```python
   assert all(col in df.columns for col, op, val in filters)
   ```

4. Review the debug logging for detailed error information
5. Check file permissions on the input directory
6. Verify format is supported

## Output Format Details

The readers module supports multiple input formats:

1. **Parquet Format**
   - Columnar storage format
   - Efficient compression
   - Partitioning support
   - Metadata tracking

2. **CSV Format**
   - Human-readable format
   - Configurable delimiters
   - Header options
   - Quoting options

3. **Custom Formats**
   - Extensible via reader plugins
   - Custom deserialization
   - Format-specific optimizations

## Best Practices

1. Use partitioning for efficient querying:
   - Partition by frequently filtered columns
   - Keep partitions at a reasonable size
   - Use meaningful partition keys

2. Select only needed columns:
   - Avoid reading unnecessary columns
   - Use column selection for memory efficiency
   - Validate column names before reading

3. Use filters effectively:
   - Push filters as early as possible
   - Use appropriate filter operators
   - Combine filters for efficiency

4. Handle metadata consistently:
   - Track simulation parameters
   - Maintain version information
   - Record generation timestamps

5. Monitor performance:
   - Track memory usage
   - Profile read operations
   - Optimize for your use case

6. Use appropriate file formats:
   - Parquet for large datasets
   - CSV for human-readable data
   - Custom formats for special cases

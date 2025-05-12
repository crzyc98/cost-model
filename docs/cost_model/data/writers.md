# Data Writers Module

The data writers module provides utilities for writing simulation data to various formats, including Parquet, CSV, and custom formats.

## QuickStart

To write simulation data to Parquet format:

```python
import pandas as pd
from cost_model.data.writers import write_parquet
from cost_model.utils.columns import EMP_ID, EMP_HIRE_DATE, EMP_ROLE

# Create a sample DataFrame with employee data
employees = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003'],
    'employee_hire_date': [
        pd.Timestamp('2023-01-15'),
        pd.Timestamp('2024-06-01'),
        pd.Timestamp('2025-01-01')
    ],
    'employee_role': ['Engineer', 'Manager', 'Analyst']
}).set_index('employee_id')

# Write to Parquet
write_parquet(
    df=employees,
    output_dir='output/simulation',
    filename='employees_2025.parquet',
    compression='snappy',
    partition_cols=['employee_role']
)

# Write multiple files with metadata
write_multiple(
    dfs={
        'employees': employees,
        'salaries': salaries_df,
        'events': events_df
    },
    output_dir='output/simulation',
    base_filename='simulation_2025',
    format='parquet',
    metadata={
        'simulation_year': 2025,
        'version': '1.0.0',
        'generated_at': pd.Timestamp.now()
    }
)
```

## Key Features

1. **File Formats**
   - Parquet (with compression options)
   - CSV (with configurable options)
   - Custom formats via extension points

2. **Partitioning**
   - Support for partitioning by columns
   - Automatic directory structure creation
   - Efficient querying of partitioned data

3. **Compression**
   - Snappy (default)
   - GZIP
   - Brotli
   - ZSTD
   - None

4. **Metadata Management**
   - Custom metadata support
   - Timestamp tracking
   - Version management
   - Simulation parameters

## Configuration Options

The writers module can be configured with:

- `output_dir`: Base directory for output files
- `filename`: Name of the output file
- `compression`: Compression algorithm
- `partition_cols`: Columns to partition by
- `metadata`: Custom metadata dictionary
- `format`: Output format (parquet, csv, etc.)

## Partitioning Strategies

1. **Role-based Partitioning**
   ```python
   write_parquet(
       df=employees,
       output_dir='output/simulation',
       filename='employees.parquet',
       partition_cols=['employee_role']
   )
   ```

2. **Year-based Partitioning**
   ```python
   write_parquet(
       df=simulation_results,
       output_dir='output/simulation',
       filename='results.parquet',
       partition_cols=['simulation_year']
   )
   ```

3. **Multiple Partition Levels**
   ```python
   write_parquet(
       df=simulation_results,
       output_dir='output/simulation',
       filename='results.parquet',
       partition_cols=['simulation_year', 'employee_role', 'location']
   )
   ```

## Error Handling

The writers module includes several safety checks:

1. Directory creation and validation
2. File existence checks
3. Data type validation
4. Partition column validation
5. Compression format validation
6. Metadata schema validation

## Performance Considerations

- Uses efficient batch writing for large datasets
- Implements parallel writing where possible
- Optimizes compression settings for different use cases
- Handles large partitions efficiently
- Caches metadata to avoid redundant calculations

## Example Scenarios

### Scenario 1: Basic Parquet Writing
```python
# Write simple DataFrame to Parquet
write_parquet(
    df=snapshot,
    output_dir='output/simulation',
    filename='snapshot_2025.parquet'
)
```

### Scenario 2: Complex Partitioning
```python
# Write with multiple partitions and compression
write_parquet(
    df=simulation_results,
    output_dir='output/simulation',
    filename='results.parquet',
    partition_cols=['year', 'quarter', 'region'],
    compression='snappy'
)
```

### Scenario 3: Multiple Files with Metadata
```python
# Write multiple related files with common metadata
write_multiple(
    dfs={
        'employees': employees_df,
        'salaries': salaries_df,
        'events': events_df
    },
    output_dir='output/simulation',
    base_filename='simulation_2025',
    format='parquet',
    metadata={
        'simulation_year': 2025,
        'version': '1.0.0',
        'parameters': {
            'termination_rate': 0.05,
            'hiring_rate': 0.10,
            'salary_growth': 0.03
        }
    }
)
```

## Troubleshooting

If you're having issues with data writing:

1. Verify output directory exists and is writable:
   ```python
   assert Path(output_dir).exists()
   ```

2. Check partition columns exist in your DataFrame:
   ```python
   assert all(col in df.columns for col in partition_cols)
   ```

3. Validate data types before writing:
   ```python
   assert df.dtypes.apply(lambda x: x in ['int64', 'float64', 'object', 'datetime64[ns]']).all()
   ```

4. Review the debug logging for detailed error information
5. Check file permissions on the output directory
6. Verify compression format is supported

## Output Format Details

The writers module supports multiple output formats:

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
   - Extensible via writer plugins
   - Custom serialization
   - Format-specific optimizations

## Best Practices

1. Use partitioning for large datasets
2. Choose appropriate compression based on use case:
   - Snappy for fast reads
   - GZIP for better compression
   - None for quick writes

3. Include metadata for tracking:
   - Simulation parameters
   - Generation timestamp
   - Version information

4. Validate data before writing
5. Use consistent naming conventions
6. Regularly clean up old simulation data
7. Monitor disk space usage
8. Use appropriate file permissions

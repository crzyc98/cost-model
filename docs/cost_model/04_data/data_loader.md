# Data Loading and Processing

## Error Classes

### DataReadError
- **Location**: `cost_model.data.readers.DataReadError`
- **Description**: Raised when there's an error reading data files
- **Inherits from**: `Exception`
- **Key Attributes**:
  - `message`: Error description
  - `file_path`: Path to the file being read
- **Search Tags**: `exception:DataReadError`, `data:io`

### DataWriteError
- **Location**: `cost_model.data.writers.DataWriteError`
- **Description**: Raised when there's an error writing data files
- **Inherits from**: `Exception`
- **Key Attributes**:
  - `message`: Error description
  - `file_path`: Path to the file being written
- **Search Tags**: `exception:DataWriteError`, `data:io`

## Reader Functions

### read_census_data
- **Location**: `cost_model.data.readers.read_census_data`
- **Description**: Reads census data from CSV or Parquet files
- **Parameters**:
  - `file_path`: Path to the input file (CSV or Parquet)
  - `date_columns`: List of date columns to parse
  - `dtype`: Data types for columns
- **Returns**: `pandas.DataFrame` with the loaded data
- **Search Tags**: `function:read_census_data`, `data:io`

### read_config
- **Location**: `cost_model.data.readers.read_config`
- **Description**: Loads configuration from YAML file
- **Parameters**:
  - `config_path`: Path to the YAML configuration file
- **Returns**: Dictionary with configuration
- **Search Tags**: `function:read_config`, `config:io`

## Writer Functions

### write_snapshots
- **Location**: `cost_model.data.writers.write_snapshots`
- **Description**: Writes simulation snapshots to disk
- **Parameters**:
  - `snapshots`: List of snapshot DataFrames
  - `output_dir`: Directory to write output files
- **Search Tags**: `function:write_snapshots`, `data:io`

### write_event_logs
- **Location**: `cost_model.data.writers.write_event_logs`
- **Description**: Writes event logs to disk
- **Parameters**:
  - `event_logs`: List of event log DataFrames
  - `output_dir`: Directory to write output files
- **Search Tags**: `function:write_event_logs`, `data:io`

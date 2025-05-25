# Configuration Reference

## Scenario Configuration

### ScenarioDefinition
- **Location**: `config/scenarios.py`
- **Purpose**: Defines simulation scenarios
- **Key Attributes**:
  - `scenario_id`: Unique identifier
  - `description`: Scenario description
  - `start_date`: Simulation start date
  - `end_date`: Simulation end date
  - `assumptions`: Economic assumptions
  - `employee_data`: Initial employee population
- **Search Tags**: `configuration`, `scenarios`, `simulation`

### EconomicAssumptions
- **Location**: `config/assumptions.py`
- **Purpose**: Economic parameters
- **Key Attributes**:
  - `inflation_rate`: Annual inflation rate
  - `salary_increase`: Base salary increase
  - `promotion_increase`: Promotion salary bump
  - `turnover_rate`: Base turnover rate
  - `hiring_rate`: New hire rate
- **Search Tags**: `configuration`, `assumptions`, `economics`

## Employee Configuration

### CompensationStructure
- **Location**: `config/compensation.py`
- **Purpose**: Defines pay structure
- **Key Attributes**:
  - `salary_bands`: Pay ranges by level
  - `bonus_targets`: Target bonus percentages
  - `equity_grants`: Equity grant guidelines
  - `merit_matrix`: Merit increase guidelines
- **Search Tags**: `configuration`, `compensation`, `pay`

### BenefitsPackage
- **Location**: `config/benefits.py`
- **Purpose**: Benefits configuration
- **Key Attributes**:
  - `health_plans`: Available health plans
  - `retirement_plans`: 401(k) details
  - `time_off`: PTO policies
  - `insurance`: Life/Disability options
- **Search Tags**: `configuration`, `benefits`, `compensation`

## Simulation Parameters

### RunSettings
- **Location**: `config/settings.py`
- **Purpose**: Runtime settings
- **Key Attributes**:
  - `random_seed`: Random number seed
  - `num_simulations`: Number of runs
  - `output_dir`: Results directory
  - `log_level`: Logging verbosity
- **Search Tags**: `configuration`, `settings`, `runtime`

### PerformanceTuning
- **Location**: `config/performance.py`
- **Purpose**: Performance settings
- **Key Attributes**:
  - `batch_size`: Processing batch size
  - `max_workers`: Thread/process count
  - `cache_size`: Cache size in MB
  - `memory_limit`: Memory limit in MB
- **Search Tags**: `configuration`, `performance`, `tuning`

## Configuration Examples

### Basic Scenario
```yaml
# basic_scenario.yaml
scenario_id: "baseline_2025"
description: "Baseline scenario for 2025 planning"
start_date: "2025-01-01"
end_date: "2030-12-31"

assumptions:
  inflation_rate: 0.025
  salary_increase: 0.03
  promotion_increase: 0.08
  turnover_rate: 0.15
  hiring_rate: 0.10

compensation:
  salary_bands:
    - level: 1
      min: 50000
      mid: 60000
      max: 70000
    - level: 2
      min: 70000
      mid: 85000
      max: 100000

benefits:
  retirement_plan:
    employer_match: 0.50
    match_up_to: 0.06
    auto_enroll: true
    default_contribution: 0.04
```

### Advanced Configuration
```yaml
# advanced_config.yaml
performance:
  batch_size: 1000
  max_workers: 8
  cache_size: 4096
  memory_limit: 16384

logging:
  level: "INFO"
  file: "simulation.log"
  max_size: 10485760  # 10MB
  backup_count: 5

output:
  directory: "results"
  formats: ["csv", "parquet"]
  include_sensitive: false
  compress: true
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG_PATH` | Path to config directory | `./config` |
| `DATA_PATH` | Path to data directory | `./data` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DEBUG` | Enable debug mode | `false` |
| `RANDOM_SEED` | Random seed | `42` |

## Configuration Precedence

1. Command-line arguments
2. Environment variables
3. Configuration files
4. Default values

## Validation Rules

### Data Types
- **Dates**: Must be in YYYY-MM-DD format
- **Percentages**: Must be between 0 and 1
- **Currencies**: Must be positive numbers
- **IDs**: Must be unique within their context

### Required Fields
- Scenario ID
- Start date
- End date
- At least one employee record
- Economic assumptions

## Versioning

### Configuration Version
- **Current Version**: 1.0.0
- **Backward Compatibility**: Maintained within major versions
- **Upgrade Path**: Migration scripts available for major version changes

### Deprecation Policy
- Fields marked as deprecated will be removed in the next major version
- Warnings will be logged for deprecated features
- Migration guides will be provided for breaking changes

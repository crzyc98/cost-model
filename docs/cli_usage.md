# Cost Model Projection CLI

## Overview

The Cost Model Projection CLI provides a command-line interface for running multi-year retirement plan cost projections. It processes a census file according to a configuration file, simulates employee behavior and plan rules over multiple years, and outputs detailed results including snapshots, event logs, and summary statistics.

## Usage

```
python -m cost_model.projections.cli [options]
```

### Options

| Option | Description | Required | Default |
|--------|-------------|----------|---------|
| `--config` | Path to the YAML configuration file | Yes | - |
| `--census` | Path to the Parquet census data file | Yes | - |
| `--output-dir` | Directory to save output files (overrides config if provided) | No | From config or `output_dev/{scenario_name}_results` |
| `--scenario-name` | Name for the scenario, used in output file naming | No | `projection_cli` |

### Output

The CLI generates the following outputs in the specified directory:

- Yearly snapshots of the workforce
- Final end-of-year snapshot
- Cumulative event log
- Summary statistics
- Employment status summary
- Configuration used for the run
- Visualizations of projection results

All outputs are saved in Parquet format for efficient storage and fast loading.

### Logging

Logs are written to:
- Console (standard output)
- `output_dev/projection_logs/projection_cli_run.log`

The log level can be configured in the global parameters section of the config file using the `log_level` parameter (default: INFO).

## Examples

Basic usage with default output location:

```bash
python -m cost_model.projections.cli --config config/scenarios.yaml --census data/census_2025.parquet
```

Specifying a custom output directory and scenario name:

```bash
python -m cost_model.projections.cli --config config/scenarios.yaml --census data/census_2025.parquet --output-dir results/scenario_a --scenario-name baseline_2025
```

Running from a script with a specific configuration:

```bash
python -m cost_model.projections.cli --config config/high_growth.yaml --census data/census_2025.parquet --scenario-name high_growth_scenario
```

## Configuration File

The configuration file should be a YAML file with the following structure:

```yaml
global_parameters:
  start_year: 2025
  projection_years: 5
  random_seed: 42
  log_level: INFO
  output_directory: output/results
  # Other global parameters...

plan_rules:
  eligibility:
    minimum_age: 21
    service_requirement_months: 3
  employer_match:
    tiers:
      - employee_contribution_min: 0.0
        employee_contribution_max: 6.0
        employer_match_rate: 0.5
    cap: 3.0
  auto_enrollment:
    enabled: true
    default_rate: 3.0
  # Other plan rules...
```

See the configuration documentation for more details on available parameters.

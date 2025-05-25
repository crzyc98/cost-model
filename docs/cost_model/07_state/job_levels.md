# Job Levels & Promotions

## Overview
The `job_levels` module manages job level definitions, compensation bands, and promotion logic using a Markov chain model.

## Key Components

### 1. Level Definition
- **Location**: `state.job_levels.models.JobLevel`
- **Purpose**: Represents a job level with compensation ranges
- **Key Attributes**:
  - `level_id`: Unique identifier (e.g., "L1", "L2")
  - `min_comp`: Minimum compensation
  - `mid_comp`: Midpoint compensation
  - `max_comp`: Maximum compensation
  - `promotion_targets`: Possible promotion targets

### 2. Promotion Logic
- **Location**: `state.job_levels.sampling`
- **Features**:
  - Markov chain-based promotion modeling
  - Vectorized compensation sampling
  - Support for default and custom promotion matrices

### 3. Configuration
- **File Format**: YAML
- **Required Fields**:
  ```yaml
  levels:
    - id: L1
      min_comp: 50000
      mid_comp: 60000
      max_comp: 70000
    - id: L2
      min_comp: 70000
      mid_comp: 85000
      max_comp: 100000
  ```

## Usage Examples

### Loading Job Levels
```python
from cost_model.state.job_levels import load_job_levels_from_config

levels = load_job_levels_from_config("path/to/levels.yaml")
```

### Sampling Promotions
```python
from cost_model.state.job_levels.sampling import apply_promotion_markov

updated_df = apply_promotion_markov(
    employee_df,
    level_col="employee_level",
    simulation_year=2025
)
```

## Related Components
- [State Management](index.md) - Core state documentation
- [Dynamics & Engines](../05_dynamics_engines.md) - How promotions integrate with simulation

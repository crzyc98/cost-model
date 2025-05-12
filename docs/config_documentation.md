# Configuration File Documentation

This document provides a comprehensive explanation of the configuration parameters used in the retirement plan cost model simulation.

## Overview

The configuration file is structured into three main sections:
1. **global_parameters**: General simulation settings
2. **plan_rules**: Retirement plan specific rules and parameters
3. **scenarios**: Defined scenarios for running simulations

## Global Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `start_year` | The starting year for the simulation | `2025` |
| `log_level` | Logging verbosity level | `DEBUG` |
| `projection_years` | Number of years to project forward | `5` |
| `random_seed` | Seed for random number generation to ensure reproducibility | `42` |
| `annual_compensation_increase_rate` | Annual rate at which employee compensation increases | `0.05` (5%) |
| `annual_termination_rate` | Base annual employee termination rate | `0.15` (15%) |
| `new_hire_termination_rate` | Termination rate specific to new hires | `0.25` (25%) |
| `new_hire_rate` | Rate at which new employees are hired | `0.17` (17%) |
| `use_expected_attrition` | Whether to use expected attrition rates | `false` |
| `days_into_year_for_cola` | Day of year when cost of living adjustments occur | `182` |
| `days_into_year_for_promotion` | Day of year when promotions occur | `182` |
| `new_hire_average_age` | Average age of new hires | `30` |
| `census_template_path` | Path to the census template file | `"data/census_template.parquet"` |
| `deterministic_termination` | Whether termination is deterministic | `false` |
| `monthly_transition` | Whether to use monthly transitions | `false` |
| `maintain_headcount` | Whether to maintain constant headcount | `false` |

### Role Distribution
Defines the distribution of employee roles:
```yaml
role_distribution:
  Staff: 1.0  # 100% of employees are Staff
```

### New Hire Compensation Parameters
Parameters that define compensation for new hires:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `comp_base_salary` | Base salary for new hires | `55000` |
| `comp_std` | Standard deviation for salary distribution | `10000` |
| `comp_increase_per_age_year` | Annual salary increase per year of age | `500` |
| `comp_increase_per_tenure_year` | Annual salary increase per year of tenure | `1000` |
| `comp_log_mean_factor` | Factor for log-normal salary distribution | `1.0` |
| `comp_spread_sigma` | Spread parameter for salary distribution | `0.3` |
| `comp_min_salary` | Minimum salary | `40000` |

### Role Compensation Parameters
Similar parameters defined per role (e.g., Staff).

### Onboarding Parameters
Parameters related to employee onboarding:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `enabled` | Whether onboarding features are enabled | `true` |
| `early_tenure_months` | Number of months considered early tenure | `6` |
| `hazard_multiplier` | Multiplier for termination risk during early tenure | `1.5` |
| `productivity_curve` | Productivity levels over time | `[0.80, 0.85, 0.90, 0.95, 1.00, 1.00]` |

## Plan Rules

### Eligibility
Defines when employees become eligible for the retirement plan:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `min_age` | Minimum age for plan eligibility | `21` |
| `min_service_months` | Minimum months of service for eligibility | `0` |

### Onboarding Bump
Parameters for compensation adjustments during onboarding:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `enabled` | Whether onboarding bumps are enabled | `true` |
| `method` | Method for calculating onboarding bumps | `sample_plus_rate` |
| `rate` | Rate for onboarding bump | `0.05` (5%) |

### Auto Enrollment
Parameters for automatic enrollment in the retirement plan:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `enabled` | Whether auto enrollment is enabled | `false` |
| `window_days` | Number of days in the enrollment window | `90` |
| `proactive_enrollment_probability` | Probability of proactive enrollment | `0.0` |
| `proactive_rate_range` | Range for proactive enrollment rates | `[0.0, 0.0]` |
| `default_rate` | Default contribution rate | `0.0` |
| `re_enroll_existing` | Whether to re-enroll existing employees | `false` |
| `opt_down_target_rate` | Target rate for opt-down | `0.0` |
| `increase_to_match_rate` | Rate to increase to match | `0.0` |
| `increase_high_rate` | High increase rate | `0.0` |

#### Auto Enrollment Outcome Distribution
Probabilities for different enrollment outcomes:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `prob_opt_out` | Probability of opting out | `0.0` |
| `prob_stay_default` | Probability of staying at default rate | `1.0` |
| `prob_opt_down` | Probability of opting down | `0.0` |
| `prob_increase_to_match` | Probability of increasing to match | `0.0` |
| `prob_increase_high` | Probability of high increase | `0.0` |

### Auto Increase
Parameters for automatic contribution increases:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `enabled` | Whether auto increase is enabled | `false` |
| `increase_rate` | Rate of automatic increase | `0.0` |
| `cap_rate` | Maximum rate cap | `0.0` |

### Employer Match
Parameters for employer matching contributions:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `tiers[].match_rate` | Rate at which employer matches | `0.0` |
| `tiers[].cap_deferral_pct` | Cap on deferral percentage | `0.0` |
| `dollar_cap` | Dollar cap on employer match | `0.0` |

### Employer NEC (Non-Elective Contribution)
Parameters for employer non-elective contributions:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `rate` | Rate of non-elective contribution | `0.01` (1%) |

### IRS Limits
Annual IRS limits for retirement contributions:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `compensation_limit` | Maximum compensation considered | `345000` |
| `deferral_limit` | Maximum deferral amount | `23000` |
| `catchup_limit` | Additional catch-up contribution limit | `7500` |
| `catchup_eligibility_age` | Age at which catch-up contributions are allowed | `50` |

### Behavioral Parameters
Parameters modeling employee behavior:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `voluntary_enrollment_rate` | Rate of voluntary enrollment | `0.20` (20%) |
| `voluntary_default_deferral` | Default deferral rate for voluntary enrollment | `0.05` (5%) |
| `voluntary_window_days` | Window for voluntary enrollment | `180` |
| `voluntary_change_probability` | Probability of changing contribution | `0.10` (10%) |
| `prob_increase_given_change` | Probability of increasing contribution | `0.40` (40%) |
| `prob_decrease_given_change` | Probability of decreasing contribution | `0.30` (30%) |
| `prob_stop_given_change` | Probability of stopping contribution | `0.05` (5%) |
| `voluntary_increase_amount` | Amount of voluntary increase | `0.01` (1%) |
| `voluntary_decrease_amount` | Amount of voluntary decrease | `0.01` (1%) |

### Contributions
Parameters for contribution calculations:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `enabled` | Whether contributions are enabled | `true` |

### Eligibility Events
Parameters for eligibility milestone events:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `milestone_months` | Milestone months for eligibility events | `[]` |
| `milestone_years` | Milestone years for eligibility events | `[]` |
| `event_type_map` | Mapping of event types | `{}` |

## Scenarios
Defined scenarios for running simulations:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `baseline.name` | Name of the baseline scenario | `TinyDev` |

Each scenario can override specific parameters from the global configuration.

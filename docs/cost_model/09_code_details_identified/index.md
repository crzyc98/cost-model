# Detailed Class Documentation

This document provides comprehensive documentation for all classes in the codebase, organized by module.

## Logging System Update (Phase 2)

- **Modules Affected**: 
  - `projections/cli.py`
  - `logging_config.py` 
  - `engines/run_one_year.py`
  - `state/snapshot.py`

- **Purpose**: 
  - Improve log clarity and consistency across the application
  - Enable per-concern traceability through separate log files
  - Ensure compatibility with debugging and production workflows
  - Implement proper log rotation and retention

- **Changes Made**:
  - **Root Logger Configuration**:
    - Properly configured to prevent interference with custom loggers
    - Console handler added for warnings and above
    - Global `_LOGGING_CONFIGURED` flag to prevent duplicate initialization
  
  **Log Files and Their Purposes**:
  - `projection_events.log` (INFO+): Main projection workflow events
  - `performance_metrics.log` (INFO+): Performance-related metrics and timings
  - `warnings_errors.log` (WARNING+): All warnings and errors
  - `debug_detail.log` (DEBUG, only if debug=True): Detailed debug information

  **Key Improvements**:
  - Added log rotation (10MB per file, 5 backups)
  - Included module name and line numbers in log format
  - Prevented duplicate log messages through proper logger hierarchy
  - Added thread safety for logging configuration

- **Test Coverage**:
  - Added `test_logging()` function to `logging_config.py`
  - Created `test_logging.py` for manual verification
  - Verified log file creation and content
  - Tested exception logging and formatting

- **Usage Example**:
  ```python
  # In any module
  from logging_config import get_logger
  
  logger = get_logger(__name__)
  logger.info("Informational message")
  logger.warning("Warning message")
  logger.error("Error message")
  try:
      1 / 0
  except Exception as e:
      logger.exception("Exception occurred")
  ```

- **Search Tags**: 
  - `module:logging_config`
  - `file:projection_events.log` 
  - `handler:debug_detail` 
  - `class:Logger`
  - `phase:2`

---

# Detailed Class Documentation

This document provides comprehensive documentation for all classes in the codebase, organized by module.

## Configuration Module

### MainConfig

- **Purpose**: Serves as the root model for the entire configuration file, containing global parameters and scenario definitions.
- **Key Methods**:
  - `check_baseline_scenario_exists()`: Validates that at least a 'baseline' scenario is defined.
- **Key Attributes**:
  - `global_parameters`: Global simulation parameters
  - `scenarios`: Dictionary of scenario definitions
- **Used In**: Configuration loading and validation throughout the application
- **Search Tags**: class:MainConfig, method:check_baseline_scenario_exists, attr:global_parameters, attr:scenarios, module:config.models

### GlobalParameters

- **Purpose**: Contains all global simulation parameters that apply across scenarios unless overridden.
- **Key Methods**:
  - `validate_global_parameters()`: Runs all validations
  - `_check_role_dist_sums_to_one()`: Validates role distribution probabilities
  - `_check_maintain_headcount_vs_growth()`: Validates headcount vs growth settings
- **Key Attributes**:
  - `start_year`, `projection_years`, `random_seed`
  - `annual_compensation_increase_rate`, `annual_termination_rate`
  - `new_hire_termination_rate`, `use_expected_attrition`
  - `plan_rules`: Container for all plan-specific rules
- **Used In**: Simulation initialization and configuration
- **Search Tags**: class:GlobalParameters, method:validate_global_parameters, attr:plan_rules, module:config.models

### ScenarioDefinition

- **Purpose**: Defines a single simulation scenario that can override global parameters.
- **Key Methods**: None (primarily a data container)
- **Key Attributes**:
  - `name`: Scenario identifier
  - Parameters that can override global settings (e.g., `start_year`, `projection_years`)
  - `plan_rules`: Optional scenario-specific plan rules
- **Used In**: Scenario management and configuration
- **Search Tags**: class:ScenarioDefinition, attr:name, attr:plan_rules, module:config.models

### AutoEnrollOutcomeDistribution

- **Purpose**: Defines the probability distribution for different auto-enrollment outcomes.
- **Key Methods**:
  - `check_probabilities_sum_to_one()`: Validates that probabilities sum to 1.0
- **Key Attributes**:
  - `prob_opt_out`, `prob_stay_default`, `prob_opt_down`
  - `prob_increase_to_match`, `prob_increase_high`
- **Used In**: Auto-enrollment simulation logic
- **Search Tags**: class:AutoEnrollOutcomeDistribution, method:check_probabilities_sum_to_one, attr:prob_opt_out, module:config.models

### OnboardingBumpRules

- **Purpose**: Configures compensation bumps for new hires during their onboarding period.
- **Key Methods**:
  - `check_method_and_rate()`: Validates that method and rate are properly configured when enabled
- **Key Attributes**:
  - `enabled`: Whether onboarding bumps are active
  - `method`: Bump calculation method ('flat_rate' or 'sample_plus_rate')
  - `rate`: Bump rate applied to compensation
- **Used In**: Compensation adjustment logic for new hires
- **Search Tags**: class:OnboardingBumpRules, method:check_method_and_rate, attr:enabled, attr:method, attr:rate, module:config.models

### EmployerMatchRules

- **Purpose**: Defines employer matching contribution rules for retirement plans.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `tiers`: List of `MatchTier` objects defining matching structure
  - `dollar_cap`: Optional annual dollar cap on total match
- **Used In**: Contribution calculations during simulation
- **Search Tags**: class:EmployerMatchRules, attr:tiers, attr:dollar_cap, module:config.models

### EmployerNecRules

- **Purpose**: Configures non-elective employer contributions.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `rate`: Non-elective contribution rate as percentage (e.g., 0.03 for 3%)
- **Used In**: Contribution calculations
- **Search Tags**: class:EmployerNecRules, attr:rate, module:config.models

### BehavioralParams

- **Purpose**: Models employee behavioral parameters for plan participation.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `voluntary_enrollment_rate`: Base rate of voluntary enrollment
  - `voluntary_default_deferral`: Default deferral rate for voluntary enrollment
  - `voluntary_window_days`: Enrollment window in days
  - `voluntary_change_probability`: Probability of voluntary contribution change
  - `prob_increase_given_change`: Probability of increase given change
  - `prob_decrease_given_change`: Probability of decrease given change
  - `prob_stop_given_change`: Probability of stopping contributions
  - `voluntary_increase_amount`: Amount to increase contributions by
  - `voluntary_decrease_amount`: Amount to decrease contributions by
- **Used In**: Enrollment and contribution behavior simulation
- **Search Tags**: class:BehavioralParams, module:config.models

### HazardModelParams

- **Purpose**: References external hazard model parameters for simulation.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `file`: Path to the hazard model parameters file
- **Used In**: Loading hazard model parameters for simulation
- **Search Tags**: class:HazardModelParams, attr:file, module:config.models

### CompensationParams

- **Purpose**: Defines generic compensation parameters for employees.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `comp_base_salary`: Base salary amount
  - `comp_std`: Standard deviation for salary distribution
  - `comp_increase_per_age_year`: Annual increase per year of age
  - `comp_increase_per_tenure_year`: Annual increase per year of tenure
  - `comp_log_mean_factor`: Log mean factor (default: 1.0)
  - `comp_spread_sigma`: Salary spread sigma (default: 0.3)
  - `comp_min_salary`: Minimum salary threshold
- **Used In**: Compensation calculations and projections
- **Search Tags**: class:CompensationParams, module:config.models

### ContributionConfig

- **Purpose**: Defines default contribution settings for retirement plans.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `default_deferral_rate`: Default employee deferral rate (default: 0.06)
  - `max_deferral_rate`: Maximum allowed deferral rate (default: 0.15)
  - `employer_match_pct`: Employer match percentage (default: 0.5)
- **Used In**: Setting baseline contribution parameters for the plan
- **Search Tags**: class:ContributionConfig, attr:default_deferral_rate, attr:max_deferral_rate, attr:employer_match_pct, module:config.plan_rules

### ContributionRules

- **Purpose**: Defines rules for contribution calculations beyond basic match/NEC.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `enabled`: Whether contributions are calculated (default: True)
- **Used In**: Controlling contribution calculation behavior
- **Search Tags**: class:ContributionRules, attr:enabled, module:config.models

### AutoEnrollmentConfig

- **Purpose**: Configures automatic enrollment settings for retirement plans.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `enabled`: Whether auto-enrollment is active (default: False)
  - `default_rate`: Default contribution rate (default: 0.03)
  - `window_days`: Auto-enrollment window in days (default: 90)
- **Used In**: Auto-enrollment simulation logic
- **Search Tags**: class:AutoEnrollmentConfig, attr:enabled, attr:default_rate, attr:window_days, module:config.plan_rules

### AutoIncreaseConfig

- **Purpose**: Configures automatic contribution rate increases.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `increase_pct`: Percentage increase (default: 0.01)
  - `cap`: Maximum contribution rate (default: 0.10)
  - `frequency_years`: Years between increases (default: 1)
- **Used In**: Automatic contribution increase simulation
- **Search Tags**: class:AutoIncreaseConfig, attr:increase_pct, attr:cap, attr:frequency_years, module:config.plan_rules

### ProactiveDecreaseConfig

- **Purpose**: Configures proactive contribution rate decreases based on historical patterns.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `lookback_months`: Lookback period in months (default: 12)
  - `threshold_pct`: Decrease threshold percentage (default: 0.05)
  - `event_type`: Event type identifier (default: "EVT_PROACTIVE_DECREASE")
- **Used In**: Proactive decrease simulation logic
- **Search Tags**: class:ProactiveDecreaseConfig, attr:lookback_months, attr:threshold_pct, attr:event_type, module:config.plan_rules

### ContributionRules

- **Purpose**: Placeholder for contribution-specific rules beyond match/NEC.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `enabled`: Whether contributions are enabled
- **Used In**: Contribution processing
- **Search Tags**: class:ContributionRules, attr:enabled, module:config.models

### PlanRules

- **Purpose**: Container for all plan rule configurations.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `eligibility`: Eligibility rules
  - `onboarding_bump`: Onboarding compensation bump rules
  - `auto_enrollment`: Auto-enrollment configuration
  - `auto_increase`: Auto-increase settings
  - `employer_match`: Employer matching rules
  - `employer_nec`: Non-elective contribution rules
  - `irs_limits`: IRS limits by year
  - `behavioral_params`: Behavioral parameters
  - `contributions`: Contribution rules
  - `eligibility_events`: Eligibility event configurations
  - `proactive_decrease`: Proactive decrease rules
  - `contribution_increase`: Contribution increase rules
- **Used In**: Central configuration for all plan-related rules
- **Search Tags**: class:PlanRules, module:config.models

### IRSYearLimits

- **Purpose**: Defines IRS limits for retirement plans for a specific year.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `compensation_limit`: Annual compensation limit (e.g., 401(a)(17))
  - `deferral_limit`: Elective deferral limit (e.g., 402(g))
  - `catchup_limit`: Catch-up contribution limit for age 50+
- **Used In**: Compliance and contribution limit calculations
- **Search Tags**: class:IRSYearLimits, attr:compensation_limit, attr:deferral_limit, attr:catchup_limit, module:config.models

### MatchTier

- **Purpose**: Defines a single tier for employer matching contributions.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `match_rate`: Employer match rate (e.g., 0.5 for 50%)
  - `cap_deferral_pct`: Maximum employee deferral percentage this tier applies to (e.g., 0.06 for 6%)
- **Used In**: Employer matching contribution calculations
- **Search Tags**: class:MatchTier, attr:match_rate, attr:cap_deferral_pct, module:config.models

### CompensationParams

- **Purpose**: Defines generic compensation parameters for employees.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `comp_base_salary`: Base salary amount
  - `comp_std`: Standard deviation (optional)
  - `comp_increase_per_age_year`: Annual increase per year of age
  - `comp_increase_per_tenure_year`: Annual increase per year of tenure
  - `comp_log_mean_factor`: Log mean factor (default: 1.0)
  - `comp_spread_sigma`: Salary spread sigma (default: 0.3)
  - `comp_min_salary`: Minimum salary threshold
- **Used In**: Compensation calculations and projections
- **Search Tags**: class:CompensationParams, module:config.models

### EligibilityRules

- **Purpose**: Defines rules for employee eligibility in retirement plans.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `entry_date`: Plan entry date
  - `waiting_period_days`: Waiting period in days
  - `min_age`: Minimum age requirement
  - `min_service_months`: Minimum service requirement in months
  - `entry_dates`: List of specific entry dates
- **Used In**: Determining employee eligibility for plan participation
- **Search Tags**: class:EligibilityRules, module:config.models

### ScenarioDefinition

- **Purpose**: Defines a simulation scenario with parameters and rules.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `name`: Scenario name
  - `description`: Scenario description
  - `start_year`: Starting year of the simulation
  - `end_year`: Ending year of the simulation
  - `plan_rules`: Plan-specific rules
  - `global_parameters`: Global simulation parameters
- **Used In**: Scenario configuration and simulation setup
- **Search Tags**: class:ScenarioDefinition, module:config.models

### EligibilityConfig

- **Purpose**: Defines basic eligibility criteria for retirement plan participation.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `min_age`: Minimum age requirement in years (default: 21)
  - `min_service_months`: Minimum service requirement in months (default: 12)
- **Used In**: Determining initial employee eligibility for retirement plans
- **Search Tags**: class:EligibilityConfig, attr:min_age, attr:min_service_months, module:config.plan_rules

### EligibilityRules

- **Purpose**: Comprehensive eligibility configuration with additional criteria.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `min_age`: Minimum age in years (optional)
  - `min_service_months`: Minimum service in months (optional)
  - `min_hours_worked`: Minimum hours worked requirement (optional)
- **Used In**: Advanced eligibility determination with multiple criteria
- **Search Tags**: class:EligibilityRules, attr:min_age, attr:min_service_months, attr:min_hours_worked, module:config.models

### EligibilityEventsConfig

- **Purpose**: Configures milestone-based eligibility events and notifications.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `milestone_months`: List of service-month milestones (e.g., [12, 24, 36])
  - `milestone_years`: List of service-year milestones (converted to months)
  - `event_type_map`: Mapping from milestone months to event types
- **Used In**: Triggering events at specific service milestones
- **Search Tags**: class:EligibilityEventsConfig, attr:milestone_months, attr:milestone_years, attr:event_type_map, module:config.plan_rules

### ProactiveDecreaseConfig

- **Purpose**: Configures proactive contribution rate decreases based on historical contribution patterns.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `lookback_months`: Number of months to look back for historical contributions (default: 12)
  - `threshold_pct`: Minimum drop from historical high to trigger decrease (e.g., 0.05 for 5%)
  - `event_type`: Type of event to emit (default: "EVT_PROACTIVE_DECREASE")
- **Used In**: Proactive contribution decrease logic to prevent contribution limit issues
- **Search Tags**: class:ProactiveDecreaseConfig, attr:lookback_months, attr:threshold_pct, attr:event_type, module:config.plan_rules

### ContributionIncreaseConfig

- **Purpose**: Configures settings for detecting and handling contribution increases.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `min_increase_pct`: Minimum percentage increase to trigger an event
  - `event_type`: Type of event to emit (default: "EVT_CONTRIB_INCREASE")
- **Used In**: Tracking and processing contribution increase events
- **Search Tags**: class:ContributionIncreaseConfig, attr:min_increase_pct, attr:event_type, module:config.plan_rules

### AutoEnrollmentConfig

- **Purpose**: Configures automatic enrollment settings for retirement plans.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `enabled`: Whether auto-enrollment is active (default: False)
  - `default_rate`: Default contribution rate for auto-enrolled employees (default: 0.03)
  - `window_days`: Number of days after eligibility for auto-enrollment window (default: 90)
- **Used In**: Automatic enrollment of eligible employees in retirement plans
- **Search Tags**: class:AutoEnrollmentConfig, attr:enabled, attr:default_rate, attr:window_days, module:config.plan_rules

### AutoIncreaseConfig

- **Purpose**: Configures automatic contribution rate increases.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `increase_pct`: Percentage increase for automatic raises (default: 0.01)
  - `cap`: Maximum contribution rate (default: 0.10)
  - `frequency_years`: Years between automatic increases (default: 1)
- **Used In**: Scheduled increases to employee contribution rates
- **Search Tags**: class:AutoIncreaseConfig, attr:increase_pct, attr:cap, attr:frequency_years, module:config.plan_rules

- **Purpose**: Configures auto-enrollment settings for retirement plans.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `enabled`: Whether auto-enrollment is active (default: False)
  - `default_rate`: Default deferral rate for auto-enrolled employees (default: 0.03)
  - `window_days`: Enrollment window in days after eligibility (default: 90)
- **Used In**: Auto-enrollment logic for retirement plans
- **Search Tags**: class:AutoEnrollmentConfig, attr:enabled, attr:default_rate, attr:window_days, module:config.plan_rules

### EnrollmentConfig

- **Purpose**: Configures enrollment settings including auto-enrollment and voluntary enrollment.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `window_days`: Enrollment window in days (default: 30)
  - `allow_opt_out`: Whether employees can opt out of enrollment (default: True)
  - `default_rate`: Default deferral rate (default: 0.05)
  - `auto_enrollment`: AutoEnrollmentConfig instance
  - `voluntary_enrollment_rate`: Rate of voluntary enrollment (default: 0.0)
- **Used In**: Plan enrollment configuration
- **Search Tags**: class:EnrollmentConfig, attr:window_days, attr:allow_opt_out, attr:default_rate, attr:auto_enrollment, attr:voluntary_enrollment_rate, module:config.plan_rules

### AutoEnrollOutcomeDistribution

- **Purpose**: Defines probability distribution for auto-enrollment outcomes.
- **Key Methods**:
  - `check_probabilities_sum_to_one()`: Validates that probabilities sum to 1.0
- **Key Attributes**:
  - `prob_opt_out`: Probability of opting out
  - `prob_stay_default`: Probability of staying at default rate
  - `prob_opt_down`: Probability of opting to a lower rate
  - `prob_increase_to_match`: Probability of increasing to match rate
  - `prob_increase_high`: Probability of increasing to higher rate
- **Used In**: Auto-enrollment simulation
- **Search Tags**: class:AutoEnrollOutcomeDistribution, method:check_probabilities_sum_to_one, attr:prob_opt_out, attr:prob_stay_default, attr:prob_opt_down, attr:prob_increase_to_match, attr:prob_increase_high, module:config.models

### AutoEnrollmentRules

- **Purpose**: Configures auto-enrollment rules and behaviors.
- **Key Methods**:
  - `validate_auto_enrollment_rules()`: Validates auto-enrollment configuration
- **Key Attributes**:
  - `enabled`: Whether auto-enrollment is active
  - `window_days`: Enrollment window in days
  - `proactive_enrollment_probability`: Probability of proactive enrollment
  - `default_rate`: Default deferral rate
  - `opt_down_target_rate`: Target rate for opt-down
  - `outcome_distribution`: AutoEnrollOutcomeDistribution instance
- **Used In**: Auto-enrollment rule configuration
- **Search Tags**: class:AutoEnrollmentRules, method:validate_auto_enrollment_rules, attr:enabled, attr:window_days, attr:proactive_enrollment_probability, attr:default_rate, attr:opt_down_target_rate, attr:outcome_distribution, module:config.models

### AutoIncreaseConfig

- **Purpose**: Configures auto-increase settings for employee contributions.
- **Key Methods**: None (data container)
- **Key Attributes**:
  - `enabled`: Whether auto-increase is active
  - `increase_pct`: Percentage increase amount
  - `cap`: Maximum deferral rate
  - `min_tenure_months`: Minimum tenure required
- **Used In**: Automatic contribution increases
- **Search Tags**: class:AutoIncreaseConfig, attr:enabled, attr:increase_pct, attr:cap, attr:min_tenure_months, module:config.plan_rules

### IRSYearLimits

- **Location**: `config.models.IRSYearLimits`
- **Description**: Manages IRS limits for retirement plans by year
- **Inherits from**: `pydantic.BaseModel`
- **Attributes**:
  - `compensation_limit`: Annual compensation limit (e.g., 401(a)(17))
  - `deferral_limit`: Elective deferral limit (e.g., 402(g))
  - `catchup_limit`: Catch-up contribution limit for age 50+
  - `catchup_eligibility_age`: Age at which catch-up contributions are allowed (default: 50)
- **Methods**:
  - `is_catchup_eligible(age)`: Checks if an employee is eligible for catch-up contributions
  - `get_effective_limit(age, is_catchup_eligible)`: Returns the applicable contribution limit

### MatchTier

- **Location**: `config.models.MatchTier`
- **Description**: Represents a tier in the employer matching formula
- **Inherits from**: `pydantic.BaseModel`
- **Attributes**:
  - `match_rate`: Employer match rate for this tier (e.g., 0.5 for 50%)
  - `cap_deferral_pct`: Maximum employee deferral percentage this tier applies to
- **Methods**:
  - `calculate_match(employee_deferral_pct, compensation)`: Calculates employer match amount

## Data Module

### DataReadError

- **Location**: `data.readers.DataReadError`
- **Description**: Exception raised when there's an error reading data
- **Inherits from**: `Exception`
- **Attributes**:
  - `message`: Error message describing the issue
  - `file_path`: Path to the file being read
  - `data_type`: Type of data being read

### DataValidator

- **Location**: `data.validators.DataValidator`
- **Description**: Validates data integrity and consistency
- **Methods**:
  - `validate_schema(data, schema)`: Validates data against a schema
  - `check_required_fields(data, required_fields)`: Ensures required fields are present
  - `validate_data_types(data, type_mapping)`: Validates field data types
  - `check_value_ranges(data, range_specs)`: Validates numeric ranges

## Dynamics Module

### SalarySampler (Protocol)

- **Location**: `dynamics.sampling.salary.SalarySampler`
- **Description**: Protocol defining the interface for salary sampling strategies
- **Methods**:
  - `sample(level, **kwargs)`: Samples a salary for a given job level
  - `validate_salary(level, salary)`: Validates if salary is within level range

### DefaultSalarySampler

- **Location**: `dynamics.sampling.salary.DefaultSalarySampler`
- **Description**: Default implementation of salary sampling using level-based ranges
- **Inherits from**: `SalarySampler`
- **Attributes**:
  - `random_state`: Random seed for reproducibility
  - `comp_ratio_range`: Tuple of (min, max) compensation ratio
  - `midpoint_progression`: Controls how salary grows with experience
- **Methods**:
  - `sample(level, experience=0, **kwargs)`: Samples salary based on level and experience
  - `_get_target_salary(level, experience)`: Calculates target salary for experience

## Engines Module

### SimulationOrchestrator

- **Location**: `engines.simulation.SimulationOrchestrator`
- **Description**: Manages the overall simulation flow
- **Methods**:
  - `run_simulation(start_year, end_year)`: Runs simulation for specified years
  - `initialize_simulation(initial_data)`: Sets up initial simulation state
  - `finalize_simulation()`: Performs cleanup and final calculations

### ProjectionEngine

- **Location**: `engines.projection.ProjectionEngine`
- **Description**: Handles multi-year projections
- **Methods**:
  - `project_workforce(years)`: Projects workforce changes over years
  - `calculate_metrics()`: Calculates key metrics from projections
  - `generate_reports()`: Creates output reports

## Plan Rules Module

### EligibilityRule

- **Location**: `plan_rules.eligibility.EligibilityRule`
- **Description**: Defines plan eligibility criteria
- **Attributes**:
  - `min_age`: Minimum age requirement
  - `min_service_months`: Minimum service requirement
  - `entry_dates`: List of annual entry dates
- **Methods**:
  - `is_eligible(employee, as_of_date)`: Checks eligibility
  - `get_next_entry_date(hire_date)`: Calculates next entry date

### ContributionRule

- **Location**: `plan_rules.contributions.ContributionRule`
- **Description**: Handles contribution calculations
- **Methods**:
  - `calculate_employee_contribution(employee, compensation)`: Calculates employee deferral
  - `calculate_employer_match(employee, employee_contribution)`: Calculates employer match
  - `get_contribution_limits(year)`: Returns applicable limits

## State Module

### EmployeeSnapshot

- **Location**: `state.snapshot.EmployeeSnapshot`
- **Description**: Represents employee state at a point in time
- **Attributes**:
  - `employee_id`: Unique employee identifier
  - `first_name`, `last_name`: Employee name
  - `hire_date`: Original hire date
  - `termination_date`: Termination date if applicable
  - `job_level`: Current job level/grade
  - `department`: Department/division
  - `base_salary`: Annual base salary

### EventLog

- **Location**: `state.events.EventLog`
- **Description**: Tracks all state changes for audit and analysis
- **Methods**:
  - `log_event(event_type, employee_id, effective_date, **metadata)`: Records an event
  - `get_employee_history(employee_id)`: Retrieves all events for an employee
  - `get_events_by_type(event_type, start_date, end_date)`: Filters events by type and date range

## Utilities Module

### Logger

- **Location**: `utils.logging.Logger`
- **Description**: Centralized logging system
- **Methods**:
  - `debug(msg, **kwargs)`: Logs debug message
  - `info(msg, **kwargs)`: Logs info message
  - `warning(msg, **kwargs)`: Logs warning message
  - `error(msg, **kwargs)`: Logs error message
  - `critical(msg, **kwargs)`: Logs critical message

### DateUtils

- **Location**: `utils.dates.DateUtils`
- **Description**: Date manipulation and validation
- **Methods**:
  - `add_years(date, years)`: Adds years to a date
  - `years_between(start_date, end_date)`: Calculates years between dates
  - `is_business_day(date)`: Checks if date is a business day
  - `next_business_day(date)`: Gets next business day
  - `format_date(date, format_str='%Y-%m-%d')`: Formats date as string

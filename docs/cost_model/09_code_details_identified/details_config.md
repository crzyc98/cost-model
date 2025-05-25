# Configuration Classes Details

## Core Configuration

### MainConfig
- **Location**: `config.models.MainConfig`
- **Description**: Root configuration model containing global parameters and scenario definitions
- **Inherits from**: `pydantic.BaseModel`
- **Key Methods**:
  - `check_baseline_scenario_exists()`: Validates baseline scenario presence
- **Key Attributes**:
  - `global_parameters`: Global simulation parameters
  - `scenarios`: Dictionary of scenario definitions
- **Search Tags**: `class:MainConfig`, `config:root`

### GlobalParameters
- **Location**: `config.models.GlobalParameters`
- **Description**: Global simulation parameters
- **Inherits from**: `pydantic.BaseModel`
- **Key Methods**:
  - `validate_global_parameters()`: Validates parameter consistency
- **Key Attributes**:
  - `start_year`, `projection_years`, `random_seed`
  - `annual_compensation_increase_rate`
  - `annual_termination_rate`
  - `new_hire_termination_rate`
- **Search Tags**: `class:GlobalParameters`, `config:parameters`

### ScenarioDefinition
- **Location**: `config.models.ScenarioDefinition`
- **Description**: Scenario-specific configuration
- **Inherits from**: `pydantic.BaseModel`
- **Key Attributes**:
  - `name`: Scenario identifier
  - Parameters that can override global settings
  - `plan_rules`: Scenario-specific plan rules
- **Search Tags**: `class:ScenarioDefinition`, `config:scenario`

## Plan Rules

### PlanRules
- **Location**: `config.plan_rules.PlanRules`
- **Description**: Container for all plan-related rules
- **Search Tags**: `class:PlanRules`, `plan:rules`

### ContributionRules
- **Location**: `config.plan_rules.ContributionRules`
- **Description**: Rules for employee and employer contributions
- **Search Tags**: `class:ContributionRules`, `plan:contributions`

### EligibilityConfig
- **Location**: `config.plan_rules.EligibilityConfig`
- **Description**: Eligibility criteria for plan participation
- **Search Tags**: `class:EligibilityConfig`, `plan:eligibility`

### EnrollmentConfig
- **Location**: `config.plan_rules.EnrollmentConfig`
- **Description**: Enrollment settings and rules
- **Search Tags**: `class:EnrollmentConfig`, `plan:enrollment`

### AutoEnrollmentConfig
- **Location**: `config.plan_rules.AutoEnrollmentConfig`
- **Description**: Auto-enrollment settings
- **Key Attributes**:
  - `enabled`: Whether auto-enrollment is active
  - `window_days`: Enrollment window in days
  - `default_rate`: Default contribution rate
- **Search Tags**: `class:AutoEnrollmentConfig`, `plan:auto_enrollment`

### AutoIncreaseConfig
- **Location**: `config.plan_rules.AutoIncreaseConfig`
- **Description**: Automatic contribution increase settings
- **Search Tags**: `class:AutoIncreaseConfig`, `plan:auto_increase`

### ContributionConfig
- **Location**: `config.plan_rules.ContributionConfig`
- **Description**: Contribution settings and limits
- **Search Tags**: `class:ContributionConfig`, `plan:contribution`

### ProactiveDecreaseConfig
- **Location**: `config.plan_rules.ProactiveDecreaseConfig`
- **Description**: Settings for proactive contribution decreases
- **Search Tags**: `class:ProactiveDecreaseConfig`, `plan:proactive_decrease`

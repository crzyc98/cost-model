# Class Inventory

This document provides a comprehensive list of all classes in the codebase, organized by module and functionality.

## Configuration (`config/`)

### models.py
- `MainConfig`: Root model for the entire configuration file
- `GlobalParameters`: Global simulation parameters
- `ScenarioDefinition`: Defines a simulation scenario
- `IRSYearLimits`: Manages IRS limits for retirement plans
- `MatchTier`: Represents a tier in the employer matching formula
- `AutoEnrollOutcomeDistribution`: Defines probability distribution for auto-enrollment outcomes
- `OnboardingBumpRules`: Configures compensation bumps for new hires
- `EmployerMatchRules`: Defines employer matching contribution rules
- `CompensationParams`: Generic compensation parameters
- `BehavioralParams`: Behavioral parameters for employee decisions
- `ContributionRules`: Rules for plan contributions
- `PlanRules`: Container for all plan rule configurations

## Data (`data/`)
- `DataReadError`: Exception for data reading errors
- `DataWriteError`: Exception for data writing errors
- `DataValidator`: Validates data integrity and consistency
- `DataTransformer`: Handles data transformation and normalization

## Dynamics (`dynamics/`)
- `DefaultSalarySampler`: Default implementation of salary sampling
- `EmployeeLifecycle`: Manages employee state transitions
- `CompensationModel`: Handles compensation changes and adjustments

## Engines (`engines/`)
- `SimulationOrchestrator`: Manages the overall simulation flow
- `ProjectionEngine`: Handles multi-year projections
- `DynamicsEngine`: Manages year-to-year workforce changes
- `RulesEngine`: Applies business rules to employee data

## Machine Learning (`ml/`)
- `MLModel`: Protocol for machine learning models with predict_proba
- `TurnoverPredictor`: Predicts employee turnover risk
- `CompensationModel`: Models salary growth and adjustments
- `BehaviorSimulator`: Simulates employee decisions and behaviors

## Plan Rules (`plan_rules/`)
- `ProactiveDecreaseConfig`: Configuration for proactive decrease rules
- `AutoEnrollmentConfig`: Configuration for auto-enrollment settings
- `EligibilityRule`: Defines plan eligibility criteria
- `EnrollmentRule`: Manages plan enrollment logic
- `ContributionRule`: Handles contribution calculations
- `VestingRule`: Manages vesting schedules and calculations

## State Management (`state/`)
- `JobLevel`: Represents a job level in the organization hierarchy
- `EmployeeSnapshot`: Represents employee state at a point in time
- `EventLog`: Tracks all state changes
- `TenureTracker`: Manages employee tenure calculations
- `SnapshotBuilder`: Builds and updates employee snapshots

## Utilities (`utils/`)
- `Logger`: Centralized logging system
- `ErrorHandler`: Standardized error handling
- `DateUtils`: Date manipulation utilities
- `ValidationUtils`: Common validation functions

## Complete Class List by Module

### config/
```
config/
├── models.py
│   ├── MainConfig
│   ├── GlobalParameters
│   ├── ScenarioDefinition
│   ├── IRSYearLimits
│   ├── MatchTier
│   ├── AutoEnrollOutcomeDistribution
│   ├── OnboardingBumpRules
│   ├── EmployerMatchRules
│   ├── CompensationParams
│   ├── BehavioralParams
│   ├── ContributionRules
│   └── PlanRules
└── plan_rules.py
    └── AutoEnrollmentConfig
```

### data/
```
data/
├── readers.py
│   └── DataReader
├── writers.py
│   └── DataWriter
└── validators.py
    └── DataValidator
```

### dynamics/
```
dynamics/
├── compensation.py
│   └── DefaultSalarySampler
└── engine.py
    └── EmployeeLifecycle
```

### engines/
```
engines/
├── run_one_year/
│   ├── orchestrator.py
│   │   └── SimulationOrchestrator
│   └── plan_rules.py
│       └── RulesEngine
└── projection.py
    └── ProjectionEngine
```

### ml/
```
ml/
├── ml_utils.py
│   └── MLModel (Protocol)
└── turnover.py
    └── TurnoverPredictor
```

### plan_rules/
```
plan_rules/
├── proactive_decrease.py
│   └── ProactiveDecreaseConfig
├── auto_increase.py
├── contributions.py
├── eligibility.py
├── enrollment.py
└── __init__.py
```

### state/
```
state/
├── job_levels/
│   ├── models.py
│   │   └── JobLevel
│   ├── loader.py
│   └── __init__.py
├── builder.py
│   └── SnapshotBuilder
├── event_log.py
│   └── EventLog
└── tenure.py
    └── TenureTracker
```

## Related Documentation

- [Configuration Classes](03_config_classes.md) - Detailed documentation of configuration classes
- [State Management](07_state_schema.md) - State tracking and schema details
- [Detailed Class Documentation](09_code_details_identified.md) - In-depth class documentation

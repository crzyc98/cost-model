# Machine Learning & Plan Rules

This document details the machine learning components and retirement plan rules in the Workforce Simulation & Cost Model system.

## Machine Learning Components

### Turnover Prediction

- **Location**: `cost_model.ml.turnover`
- **Description**: Predicts employee turnover risk using machine learning
- **Key Functions**:
  - `apply_ml_turnover()`: Applies a trained model to predict turnover
  - `train_turnover_model()`: Trains a LightGBM model for turnover prediction
  - `build_training_data()`: Prepares training data from historical census files
- **Features Used**:
  - Age
  - Tenure
  - Gross compensation
  - Job level
  - Department
  - Historical turnover patterns

### ML Utilities

- **Location**: `cost_model.ml.ml_utils`
- **Description**: Common utilities for machine learning tasks
- **Key Components**:
  - `MLModel` protocol: Defines the interface for ML models
  - `predict_turnover()`: Helper function for making turnover predictions
  - Feature preparation and preprocessing utilities

### Simulation Integration

- **Location**: `cost_model.utils.simulation_utils`
- **Description**: Integrates ML components into the simulation workflow
- **Key Functions**:
  - `sample_terminations()`: Applies stochastic termination sampling
  - `apply_ml_turnover()`: Integrates ML-based turnover predictions
  - `apply_comp_increase()`: Handles compensation increases

## Plan Rules

### Eligibility

- **Location**: `cost_model.plan_rules.eligibility`
- **Description**: Determines employee eligibility for retirement plans
- **Key Features**:
  - Configurable age and service requirements
  - Support for entry dates and waiting periods
  - Integrated with event logging
- **Usage Example**:
  ```python
  from cost_model.plan_rules.eligibility import run as run_eligibility
  
  # Run eligibility check
  eligibility_events = run_eligibility(snapshot, as_of_date, config)
  ```

### Enrollment

- **Location**: `cost_model.plan_rules.enrollment`
- **Description**: Handles plan enrollment and auto-enrollment
- **Key Features**:
  - Automatic enrollment with configurable default rates
  - Opt-out handling
  - Integration with eligibility rules
  - Event logging

### Contributions

- **Location**: `cost_model.plan_rules.contributions`
- **Description**: Manages employee and employer contributions
- **Key Features**:
  - Configurable matching formulas
  - Support for multiple match tiers
  - Integration with IRS limits
  - Detailed event logging

### Auto-Increase

- **Location**: `cost_model.plan_rules.auto_increase`
- **Description**: Implements automatic contribution rate increases
- **Key Features**:
  - Configurable increase rates and caps
  - Support for opt-out
  - Integration with enrollment status

### Proactive Decrease

- **Location**: `cost_model.plan_rules.proactive_decrease`
- **Description**: Identifies employees who have decreased contributions
- **Key Features**:
  - Configurable lookback periods
  - Threshold-based event generation
  - Integration with contribution history

### Eligibility Events

- **Location**: `cost_model.plan_rules.eligibility_events`
- **Description**: Emits milestone eligibility events
- **Key Features**:
  - Configurable milestone periods
  - Support for both months and years
  - Custom event type mapping

## Configuration

### Plan Rules Configuration Example

```yaml
plan_rules:
  # Eligibility settings
  eligibility:
    min_age: 21
    min_service_months: 3
    entry_dates: ["01/01", "04/01", "07/01", "10/01"]
  
  # Auto-enrollment settings
  auto_enrollment:
    enabled: true
    default_rate: 0.03  # 3% default deferral rate
    auto_increase: true
    auto_increase_rate: 0.01  # 1% annual increase
    auto_increase_max: 0.10  # Maximum 10% deferral rate
  
  # Employer match settings
  employer_match:
    enabled: true
    tiers:
      - match_rate: 1.0  # 100% match
        cap_deferral_pct: 0.03  # Up to 3% of compensation
      - match_rate: 0.5  # 50% match
        cap_deferral_pct: 0.05  # On next 2% of compensation
  
  # Auto-increase settings
  auto_increase:
    enabled: true
    increase_pct: 0.01  # 1% annual increase
    cap: 0.10  # Maximum 10% deferral rate
  
  # Proactive decrease settings
  proactive_decrease:
    enabled: true
    lookback_months: 12  # Look back 12 months for peak contribution
    threshold_pct: 0.05  # 5% decrease threshold
    
  # Eligibility events settings
  eligibility_events:
    milestone_months: [3, 6, 9]  # Months of service milestones
    milestone_years: [1, 3, 5]  # Years of service milestones
    event_type_map:  # Maps milestones to event types
      3: "EVT_3_MONTH_ELIGIBILITY"
      12: "EVT_1_YEAR_ELIGIBILITY"
```

## Implementation Details

### ML Integration

The system uses machine learning for turnover prediction, which can be configured through the simulation parameters. The ML model is trained on historical data and can be updated as needed.

### Event Processing

All plan rules generate events that are logged and can be analyzed. The event log provides a complete audit trail of all plan-related activities.

### Performance Considerations

- The system is designed to handle large employee populations efficiently
- Caching is used where appropriate to improve performance
- Batch processing is used for operations that can be vectorized

## Related Documentation

- [Configuration Classes](03_config_classes.md) - Detailed configuration options
- [State Management](07_state_schema.md) - How plan participation is tracked
- [Dynamics & Engines](05_dynamics_engines.md) - How plan rules are applied in simulations
- [Data Classes](04_data_classes.md) - Data structures and schemas
- [API Reference](../api/plan_rules.md) - Complete API documentation for plan rules

## Best Practices

1. **Configuration Management**
   - Use version control for configuration files
   - Document all custom rules and exceptions
   - Test configuration changes in a non-production environment

2. **Performance Tuning**
   - Use batch processing for large datasets
   - Monitor performance metrics
   - Consider using caching for frequently accessed data

3. **Testing**
   - Write unit tests for custom rules
   - Validate configuration before deployment
   - Test edge cases and boundary conditions

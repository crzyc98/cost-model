# Plan Rules Engine

## Eligibility Rules

### EligibilityEngine
- **Location**: `cost_model.plan_rules.eligibility.EligibilityEngine`
- **Description**: Determines employee eligibility for retirement plans
- **Key Methods**:
  - `is_eligible()`: Checks if employee meets criteria
  - `get_eligibility_date()`: Calculates when employee becomes eligible
  - `process_eligibility()`: Processes eligibility for employee group
- **Search Tags**: `class:EligibilityEngine`, `rules:eligibility`

### Configuration
```yaml
eligibility:
  minimum_age: 21
  service_months: 3
  entry_dates: ["01/01", "04/01", "07/01", "10/01"]
  classes:
    - name: "Hourly"
      minimum_hours: 1000
    - name: "Salary"
      minimum_months: 12
```

## Enrollment Rules

### AutoEnrollment
- **Location**: `cost_model.plan_rules.enrollment.AutoEnrollment`
- **Description**: Handles automatic plan enrollment
- **Features**:
  - Configurable opt-out periods
  - Default contribution rates
  - Escalation schedules
- **Search Tags**: `class:AutoEnrollment`, `rules:enrollment`

### EnrollmentManager
- **Location**: `cost_model.plan_rules.enrollment.EnrollmentManager`
- **Description**: Manages enrollment lifecycle
- **Search Tags**: `class:EnrollmentManager`, `rules:enrollment`

## Contribution Rules

### ContributionEngine
- **Location**: `cost_model.plan_rules.contributions.ContributionEngine`
- **Description**: Calculates employee and employer contributions
- **Features**:
  - Multiple contribution types (match, non-elective, profit sharing)
  - IRS limits and testing
  - True-up calculations
- **Search Tags**: `class:ContributionEngine`, `rules:contributions`

### ContributionTiers
```yaml
contribution_tiers:
  - match_percent: 100
    up_to_percent: 3
  - match_percent: 50
    up_to_percent: 5
non_elective: 3.0
catch_up_enabled: true
```

## Vesting Rules

### VestingSchedule
- **Location**: `cost_model.plan_rules.vesting.VestingSchedule`
- **Description**: Tracks and applies vesting schedules
- **Vesting Types**:
  - Cliff vesting
  - Graded vesting
  - Hybrid schedules
- **Search Tags**: `class:VestingSchedule`, `rules:vesting`

### VestingCalculator
- **Location**: `cost_model.plan_rules.vesting.VestingCalculator`
- **Description**: Calculates vested balances
- **Search Tags**: `class:VestingCalculator`, `rules:vesting`

## Integration

### RulesEngine
- **Location**: `cost_model.plan_rules.engine.RulesEngine`
- **Description**: Coordinates all plan rules
- **Features**:
  - Rule chaining
  - Dependency management
  - Transaction support
- **Search Tags**: `class:RulesEngine`, `rules:engine`

### Event Processing
```python
# Example rule processing
def process_rules(employee, effective_date):
    # Check eligibility
    if not eligibility_engine.is_eligible(employee, effective_date):
        return []
        
    events = []
    
    # Process auto-enrollment
    if enrollment_engine.should_auto_enroll(employee, effective_date):
        events.append(enrollment_engine.create_enrollment_event(employee))
    
    # Calculate contributions
    contribution = contribution_engine.calculate(employee)
    if contribution:
        events.append(contribution_engine.create_contribution_event(employee, contribution))
    
    return events
```

## Testing Rules

### Test Cases
- **Location**: `tests/unit/plan_rules/`
- **Coverage**:
  - Edge cases
  - Boundary conditions
  - Error scenarios
  - Performance testing

### Test Data Generator
- **Location**: `tests/data_generators/`
- **Features**:
  - Randomized test data
  - Scenario builders
  - Data validation

# Compensation Engine

## Purpose
Manages all aspects of employee compensation including base salary, raises, and bonuses.

## Key Features
- Annual salary adjustments
- Promotion raises
- Merit increases
- Bonus calculations
- Compensation bands and structures
- COLA (Cost of Living Adjustments)

## Key Functions

### update_salary()
- **Location**: `cost_model.dynamics.compensation.update_salary`
- **Description**: Applies COLA, promotion raises, and merit distributions
- **Parameters**:
  - `employee`: Employee record
  - `year`: Current simulation year
  - `config`: Compensation configuration
- **Search Tags**: `function:update_salary`, `compensation:salary`

### apply_comp_bump()
- **Location**: `cost_model.dynamics.compensation.apply_comp_bump`
- **Description**: Handles compensation increases with configurable parameters
- **Search Tags**: `function:apply_comp_bump`, `compensation:adjustment`

## Configuration

```yaml
compensation:
  annual_increase_rate: 0.03
  promotion_increase:
    min: 0.05
    max: 0.15
  cola:
    enabled: true
    rate: 0.02
```

## Compensation Bands

| Band | Min Salary | Max Salary |
|------|------------|------------|
| 1    | 50,000     | 70,000     |
| 2    | 65,000     | 90,000     |
| 3    | 85,000     | 120,000    |
| 4    | 100,000    | 150,000    |
| 5    | 140,000    | 200,000    |

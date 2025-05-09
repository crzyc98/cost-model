# cost_model.plan_rules

This package contains the business logic for plan eligibility, enrollment, contributions, and auto-increase features. Each module encapsulates a specific set of plan rules used throughout the simulation.

## Enrollment
Handles eligibility checks and enrollment logic for employees.

::: cost_model.plan_rules.enrollment

## Contributions
Calculates employee and employer contributions, applying IRS limits and plan-specific formulas.

::: cost_model.plan_rules.contributions

## Auto-Increase
Implements logic for automatic deferral rate increases (auto-increase/auto-escalation).

::: cost_model.plan_rules.auto_increase

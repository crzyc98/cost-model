# Population Growth Calculation

## Overview
This document explains how the model dynamically grows the population each simulation year to achieve a target net growth rate while accounting for actual terminations.

## Modeling Process
1. **Reset New-Hire Flags**: Clear `is_new_hire` for all agents at start of year.
2. **Identify Terminations**: Use `annual_termination_rate` (adjusted by tenure, age, compensation) to determine actual terminations (count `T_t`).
3. **Agent Steps & Scheduler**: Execute agent behaviors (comp, eligibility, deferral, contributions), then advance the scheduler.
4. **Update Status**: Mark terminated agents (`Terminated` or `New Hire Terminated`), active continuous, previously terminated, or new hires.
5. **Growth Calculation**: Compute hires needed to meet net growth target.
6. **New Hires**: Create `H_t` new agents with random hire dates and census-based compensation.
7. **Proration & Data Collection**: Prorate compensation, collect model and agent data.

## Growth Calculation Formula
- **Nₜ** = active headcount at start of year _t_.
- **g** = `annual_growth_rate` (e.g., 0.02 for 2%).
- **Tₜ** = actual terminations in year _t_.

1. Compute desired net increase:  
   &nbsp;&nbsp;Δₜ = round(Nₜ × g)
2. Compute hires:  
   &nbsp;&nbsp;Hₜ = Δₜ + Tₜ

After terminations and hires, final headcount = Nₜ − Tₜ + Hₜ = Nₜ + Δₜ.

## Key Assumptions & Parameters
- **annual_termination_rate**: Base attrition, adjusted by agent tenure, age, and compensation.
- **new_hire_termination_rate**: Used to enforce minimum/maximum tenure for hires in their first year.
- **Hire Dates**: Uniformly random across the calendar year.
- **Compensation**: Sampled via normal distribution using census-derived mean/stddev, floored at 30% of mean.
- **Agent IDs**: Incremental unique IDs assigned in sequence.
- **Plan Rules**: Growth does not depend on deferral or contribution rules.

Configuration parameters in `data/config.yaml`:
```yaml
annual_termination_rate: 0.13
annual_growth_rate: 0.02
new_hire_termination_rate: 0.20
new_hire_average_age: 30
```

## Comparison to Previous Logic
Previously, hires were computed via a calibration formula:
```
H = N × (t + g) / (1 − h_t)
```
This was replaced by the simpler additive formula to directly tie hires to observed terminations and target growth.

## Benefits of Dynamic Approach
- **Stability**: Guarantees net growth of _g_ regardless of attrition swings.
- **Flexibility**: Reacts to actual termination count year-over-year.
- **Transparency**: Easy to trace hire counts and growth outcomes.

---
*Document generated on 2025-04-18*

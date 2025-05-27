# Data Validation and Known Issues

This document tracks data validation issues and their resolutions in the cost model simulation.

## Resolved Issues

### 1. NaN Values in EMP_LEVEL (Resolved ✅)

**Issue:** NaN values were appearing in the `employee_level` column during Markov promotions.

**Root Cause:** 
- The inner `apply_promotion_markov` function in `sampling.py` wasn't properly handling cases where an employee's current level might be NaN.
- The promotion logic needed to ensure all employees have valid levels after promotion/demotion.

**Solution:**
- Added defensive programming in `sampling.py` to handle NaN levels by initializing `new_levels` from `current_levels` and ensuring proper handling of invalid levels.
- Added validation to ensure no NaN values remain after promotion logic.

**Verification:**
- Added diagnostic logging in `markov_promotion.py` to verify clean data flow.
- Confirmed that `[MARKOV_PROMOTION DIAGNOSTIC] IMMEDIATELY after apply_promotion_markov: No NaN values in employee_level, all good!` appears in logs.

### 2. Tenure Band Format Standardization (Resolved ✅)

**Issue:** Mismatch in tenure band formats between employee data (`0-1`) and hazard table (`0-1yr`) was causing missing termination rates.

**Root Cause:**
- The hazard table expected tenure bands in the format `X-Yyr` (e.g., `0-1yr`) while employee data used `X-Y` (e.g., `0-1`).
- This mismatch caused termination rates to be missing for affected employees, particularly level 2 employees.

**Solution:**
- Implemented standardization in `orchestrator.py` to convert tenure bands from `X-Y` to `X-Yyr` format before hazard table lookups.
- Added logging to track when standardization occurs and how many entries were converted.

**Verification:**
- Verified in logs that standardization is being applied correctly.
- Confirmed reduction in "missing term_rate" warnings after implementation.
- Validated that termination rates are now being correctly applied to all employees, including level 2.

## Current Warnings (Under Investigation)

### 1. Missing Termination Rates

**Warning:** `[TERM] Year XXXX: N employees missing term_rate after merge. Filling with 0.`

**Potential Causes:**
1. Mismatch between `employee_tenure_band` values in the employee data and the hazard table.
2. Missing combinations of `employee_level` and `employee_tenure_band` in the hazard table.

**Investigation Status:**
- Standardized tenure band format to use '0-1' consistently across the codebase.
- The format change was made to ensure consistency between employee data and the hazard table.
- Need to verify if all required level/tenure combinations exist in the hazard table.

### 2. Demotion Warnings

**Warning:** `Employee X demoted from level Y to Z`

**Status:** 
- These are informational messages from the promotion/demotion logic.
- Verify if these demotion rates match expectations from the promotion matrix.

### 3. Event Data Issues

**Warning:** `Found NA values in event_time/event_type for new events`

**Status:**
- Need to ensure all events have required metadata.
- May need to add validation when creating new events.

## Next Steps

1. **Hazard Table Coverage:**
   - Generate a report of all unique `(employee_level, employee_tenure_band)` combinations in the population.
   - Compare against the hazard table to identify missing combinations.
   - Consider adding default rates for any missing combinations.

2. **Event Data Validation:**
   - Add validation when creating new events to ensure all required fields are populated.
   - Consider making `event_time` and `event_type` required fields.

3. **Logging Improvements:**
   - Add more context to warning messages (e.g., which employees are affected).
   - Consider promoting some warnings to errors if they indicate data integrity issues.

## Appendix: Diagnostic Queries

To investigate missing termination rates:

```python
# Get unique level/tenure combinations in the population
pop_combos = df[[EMP_LEVEL, EMP_TENURE_BAND]].drop_duplicates()

# Get unique level/tenure combinations in the hazard table
hazard_combos = hazard_table[[EMP_LEVEL, EMP_TENURE_BAND]].drop_duplicates()

# Find combinations in population not covered by hazard table
missing = pop_combos.merge(
    hazard_combos, 
    on=[EMP_LEVEL, EMP_TENURE_BAND],
    how='left',
    indicator=True
).query('_merge == "left_only"')
```

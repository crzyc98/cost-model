# Column Names and Data Schema

## Overview

The cost model uses a centralized column naming system defined in `cost_model/utils/columns.py`. This approach eliminates magic strings, prevents drift between scripts, and makes future renaming trivial.

## Current Implementation

The column names are organized into several categories in `cost_model/utils/columns.py`:

### Census and Snapshot Columns

```python
# Census and snapshot columns (raw and standardized)
EMP_SSN = "employee_ssn"
EMP_ROLE = "employee_role"
EMP_BIRTH_DATE = "employee_birth_date"
EMP_HIRE_DATE = "employee_hire_date"
EMP_TERM_DATE = "employee_termination_date"
EMP_GROSS_COMP = "employee_gross_compensation"
EMP_TENURE = "employee_tenure"  # Standard column for years of service
EMP_PLAN_YEAR_COMP = "employee_plan_year_compensation"
EMP_CAPPED_COMP = "employee_capped_compensation"
EMP_DEFERRAL_RATE = "employee_deferral_rate"
EMP_CONTR = "employee_contribution"
EMPLOYER_CORE = "employer_core_contribution"
EMPLOYER_MATCH = "employer_match_contribution"
EMP_ID = "employee_id"
```

### Status and Eligibility Flags

```python
# Flags
IS_ELIGIBLE = "is_eligible"
IS_PARTICIPATING = "is_participating"
ELIGIBILITY_ENTRY_DATE = "eligibility_entry_date"
STATUS_COL = "status"
ACTIVE_STATUS = "Active"
INACTIVE_STATUS = "Inactive"
HOURS_WORKED = "hours_worked"
```

### Auto Enrollment and Auto Increase Columns

```python
# Auto Enrollment (AE) columns
AE_OPTED_OUT = "ae_opted_out"
PROACTIVE_ENROLLED = "proactive_enrolled"
AUTO_ENROLLED = "auto_enrolled"

# Auto Increase (AI) columns
AI_OPTED_OUT = "ai_opted_out"
AI_ENROLLED = "ai_enrolled"

ENROLLMENT_DATE = "enrollment_date"
AE_WINDOW_START = "ae_window_start"
AE_WINDOW_END = "ae_window_end"
FIRST_CONTRIBUTION_DATE = "first_contribution_date"
AE_OPT_OUT_DATE = "ae_opt_out_date"
AUTO_REENROLLED = "auto_reenrolled"
ENROLLMENT_METHOD = "enrollment_method"
BECAME_ELIGIBLE_DURING_YEAR = "became_eligible_during_year"
WINDOW_CLOSED_DURING_YEAR = "window_closed_during_year"
```

### Summary Columns

```python
# Summary CSV column names
SUM_EMP_CONTR = "total_employee_contributions"
SUM_EMP_MATCH = "total_employer_match"
```

### Event Log Columns

```python
# Event log columns
EVENT_TYPE = "event_type"  # Type of event (hire, term, comp, etc.)
EVENT_DATE = "event_date"  # When the event occurred
EVENT_ID = "event_id"      # Unique identifier for the event
EVENT_DATA = "event_data"  # JSON data associated with the event
```

### Demographic Analysis Columns

```python
# Band columns for demographic analysis
TENURE_BAND = "tenure_band"  # Categorical tenure (e.g., "0-1 years")
AGE_BAND = "age_band"        # Categorical age (e.g., "21-30")
```

## Usage in Code

To use these column names in your code:

```python
from cost_model.utils.columns import EMP_DEFERRAL_RATE, IS_ELIGIBLE, EMPLOYER_MATCH

# Access DataFrame columns
df[EMP_DEFERRAL_RATE] = 0.03
total_match = df[EMPLOYER_MATCH].sum()

# Check column existence
assert IS_ELIGIBLE in df.columns
```

## Best Practices

1. **Always import from columns.py**: Never use string literals for column names
   ```python
   # Bad
   df["employee_deferral_rate"] = 0.03
   
   # Good
   from cost_model.utils.columns import EMP_DEFERRAL_RATE
   df[EMP_DEFERRAL_RATE] = 0.03
   ```

2. **Use column constants in tests**: Ensure tests are resilient to column name changes
   ```python
   # In tests
   from cost_model.utils.columns import EMP_DEFERRAL_RATE
   assert EMP_DEFERRAL_RATE in df.columns
   ```

3. **Add new columns to columns.py**: When adding new data fields, define them in the central location

4. **Document column meanings**: When adding new columns, include a comment explaining their purpose

## Schema Definition

The central schema for snapshots is defined in `cost_model.state.snapshot` with column names from `columns.py` and appropriate data types:

```python
# Example from snapshot.py
SNAPSHOT_COLS = [
    EMP_ID, EMP_ROLE, EMP_BIRTH_DATE, EMP_HIRE_DATE, EMP_TERM_DATE,
    EMP_GROSS_COMP, EMP_PLAN_YEAR_COMP, EMP_DEFERRAL_RATE,
    IS_ELIGIBLE, IS_PARTICIPATING, TENURE_BAND
]

SNAPSHOT_DTYPES = {
    EMP_ID: pd.StringDtype(),
    EMP_ROLE: pd.StringDtype(),
    EMP_BIRTH_DATE: 'datetime64[ns]',
    EMP_HIRE_DATE: 'datetime64[ns]',
    EMP_TERM_DATE: 'datetime64[ns]',
    EMP_GROSS_COMP: pd.Float64Dtype(),
    EMP_PLAN_YEAR_COMP: pd.Float64Dtype(),
    EMP_DEFERRAL_RATE: pd.Float64Dtype(),
    IS_ELIGIBLE: pd.BooleanDtype(),
    IS_PARTICIPATING: pd.BooleanDtype(),
    TENURE_BAND: pd.StringDtype()
}
```

## Recent Updates

Recent updates to the column definitions include:

1. Added `TENURE_BAND` and `EMP_DEFERRAL_RATE` to `SNAPSHOT_COLS` and their respective data types to `SNAPSHOT_DTYPES` in `cost_model.state.snapshot.py` to resolve KeyError issues in multi-year projections.

2. Added special constants for auto-enrollment events: `BECAME_ELIGIBLE_DURING_YEAR` and `WINDOW_CLOSED_DURING_YEAR` to replace hard-coded strings in the auto-enrollment module.

3. Added event type constants for contribution events to standardize event logging.
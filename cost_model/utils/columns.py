import pandas as pd

# cost_model/utils/columns.py

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
EMP_LEVEL = "employee_level"    
EMP_ACTIVE      = "active"
EMP_TENURE_BAND = "tenure_band"
EMP_LEVEL_SOURCE= "job_level_source"
EMP_EXITED      = "exited"
SIMULATION_YEAR = "simulation_year"
TERM_RATE = "term_rate"
COMP_RAISE_PCT = "comp_raise_pct"
NEW_HIRE_TERM_RATE = "new_hire_term_rate"
COLA_PCT = "cola_pct"
CFG = "cfg"

# Event log columns
EVENT_ID = "event_id"
EVENT_TYPE = "event_type"
EVENT_TIME = "event_time"
VALUE_JSON = "value_json"
META = "meta"

# Event types
EVT_HIRE = "EVT_HIRE"
EVT_TERM = "EVT_TERM"
EVT_CONTRIB = "EVT_CONTRIB"
EVT_CONTRIB_INCR = "EVT_CONTRIB_INCR"

# Flags
IS_ELIGIBLE = "is_eligible"
IS_PARTICIPATING = "is_participating"
ELIGIBILITY_ENTRY_DATE = "eligibility_entry_date"
STATUS_COL = "status"
ACTIVE_STATUS = "Active"
INACTIVE_STATUS = "Inactive"
HOURS_WORKED = "hours_worked"

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

# — new summary‐CSV column names —
SUM_EMP_CONTR = "total_employee_contributions"
SUM_EMP_MATCH = "total_employer_match"
SUM_EMP_CORE = "total_employer_core"
SUM_EMP_COST = "total_employer_cost"
SUM_CONTRIB = "total_contributions"
SUM_PLAN_COMP = "total_plan_year_compensation"
SUM_CAP_COMP = "total_capped_compensation"

# Hazard table columns
HazardTable = "hazard_table"
SIMULATION_YEAR = "simulation_year"
TERM_RATE = "term_rate"
COMP_RAISE_PCT = "comp_raise_pct"
NEW_HIRE_TERM_RATE = "new_hire_termination_rate"
COLA_PCT = "cola_pct"
CFG = "cfg"

# Note: EMP_LEVEL is already defined above as "employee_level"

AVG_DEFERRAL_PART = "avg_deferral_rate_participants"
AVG_DEFERRAL_TOTAL = "avg_deferral_rate_total"

# and so on for headcount, participation rates, plan_comp, cap_comp, pct columns…
SUM_HEADCOUNT = "headcount"
SUM_ELIGIBLE = "eligible"
SUM_PARTICIPATING = "participating"
RATE_PARTICIP_ELIG = "participation_rate_eligible"
RATE_PARTICIP_TOTAL = "participation_rate_total"
PCT_EMP_COST_PLAN = "employer_cost_pct_plan_comp"
PCT_EMP_COST_CAP = "employer_cost_pct_capped_comp"

DATE_COLS = [EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE]

# Central raw→standard mapping
RAW_TO_STD_COLS = {
    "ssn": EMP_SSN,
    "role": EMP_ROLE,
    "birth_date": EMP_BIRTH_DATE,
    "employee_birth_date": EMP_BIRTH_DATE,
    "hire_date": EMP_HIRE_DATE,
    "employee_hire_date": EMP_HIRE_DATE,
    "termination_date": EMP_TERM_DATE,
    "employee_termination_date": EMP_TERM_DATE,
    "gross_compensation": EMP_GROSS_COMP,
    "plan_year_compensation": EMP_PLAN_YEAR_COMP,
    "capped_compensation": EMP_CAPPED_COMP,
    "employee_deferral_pct": EMP_DEFERRAL_RATE,
    "pre_tax_deferral_percentage": EMP_DEFERRAL_RATE,
    "employee_contribution_amt": EMP_CONTR,
    "pre_tax_contributions": EMP_CONTR,
    "employer_core_contribution_amt": EMPLOYER_CORE,
    "employer_core_contribution": EMPLOYER_CORE,
    "employer_match_contribution_amt": EMPLOYER_MATCH,
    "employer_match_contribution": EMPLOYER_MATCH,
    "eligibility_entry_date": ELIGIBILITY_ENTRY_DATE,
}

def to_nullable_bool(series: pd.Series) -> pd.Series:
    """
    Convert a boolean‐like Series into pandas’ nullable BooleanDtype.
    """
    return series.astype("boolean")

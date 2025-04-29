# utils/columns.py

# Census and snapshot columns (raw and standardized)
EMP_SSN = "employee_ssn"
EMP_ROLE = "employee_role"
EMP_BIRTH_DATE = "employee_birth_date"
EMP_HIRE_DATE = "employee_hire_date"
EMP_TERM_DATE = "employee_termination_date"

EMP_GROSS_COMP = "employee_gross_compensation"
EMP_PLAN_YEAR_COMP = "employee_plan_year_compensation"
EMP_CAPPED_COMP = "employee_capped_compensation"
EMP_DEFERRAL_RATE = "employee_deferral_rate"
EMP_PRE_TAX_CONTR = "employee_pre_tax_contribution"

EMPLOYER_MATCH = "employer_match_contribution"
EMPLOYER_NEC = "employer_non_elective_contribution"

# Standardized (post-rename) columns for downstream logic
GROSS_COMP = "gross_compensation"
PLAN_YEAR_COMP = "plan_year_compensation"
CAPPED_COMP = "capped_compensation"
DEFERRAL_RATE = "deferral_rate"
PRE_TAX_CONTR = "pre_tax_contributions"

# Flags
IS_ELIGIBLE = "is_eligible"
IS_PARTICIPATING = "is_participating"
ELIGIBILITY_ENTRY_DATE = "eligibility_entry_date"

# Central raw→standard mapping
RAW_TO_STD_COLS = {
    'ssn': EMP_SSN,
    'role': EMP_ROLE,
    'birth_date': EMP_BIRTH_DATE,
    'hire_date': EMP_HIRE_DATE,
    'termination_date': EMP_TERM_DATE,
    'gross_compensation': EMP_GROSS_COMP,
    'plan_year_compensation': EMP_PLAN_YEAR_COMP,
    'capped_compensation': EMP_CAPPED_COMP,
    'employee_deferral_pct': EMP_DEFERRAL_RATE,
    'pre_tax_deferral_percentage': EMP_DEFERRAL_RATE,
    'employee_contribution_amt': EMP_PRE_TAX_CONTR,
    'pre_tax_contributions': EMP_PRE_TAX_CONTR,
    'employer_core_contribution_amt': EMPLOYER_NEC,
    'employer_non_elective_contribution': EMPLOYER_NEC,
    'employer_match_contribution_amt': EMPLOYER_MATCH,
    'employer_match_contribution': EMPLOYER_MATCH,
    'eligibility_entry_date': ELIGIBILITY_ENTRY_DATE,
}

# Date columns to format post-rename
DATE_COLS = [EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE, ELIGIBILITY_ENTRY_DATE]

import pandas as pd

def to_nullable_bool(series: pd.Series) -> pd.Series:
    """
    Convert a boolean‐like Series into pandas’ nullable BooleanDtype.
    """
    return series.astype('boolean')
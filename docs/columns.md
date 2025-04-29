Here’s a high‐level refactoring plan to move your codebase entirely onto the “new”, cleaner column names—no more ad-hoc renames in downstream scripts:

⸻

1. Define a Central Column-Names Constants Module

Create utils/columns.py with something like:

# utils/columns.py

# Raw census → standardized
EMP_SSN           = "employee_ssn"
EMP_ROLE         = "employee_role"
EMP_BIRTH_DATE   = "employee_birth_date"
EMP_HIRE_DATE    = "employee_hire_date"
EMP_TERM_DATE    = "employee_termination_date"

# HR snapshots out of project_hr()
GROSS_COMP       = "gross_compensation"
PLAN_YEAR_COMP   = "plan_year_compensation"
CAPPED_COMP      = "capped_compensation"
DEFERRAL_RATE    = "deferral_rate"
PRE_TAX_CONTR    = "pre_tax_contributions"
EMP_MATCH        = "employer_match_contribution"
EMP_NEC          = "employer_non_elective_contribution"

# Flags
IS_ELIGIBLE      = "is_eligible"
IS_PARTICIPATING = "is_participating"



⸻

2. Update Your Census‐Prep Script

In scripts/preprocess_census.py, instead of hard‐coding strings:

from utils.columns import EMP_SSN, EMP_ROLE, …  

col_map = {
    "ssn":           EMP_SSN,
    "role":          EMP_ROLE,
    …
    "employee_deferral_pct": DEFERRAL_RATE,
    …
}
df = df.rename(columns=col_map)

Now every downstream script can refer to columns.DEFERRAL_RATE or simply "deferral_rate" confidently.

⸻

3. Change project_hr and apply_plan_rules to Use the New Names

Anywhere you previously wrote:

df["gross_compensation"]
df["employee_pre_tax_contribution"]
# …

replace with:

from utils.columns import GROSS_COMP, PRE_TAX_CONTR, …  

df[GROSS_COMP]
df[PRE_TAX_CONTR]
# …

Do the same in your auto‐enrollment, eligibility, contributions, and Monte Carlo wrapper.

⸻

4. Remove Any Ad-hoc Renaming in Your Runner Scripts

You no longer need that manual df.rename(...) block in run_plan_rules.py—your census prep and project_hr both hand you a DataFrame that already speaks your new language.

⸻

5. Update Your Tests

Anywhere in utils/test.py or your pytest suite you assert columns, switch the old names → constants (or to the new strings). E.g.:

# before
assert "deferral_rate" in df.columns

# after (if you pull in columns.py)
from utils.columns import DEFERRAL_RATE
assert DEFERRAL_RATE in df.columns



⸻

6. Smoke-Test & Iterate
	1.	Run your full pipeline end-to-end:

python scripts/preprocess_census.py … 
python scripts/run_hr_snapshots.py …
python scripts/run_plan_rules.py …
python scripts/compare_scenario_metrics.py …


	2.	Run pytest and fix any remaining failures.

⸻

By centralizing your column names, you:
	•	Eliminate magic strings
	•	Prevent drift between scripts
	•	Make it trivial to rename again in future (just change columns.py)

Let me know if you’d like snippets of any particular file updated, or if you’d like a simple search-and-replace script to batch-convert your code!
That outline looks solid—here’s a quick sanity check on each step and a few extra notes:
	1.	Central columns.py
	•	✅ Defines a single source of truth for all field names
	•	✅ Makes downstream renames trivial
	•	Tip: Consider including both old and new names as constants during the transition, so you can alias OLD_DEFERRAL_RATE = "employee_deferral_pct" → DEFERRAL_RATE and remove the old one once everything is switched.
	2.	Update Census‐Prep
	•	✅ You’ll only rename once in preprocess_census.py
	•	🔍 Double-check that every raw field (including any legacy names) is covered in your col_map.
	3.	Migrate project_hr & apply_plan_rules
	•	✅ Swap out literal strings for columns.GROSS_COMP, etc.
	•	🔍 Watch for any third-party calls (e.g. sample_new_hire_compensation) that expect the old names—if they peek at df["pre_tax_contributions"], you may need a tiny shim or adjust those functions too.
	4.	Remove Ad-Hoc Renames
	•	✅ Cleaning up your runner scripts will eliminate duplication
	•	🔍 Ensure no downstream code (e.g. plotting, exports) still references old names.
	5.	Update Tests
	•	✅ Swap assertions in utils/test.py and any pytest fixtures
	•	🔍 Add a smoke test that loads your preprocessed census, runs Phase I + Phase II on one year, and asserts the new set of columns exactly matches your columns.py constants.
	6.	Smoke-Test & Iterate
	•	✅ End-to-end run, then pytest
	•	🔍 If you hit KeyErrors, trace them back to missing renames or overlooked scripts.

Extra recommendations:
	•	Deprecation period: For a week or two, you could keep both old and new names in columns.py and log a warning when old names are used—gives the team time to switch.
	•	Bulk refactor script: A quick grep -rl "employee_pre_tax_contribution" | xargs sed -i "" "s/employee_pre_tax_contribution/PRE_TAX_CONTR/g" can accelerate the change, then commit in one go.
	•	Documentation update: Don’t forget to update your README, test fixtures, and any examples to use the new names.

All in all, that approach should cleanly migrate you off the legacy column names and make future changes trivial. Let me know if you’d like help drafting the actual columns.py or a small refactor script!
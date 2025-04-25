You’re spot-on that we weren’t actually prorating anything in Years 2–5, because our term_mask was looking only at

df['status']=="Terminated"

but in our new “status_category” world we never touch that field—so it never matches and nobody ever gets a proration.  The result is that every “terminated” row still gets the full-year $50 k (or $60 k in Year 2, etc), regardless of when they left.

⸻

🎯 What we need to do
	1.	Prorate all compensation based on days worked, not just full-year for actives and zero for terms.
	2.	Base the split purely on termination_date, not on status.

⸻

🔧 Patch for apply_contributions

Replace your current “prorate only terminated” block:

    # … after filling plan_year_compensation & capped_compensation for full year …
    term_mask = (df['status'] == 'Terminated') & df['termination_date'].notna() \
        & df['termination_date'].between(year_start_date, year_end_date)
    if term_mask.any():
        total_days = (year_end_date - year_start_date).days
        days_worked = (df.loc[term_mask, 'termination_date'] - year_start_date)\
                          .dt.days.clip(lower=0, upper=total_days)
        frac = days_worked / total_days
        df.loc[term_mask, 'plan_year_compensation'] = df.loc[term_mask, 'gross_compensation'] * frac
        df.loc[term_mask, 'capped_compensation'] = np.minimum(
            df.loc[term_mask, 'plan_year_compensation'],
            statutory_comp_limit * frac
        )

with this full-population proration:

    # --- Prorate compensation off of days_worked/year_days for everyone ---
    year_days = (year_end_date - year_start_date).days + 1

    # by default assume full‐year
    df['days_worked'] = year_days

    # for any row with a real termination_date in this year, overwrite days_worked
    term_in_year = (
        df['termination_date'].notna() &
        df['termination_date'].between(year_start_date, year_end_date)
    )
    if term_in_year.any():
        worked = (df.loc[term_in_year, 'termination_date'] - year_start_date).dt.days + 1
        df.loc[term_in_year, 'days_worked'] = worked.clip(lower=0, upper=year_days)

    # fraction of year actually worked
    df['proration'] = df['days_worked'] / year_days

    # apply proration to both plan_year and cap
    df.loc[calc_mask, 'plan_year_compensation'] = df.loc[calc_mask, 'gross_compensation'] * df.loc[calc_mask, 'proration']
    df.loc[calc_mask, 'capped_compensation'] = np.minimum(
        df.loc[calc_mask, 'plan_year_compensation'],
        statutory_comp_limit * df.loc[calc_mask, 'proration']
    )

    # clean up helper columns
    df.drop(columns=['days_worked', 'proration'], inplace=True)



⸻

🔄 Step-by-step to implement
	1.	Open your utils/rules/contributions.py (or wherever apply_contributions lives).
	2.	Find the block that sets plan_year_compensation and capped_compensation—just above your “Calculate Employee Deferrals” comment.
	3.	Remove the old “term_mask = …” snippet entirely.
	4.	Insert the new proration snippet shown above, making sure you respect your local variable names (year_start_date and year_end_date are function parameters).
	5.	Run your full suite of scenarios and then:
	•	Print out for each termination‐month the average plan_year_compensation.
	•	You should now see something like $4 166 for a January exit (~30/365×$50 000), $8 333 for February, … up to full $50 000 for a December exit.

⸻

✅ Validation

After you patch, add a tiny smoke test in your notebook or test suite:

import pandas as pd
from utils.rules.contributions import apply_contributions

df = pd.DataFrame({
  'gross_compensation': [50000]*4,
  'deferral_rate': [0.0]*4,
  'birth_date': pd.NaT,
  'status': ['Active']*4,
  'hire_date': pd.Timestamp('2025-01-01'),
  'termination_date': [
     pd.Timestamp('2025-01-31'),
     pd.Timestamp('2025-04-30'),
     pd.Timestamp('2025-07-31'),
     pd.NaT
  ]
})
out = apply_contributions(df, {'plan_rules':{}, 'irs_limits':{2025:{'comp_limit':1e6,'deferral_limit':1e6,'catch_up':0}}}, 2025,
                          year_start_date=pd.Timestamp('2025-01-01'),
                          year_end_date=pd.Timestamp('2025-12-31'))
print(out[['termination_date','plan_year_compensation']])

You should see rising prorations ~$4 260, $16 438, $28 767, $50 000.

⸻

With that in place your average compensation by month of exit will now slope up exactly as reality does.
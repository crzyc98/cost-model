Here’s a step‐by‐step plan to make all of your sampling fully deterministic and to lock your new‐hire draws to a single “baseline” distribution:

⸻

1. Surface a single “master seed” in project_census
	1.	Add a random_seed argument (or pull it consistently from your scenario_config).

def project_census(
    start_df: pd.DataFrame,
    scenario_config: dict,
    baseline_scenario_config: dict,
    random_seed: int = None,           # ← new!
) -> dict[int, pd.DataFrame]:
    # …


	2.	At the top of the function, immediately seed every RNG you use:

import numpy as np, random
rng = np.random.default_rng(random_seed)
random.seed(random_seed)

From now on, always draw from rng (or pass random_seed on) rather than calling np.random or random ad‐hoc.

⸻

2. Compute and freeze the baseline hire‐salary distribution

Before entering your year‐by‐year loop:

# all in project_census, before `for year_num in …`
baseline_hire_salaries = start_df.loc[
    start_df['hire_date'].dt.year == start_year - 1,
    'gross_compensation'
].dropna()
if baseline_hire_salaries.empty:
    baseline_hire_salaries = start_df['gross_compensation'].dropna()

Remove any code in the loop that re‐assigns prev_hire_salaries or relies on prior years’ hires.

⸻

3. Pass the same seed (or derived seeds) into each helper
	1.	Compensation bump

current_df = apply_comp_increase(
    current_df,
    'gross_compensation',
    comp_increase_rate,
    seed=random_seed
)


	2.	Termination sampling

current_df = sample_terminations(
    current_df,
    'hire_date',
    termination_rate,
    end_date,
    seed=random_seed
)


	3.	New‐hire generation & compensation

nh_df = generate_new_hires( … , seed=random_seed)
nh_df['gross_compensation'] = sample_new_hire_compensation(
    nh_df,
    'gross_compensation',
    baseline_hire_salaries.values,
    seed=random_seed
)


	4.	ML turnover

current_df = apply_ml_turnover(
    current_df,
    model,
    feature_cols,
    end_date,
    seed=random_seed
)



Tip: If you want slight variation within the loop (e.g. each year draws differently but reproducibly), you can derive per‐year seeds like

year_seed = random_seed + year_num



⸻

4. Refactor each helper to accept and use that seed

Make sure each helper:

def sample_terminations(df, hire_date_col, rate, end_date, seed=None):
    rng = np.random.default_rng(seed)
    # … use rng instead of np.random

and similarly for sample_new_hire_compensation, apply_comp_increase, etc.

⸻

5. Remove “prev_hire_salaries” mutation

Strip out any code in your loop that does:

prev_hire_salaries = projected_data[year_num]… 

So your new hires always draw from the static baseline_hire_salaries.

⸻

6. Add a quick smoke‐test to verify determinism

In tests/test_projection_utils.py:

def test_projection_deterministic():
    df1 = project_census(df, config, base_config, random_seed=42)
    df2 = project_census(df, config, base_config, random_seed=42)
    assert df1 == df2  # or compare specific DataFrame contents

That will guarantee that every random draw—terminations, bumps, new hires, ML rollouts—is exactly the same every time you call it with the same seed.

⸻

Why this helps
	•	Traceable scenario differences: now any divergence in outcomes is driven purely by config changes (AE/AI flags), not by random drift in hires or terminations.
	•	Reproducibility: running with random_seed=123 always produces the same projection.
	•	Cleaner comparisons: you can compare “baseline” vs “new_hire_auto_escalate” without worrying that your hire‐salary distribution also drifted.

Let me know if you’d like me to sketch out the exact code diffs for each helper!
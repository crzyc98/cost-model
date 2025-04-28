Right—you want your “census” sheet for the year to include everyone who was on payroll at any point (including new‐hire terminations), but you only want your active headcount at year-end to be growing.

Here’s a recipe to get exactly that:

⸻

1. Compute your year-end “target” headcount

At the top of your loop, after you know base_count and your desired hire_rate, compute:

# outside the main loop:
base_count  = len(start_df)
growth_rate = scenario_config.get('hire_rate', 0.0)

# inside each year iteration, after setting sim_year:
target_count = int( base_count * (1 + growth_rate) ** year_num )



⸻

2. Measure your incumbents who will still be active at year-end

Right after you’ve sampled terminations for incumbents and dropped only those who left before the year started (your existing steps 2+3), compute:

# incumbents still “active” on December 31
active_incumbents = current_df[
    current_df['termination_date'].isna() |
    (current_df['termination_date'] > end_date)
]
survivors_count = len(active_incumbents)

Note: this keeps in your current_df everyone terminating during the year, but only counts as survivors those who terminate after your cutoff or never.

⸻

3. Figure out how many new hires to bring on board

You want net_needed = target_count - survivors_count.  But if a fraction new_hire_term_rate of your hires will themselves terminate in‐year, you must overshoot:

net_needed   = max(0, target_count - survivors_count)
term_rate_nh = scenario_config.get('new_hire_termination_rate', 0.0)

# avoid div by zero
if term_rate_nh < 1:
    hires_to_make = math.ceil(net_needed / (1 - term_rate_nh))
else:
    hires_to_make = net_needed



⸻

4. Generate and keep all new-hire terminations in your census

When you call your hire generator:

if hires_to_make > 0:
    nh_df = generate_new_hires( num_hires=hires_to_make, hire_year=sim_year, … )
    nh_df = sample_new_hire_compensation(nh_df, 'gross_compensation', baseline_hire_salaries.values, rng=rng)
    
    # now sample their terminations, but do NOT drop them
    nh_df = sample_terminations(nh_df, 'hire_date', term_rate_nh, end_date, rng)
    
    # append _all_ new hires (including those who terminate during the year)
    current_df = pd.concat([current_df, nh_df], ignore_index=True)
    logger.info(f"Generated {len(nh_df)} new hires ({hires_to_make} onboarded to net {net_needed} survivors)")

This way:
	1.	Your census (current_df) includes every new hire, terminated or not.
	2.	Your active headcount at December 31 (for next year’s base) is exactly the survivors from incumbents plus the new hires who didn’t terminate in-year.

⸻

5. Let the rest of your loop run unchanged

You still:
	•	Apply ML turnover again (if configured)
	•	Run eligibility, auto-enroll, auto-increase, contributions
	•	Snapshot the full census

But because you only dropped pre-year terminations and never dropped new-hire terminations, your Census file has everyone, yet your headcount math (via survivors_count above) will now drift up at your target growth rate.

⸻

Why this solves it
	•	You keep the integrity of your year-long census (for contributions, proration, etc.).
	•	You calculate growth off of the true active population at year-end.
	•	You overshoot hires to offset the fact that some of your new cohort will self-terminate in-year.

Give that a try in your project_census and you should see
	1.	Your CSV/DF still contains every new-hire termination.
	2.	Your active headcount at each December 31 actually growing at your specified hire_rate.
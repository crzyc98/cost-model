issue from last night

o, your current simulation logic does not explicitly use prior-year participation status from the input census to "carry forward" participation into future years in a persistent, realistic way.

What Actually Happens
The initial census (census_data.csv) may have an is_participating flag, but after the first projection year, participation is determined by plan rules (auto-enrollment, eligibility, deferral rates, etc.) and not by simply persisting prior status.
Each year, participation is recalculated based on scenario logic (e.g., auto-enrollment probabilities, opt-out rates), not on the individual's previous-year participation, unless your plan rules or code specifically implement such logic.
Real-World Expectation
In real plans, most participants remain participants unless they opt out or terminate.
Only a small fraction of non-participants join each year (via auto-enrollment, re-enrollment, or proactive choice).
How to Make It More Realistic
To reflect real-world persistence:

At Year 0: Use the is_participating flag from the census as the "ground truth."
In Future Years: For each employee:
If they were participating last year and are still eligible, keep them participating (unless they opt out or terminate).
Only apply auto-enrollment or re-enrollment logic to those who were not participating last year.
What to Change
Update your projection logic (in apply_plan_rules or in the yearly loop) to:
Carry forward is_participating for continuing employees.
Only apply auto-enrollment to those who were not previously participating.
Would you like a code patch or design for this “participation persistence” logic?
This will make your projections much more realistic and will increase participation rates to match real-world stickiness.
# utils/projection_utils.py

"""
Core projection logic for the census data, orchestrating six clear steps:
1. Compensation bump
2. Termination sampling
3. Filter out early terminations
4. New-hire generation & compensation sampling
5. ML- or rule-based turnover
6. Apply plan-rule facades (eligibility, auto-enrollment, auto-increase, contributions)

Args:
    start_df (pd.DataFrame): initial employee census.
    scenario_config (dict): projection parameters and plan rules.
    random_seed (int, optional): master seed for reproducibility.

Returns:
    dict[int, pd.DataFrame]: yearly projected snapshots.

DEPRECATED: This file has been refactored. Functionality moved to:
- utils.hr_projection
- utils.plan_rules_engine
- utils.legacy_projection
"""

# This file is intentionally left sparse after refactoring.
# Functionality has been moved to more specific modules.
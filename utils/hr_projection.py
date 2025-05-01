# utils/hr_projection.py
# This file contains HR-related projection logic (Phase I)

import logging
import math
import pandas as pd
import numpy as np
from typing import Union, Optional, Mapping, Any, Dict

# Use relative imports assuming these utils exist at the same level or are importable
from .date_utils import calculate_tenure
from .data_generation_utils import generate_new_hires # Use refactored generator
from .sampling.terminations import sample_terminations
from .sampling.new_hires import sample_new_hire_compensation
from .sampling.salary import SalarySampler, DefaultSalarySampler
from .ml.ml_utils import try_load_ml_model, predict_turnover
# Import necessary column names if not already available globally
from .columns import EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP

# Module-level logger as fallback
logger = logging.getLogger(__name__)


# Helper: Turnover (Accept and use logger)
def _apply_turnover(
    df: pd.DataFrame,
    hire_col: str,
    probs_or_rate: Union[float, pd.Series, np.ndarray], # Allow ndarray too
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rng: np.random.Generator,
    sampler: SalarySampler,
    prev_term_salaries: pd.Series,
    log: logging.Logger # Accept logger instance
) -> pd.DataFrame:
    # Log the type of turnover input for debugging
    log.debug(f"_apply_turnover called with rate/probs type: {type(probs_or_rate)}, shape/value: {getattr(probs_or_rate, 'shape', probs_or_rate)}")

    # sample_terminations should ideally log internally too
    df2 = sample_terminations(df, hire_col, probs_or_rate, end_date, rng)

    term_here = df2['employee_termination_date'].between(start_date, end_date, inclusive='both') # Check terms within the period
    n_term = term_here.sum()
    log.debug(f"_apply_turnover: {n_term} terminations marked for period {start_date.date()} to {end_date.date()}.")

    if n_term > 0:
        # Ensure prev_term_salaries is not empty before sampling
        if not prev_term_salaries.empty:
            log.debug(f"Sampling {n_term} termination salaries from previous distribution (size={len(prev_term_salaries)}).")
            # Pass the Series, sampler method expects it for .dropna()
            draws = sampler.sample_terminations(prev_term_salaries, size=n_term, rng=rng)
            # Assign sampled termination draws; draws may be numpy array or pandas Series
            # Ensure index alignment if draws is a Series, or direct assignment if array
            df2.loc[term_here, 'employee_gross_compensation'] = draws if isinstance(draws, np.ndarray) else draws.values
        else:
             log.warning("_apply_turnover: prev_term_salaries is empty, cannot sample termination salaries. Terminees will keep last comp.")
    return df2


# Helper: Compensation Bump (Accept and use logger)
def _apply_comp_bump(
    df: pd.DataFrame,
    comp_col: str,
    dist: Mapping[str, Any],
    rate: float,
    rng: np.random.Generator,
    sampler: SalarySampler,
    log: logging.Logger # Accept logger instance
) -> pd.DataFrame:
    df2 = df.copy()
    if comp_col not in df2:
        log.warning("Missing '%s' column, skipping comp bump", comp_col)
        return df2
    log.debug(f"Applying comp bump. Rate={rate:.2%}. Current avg comp: {df2[comp_col].mean():.0f}")
    # Ensure compensation column is float
    df2[comp_col] = pd.to_numeric(df2[comp_col], errors='coerce').astype(float)
    # Define groups by tenure (handle potential NaNs in tenure)
    df2['tenure'] = pd.to_numeric(df2['tenure'], errors='coerce')
    mask_second = df2['tenure'] == 1
    mask_exp = df2['tenure'] >= 2

    # 1) Second-year employees get sampler bump
    n_second_year = mask_second.sum()
    if n_second_year > 0:
        log.debug(f"Applying sampler bump to {n_second_year} second-year employees.")
        # Ensure sampler method exists
        if hasattr(sampler, 'sample_second_year'):
             df2.loc[mask_second, comp_col] = sampler.sample_second_year(
                 df2.loc[mask_second], comp_col=comp_col, dist=dist, rate=rate, rng=rng
             )
        else:
            log.warning("Salary sampler missing 'sample_second_year' method. Applying flat rate to second-year employees.")
            df2.loc[mask_second, comp_col] *= (1 + rate)

    # 2) Experienced employees get flat increase
    n_exp = mask_exp.sum()
    if n_exp > 0:
        log.debug(f"Applying flat rate bump ({rate:.1%}) to {n_exp} experienced employees.")
        df2.loc[mask_exp, comp_col] *= (1 + rate)

    log.info(
        "Comp bump applied: %d second-year, %d experienced. New avg comp: %.0f",
        n_second_year, n_exp, df2[comp_col].mean()
    )
    return df2


# Helper: Onboarding Bump (Accept and use logger)
def apply_onboarding_bump(
    df: pd.DataFrame,
    comp_col: str,
    ob_cfg: Mapping[str, Any],
    baseline_hire_salaries: pd.Series,
    rng: np.random.Generator,
    log: logging.Logger # Accept logger instance
) -> pd.DataFrame:
    """
    Apply onboarding bump for new hires.
    Methods: 'flat_rate' or 'sample_plus_rate'.
    """
    df2 = df.copy()
    if not ob_cfg.get('enabled', False):
        log.debug("Onboarding bump disabled.")
        return df2

    method = ob_cfg.get('method', '')
    rate = ob_cfg.get('rate', ob_cfg.get('flat_rate', 0.0)) # Support legacy key
    log.info(f"Applying onboarding bump: method='{method}', rate={rate:.2%}")

    if method == 'flat_rate':
        df2[comp_col] = df2[comp_col] * (1 + rate)
    elif method == 'sample_plus_rate':
        if not baseline_hire_salaries.empty:
            log.debug(f"Sampling {len(df2)} baseline salaries for onboarding bump.")
            # Ensure choice works with Series, might need .values if it expects array
            draws = rng.choice(baseline_hire_salaries.values, size=len(df2))
            df2[comp_col] = draws * (1 + rate)
        else:
            log.warning("Onboarding bump 'sample_plus_rate' method chosen, but baseline_hire_salaries is empty! Applying flat rate to existing comp instead.")
            df2[comp_col] = df2[comp_col] * (1 + rate) # Fallback
    else:
         log.warning(f"Unknown onboarding bump method '{method}', no bump applied.")

    log.debug(f"Onboarding bump applied. New avg comp for hires: {df2[comp_col].mean():.0f}")
    return df2


# Phase I: HR-only projection (census-only snapshots)
def project_hr(
    start_df: pd.DataFrame,
    scenario_config: Mapping[str, Any],
    random_seed: Optional[int] = None,
    parent_logger: Optional[logging.Logger] = None # Accept logger
) -> Dict[int, pd.DataFrame]:
    """
    Phase I: compensation bumps, terminations (rule + ML), headcount resets & new hires.
    Returns a dict mapping year→census-only DataFrame (no plan rules applied).
    """
    # --- Setup Logger ---
    log = parent_logger or logger # Use passed logger or module logger
    log.info("--- Starting HR Projection (project_hr) ---")

    # --- RNG Setup ---
    seed = random_seed if random_seed is not None else scenario_config.get('random_seed')
    log.info(f"Initializing RNGs with seed: {seed}")
    master_ss = np.random.SeedSequence(seed)
    ss_bump, ss_term, ss_nh, ss_ml = master_ss.spawn(4)
    rng_bump = np.random.default_rng(ss_bump)
    rng_term = np.random.default_rng(ss_term)
    rng_nh = np.random.default_rng(ss_nh)
    rng_ml = np.random.default_rng(ss_ml)
    log.debug("RNGs initialized.")

    # --- Config Parameters ---
    projection_years = scenario_config['projection_years']
    start_year = scenario_config['start_year'] # Simulation start year
    log.info(f"Projecting from {start_year} for {projection_years} years.")
    pr_cfg = scenario_config.get('plan_rules', {}) # Plan rules might have some HR params
    comp_increase_rate = scenario_config.get('annual_compensation_increase_rate', pr_cfg.get('annual_compensation_increase_rate', 0.0))
    termination_rate = scenario_config.get('annual_termination_rate', pr_cfg.get('annual_termination_rate', 0.0))
    growth_rate = scenario_config.get('annual_growth_rate', pr_cfg.get('annual_growth_rate', 0.0))
    maintain_hc = scenario_config.get('maintain_headcount', None)
    maintain_headcount = maintain_hc if maintain_hc is not None else pr_cfg.get('maintain_headcount', True)
    nh_term_rate = scenario_config.get('new_hire_termination_rate', pr_cfg.get('new_hire_termination_rate', 0.0))
    log.info(f"HR Params: CompRate={comp_increase_rate:.2%}, TermRate={termination_rate:.2%}, NHTermRate={nh_term_rate:.2%}, MaintainHC={maintain_headcount}, GrowthRate={growth_rate:.2%}")

    # --- ML Model Setup ---
    model_path = scenario_config.get('ml_model_path', '')
    features_path = scenario_config.get('model_features_path', '')
    projection_model, feature_cols = None, []
    if model_path and features_path:
        # try_load_ml_model should ideally log success/failure internally
        ml_pair = try_load_ml_model(model_path, features_path)
        if ml_pair:
             projection_model, feature_cols = ml_pair
             log.info(f"Loaded ML turnover model from {model_path} with {len(feature_cols)} features.")
    use_ml = scenario_config.get('use_ml_turnover', False) and projection_model is not None
    log.info(f"Using ML for turnover: {use_ml}")

    # --- Baseline Salaries ---
    baseline_hire_year = start_year - 1
    baseline_hire_salaries = start_df.loc[
        start_df[EMP_HIRE_DATE].dt.year == baseline_hire_year,
        EMP_GROSS_COMP # Make sure this col name is correct
    ].dropna()
    if baseline_hire_salaries.empty:
        log.warning(f"No hires found in baseline year {baseline_hire_year}. Using all initial employees for baseline salary distribution.")
        baseline_hire_salaries = start_df[EMP_GROSS_COMP].dropna()
    else:
        log.info(f"Using {len(baseline_hire_salaries)} salaries from year {baseline_hire_year} hires for baseline distribution.")

    # --- Initial State ---
    current_df = start_df.copy()
    # Ensure essential columns exist
    required_cols = [EMP_HIRE_DATE, EMP_GROSS_COMP]
    for col in required_cols:
         if col not in current_df.columns:
             log.error(f"Missing required column in start_df: '{col}'. Aborting.")
             return {}
    if EMP_TERM_DATE not in current_df.columns:
         log.warning(f"Column '{EMP_TERM_DATE}' not found. Creating with NaT.")
         current_df[EMP_TERM_DATE] = pd.NaT

    # Initialize prev_term_salaries as Series
    prev_term_salaries = current_df[EMP_GROSS_COMP].dropna()
    initial_base_count = len(start_df)
    log.info(f"Initial headcount: {initial_base_count}")
    hr_snapshots: Dict[int, pd.DataFrame] = {}

    # --- Yearly Projection Loop ---
    for year_idx in range(projection_years): # Loop 0 to N-1
        sim_year = start_year + year_idx
        log.info(f"--- Processing Simulation Year {sim_year} (Index {year_idx}) ---")
        start_date = pd.Timestamp(f"{sim_year}-01-01")
        end_date = pd.Timestamp(f"{sim_year}-12-31")

        hc_start_of_year = len(current_df)
        log.info(f"Year {sim_year} Start: Headcount={hc_start_of_year}")

        # Calculate Tenure
        current_df['tenure'] = calculate_tenure(current_df[EMP_HIRE_DATE], start_date)
        log.debug(f"Year {sim_year}: Recalculated tenure.")

        # 1. Comp Bump
        log.debug(f"Year {sim_year}: Applying compensation bump...")
        current_df = _apply_comp_bump(
            current_df, EMP_GROSS_COMP,
            scenario_config.get('second_year_compensation_dist', {}),
            comp_increase_rate, rng_bump, DefaultSalarySampler(), log=log # Pass logger
        )

        # 2. Early/Baseline Turnover (BEFORE new hires)
        log.debug(f"Year {sim_year}: Applying early/baseline turnover (Rate={termination_rate:.2%})...")
        current_df = _apply_turnover(
            current_df, EMP_HIRE_DATE, termination_rate,
            start_date, end_date, rng_term, DefaultSalarySampler(), prev_term_salaries, log=log # Pass logger
        )
        # Update prev_term_salaries *after* this round
        prev_term_salaries = current_df.loc[
            current_df[EMP_TERM_DATE].between(start_date, end_date, inclusive='both'),
            EMP_GROSS_COMP
        ].dropna()
        log.debug(f"Year {sim_year}: Updated prev_term_salaries distribution (size={len(prev_term_salaries)}) after early terms.")

        # 3. Drop Pre-Period Terminations (Sanity check)
        pre_period_terms_mask = current_df[EMP_TERM_DATE].notna() & (current_df[EMP_TERM_DATE] < start_date)
        n_pre_terms = pre_period_terms_mask.sum()
        if n_pre_terms > 0:
            log.warning(f"Year {sim_year}: Removing {n_pre_terms} employees terminated before {start_date.date()}.")
            current_df = current_df[~pre_period_terms_mask].copy()

        hc_after_early_term_and_cleanup = len(current_df[current_df[EMP_TERM_DATE].isna()])
        log.debug(f"Year {sim_year}: Headcount after early terms and cleanup = {hc_after_early_term_and_cleanup}")

        # 4. New Hires & Comp Sampling
        # Determine hires needed based on survivors *before* final term step
        survivors = len(current_df[current_df[EMP_TERM_DATE].isna()]) # Still active after early terms
        if maintain_headcount:
            target_count = initial_base_count
            log.debug(f"Year {sim_year}: MaintainHC=True. Target={target_count}, Survivors={survivors}")
            needed = max(0, target_count - survivors)
        else:
            target_count = int(initial_base_count * (1 + growth_rate) ** (year_idx + 1))
            net_needed = max(0, target_count - survivors)
            log.debug(f"Year {sim_year}: MaintainHC=False. Target={target_count}, Survivors={survivors}, NetNeeded={net_needed}")
            if nh_term_rate < 1 and net_needed > 0:
                needed = math.ceil(net_needed / (1 - nh_term_rate))
                log.debug(f"Year {sim_year}: Grossing up hires needed ({net_needed}) by NH term rate ({nh_term_rate:.1%}) to {needed}")
            else:
                needed = net_needed # Avoid division by zero if rate is 100% or net_needed is 0

        # Override needed if specific hire_rate is provided AND not maintaining headcount
        hire_rate = scenario_config.get('hire_rate')
        if hire_rate is not None and not maintain_headcount:
            needed = int(initial_base_count * hire_rate)
            log.info(f"Year {sim_year}: Overriding hires needed based on hire_rate ({hire_rate:.1%}). New needed = {needed}")

        log.info(f"Year {sim_year}: Hires needed = {needed}")

        if needed > 0:
            log.info(f"Generating {needed} new hires for {sim_year}.")
            # Get age parameters safely
            age_mean = float(scenario_config.get('new_hire_average_age', pr_cfg.get('new_hire_average_age', scenario_config.get('age_mean', 30))))
            age_std = float(scenario_config.get('new_hire_age_std_dev', pr_cfg.get('age_std_dev', scenario_config.get('age_std_dev', 5.0))))
            min_age = int(scenario_config.get('min_working_age', 18))
            max_age = int(scenario_config.get('max_working_age', 65))
            log.debug(f"New Hire Params: AgeMean={age_mean}, AgeStd={age_std}, MinAge={min_age}, MaxAge={max_age}")

            # Call generator function
            # Ensure generate_new_hires returns required columns like EMP_HIRE_DATE, EMP_BIRTH_DATE, etc.
            existing_ids = current_df['employee_id'].tolist() if not current_df.empty else []
            nh_df = generate_new_hires(
                num_hires=needed,
                hire_year=sim_year,
                scenario_config=scenario_config, # Pass the whole config
                existing_ids=existing_ids,
                rng=rng_nh
            )

            log.debug(f"Generated {len(nh_df)} raw new hire records.")

            # Sample Compensation for New Hires
            log.debug("Sampling compensation for new hires...")
            nh_df = sample_new_hire_compensation(
                nh_df, EMP_GROSS_COMP, baseline_hire_salaries, rng_nh
            )
            # Log sampled comp stats safely
            if not nh_df.empty and EMP_GROSS_COMP in nh_df.columns:
                 log.debug(f"  NH Comp Stats: Mean={nh_df[EMP_GROSS_COMP].mean():.0f}, "
                           f"Std={nh_df[EMP_GROSS_COMP].std():.0f}, "
                           f"Min={nh_df[EMP_GROSS_COMP].min():.0f}, "
                           f"Max={nh_df[EMP_GROSS_COMP].max():.0f}")

            # Apply Onboarding Bump
            ob_cfg = scenario_config.get('plan_rules', {}).get('onboarding_bump', {})
            log.debug("Applying onboarding bump to new hires...")
            nh_df = apply_onboarding_bump(
                nh_df, EMP_GROSS_COMP, ob_cfg, baseline_hire_salaries, rng_nh, log=log # Pass logger
            )

            # *** Log Detailed New Hire Info ***
            log.info(f"Logging details for {len(nh_df)} generated new hires (DEBUG level)...")
            for index, hire in nh_df.iterrows():
                 # Safely get attributes for logging using .get() for dict-like access on Series row
                 log.debug(f"  Generated NH {index+1}/{len(nh_df)}: "
                           f"ID={hire.get('employee_id', 'N/A')}, " # Assuming ID column exists
                           f"Role={hire.get('role', 'N/A')}, "
                           f"DOB={hire.get(EMP_BIRTH_DATE).strftime('%Y-%m-%d') if pd.notna(hire.get(EMP_BIRTH_DATE)) else 'N/A'}, "
                           f"HireDate={hire.get(EMP_HIRE_DATE).strftime('%Y-%m-%d') if pd.notna(hire.get(EMP_HIRE_DATE)) else 'N/A'}, "
                           f"Comp={hire.get(EMP_GROSS_COMP, 0):.0f}") # Format compensation

            # Add hires to current dataframe
            log.info(f"Adding {len(nh_df)} new hires to the census.")
            current_df = pd.concat([current_df, nh_df], ignore_index=True, sort=False)
            log.info(f"Total headcount after hires: {len(current_df)}")

            # *** Initialize plan-related columns for ALL rows AFTER concatenation ***
            # This ensures consistency for both existing and new employees if columns are missing
            plan_cols_defaults = {
                'is_eligible': False,
                'is_participating': False,
                'employee_deferral_rate': 0.0,
                'employee_pre_tax_contribution': 0.0,
                'employer_core_contribution': 0.0,
                'employer_match_contribution': 0.0,
                'eligibility_entry_date': pd.NaT,
                'employee_plan_year_compensation': 0.0, # May be recalculated by plan rules
                'employee_capped_compensation': 0.0,   # May be recalculated by plan rules
                'yos': 0 # Will be calculated next
                # Add other plan-specific columns needing default init here
            }
            for col, default_val in plan_cols_defaults.items():
                if col not in current_df.columns:
                    log.debug(f"Initializing column '{col}' with default: {default_val}")
                    current_df[col] = default_val
                else:
                    # Optionally fill NaNs if column exists but might have gaps, esp. for new hires
                    # Example: current_df[col] = current_df[col].fillna(default_val)
                    pass # Decide if filling existing NaNs is needed

        else:
            log.info("No new hires generated or added.")

        # 5. Final Turnover (applied AFTER new hires join)
        log.debug(f"Year {sim_year}: Applying final turnover (ML={use_ml})...")
        if use_ml:
            # ML-based turnover
            log.debug("Predicting ML turnover probabilities...")
            # Ensure current_df has features needed by the model
            probs = predict_turnover(
                current_df, projection_model, feature_cols, random_state=rng_ml
            )
            term_rate_or_probs = probs
        else:
             # Rule-based: Apply different rates if configured
             log.debug(f"Applying rule-based turnover (NH Rate={nh_term_rate:.2%}, Other Rate={termination_rate:.2%})")
             is_new_hire_this_year = (current_df[EMP_HIRE_DATE].dt.year == sim_year)
             term_rate_or_probs = np.where(is_new_hire_this_year, nh_term_rate, termination_rate)

        # Apply final turnover step
        current_df = _apply_turnover(
            current_df, EMP_HIRE_DATE, term_rate_or_probs,
            start_date, end_date, rng_term, # Re-use term RNG
            DefaultSalarySampler(), prev_term_salaries, log=log # Pass logger
        )

        # --- LOGGING: Track Terminations of Recent Hires ---
        log.info(f"Year {sim_year}: Analyzing terminations of employees hired >= {start_year}...")
        # Identify terminations *within the current simulation year*
        terminated_in_year_mask = current_df[EMP_TERM_DATE].between(start_date, end_date, inclusive='both')
        terminated_ids_this_year = current_df.loc[terminated_in_year_mask, 'employee_id'] # Assuming 'employee_id' exists

        new_hires_terminated_count = 0
        if not terminated_ids_this_year.empty:
            n_terms_total = len(terminated_ids_this_year)
            log.info(f"Year {sim_year}: Total {n_terms_total} employees terminated in this period.")
            # Check hire dates of those terminated this year
            terminated_df_subset = current_df[current_df['employee_id'].isin(terminated_ids_this_year.tolist())]
            # Check if hired in or after the overall simulation start_year
            hired_during_sim_mask = terminated_df_subset[EMP_HIRE_DATE].dt.year >= start_year
            new_hires_terminated_count = hired_during_sim_mask.sum()

            if new_hires_terminated_count > 0:
                terminated_new_hire_ids = terminated_df_subset.loc[hired_during_sim_mask, 'employee_id'].tolist()
                log.info(f"Year {sim_year}: {new_hires_terminated_count} terminated employees were hired >= {start_year}.")
                log.debug(f"Year {sim_year}: Terminated recent hire IDs: {terminated_new_hire_ids}")
            else:
                log.info(f"Year {sim_year}: No terminated employees were hired >= {start_year}.")
        else:
             log.info(f"Year {sim_year}: No employees terminated in this period.")
        # --- End Termination Tracking Log ---


        # Update prev_term_salaries for *next* year's sampling base
        # Use only salaries of those actually terminated in *this* period (sim_year)
        newly_terminated_salaries = current_df.loc[
            terminated_in_year_mask, # Use mask calculated above
            EMP_GROSS_COMP
        ].dropna()
        if not newly_terminated_salaries.empty:
             prev_term_salaries = newly_terminated_salaries # Overwrite with only this year's terms for next year
             log.debug(f"Year {sim_year}: Updated prev_term_salaries with {len(prev_term_salaries)} values from this year's terminations.")
        else:
             # Keep old prev_term_salaries if no one was terminated this year? Or reset?
             # Resetting might be safer if termination sampling expects recent data.
             # Let's keep the old ones for now if none terminated this cycle.
             log.debug(f"Year {sim_year}: No new terminations, prev_term_salaries distribution unchanged for next iter.")


        # --- Snapshotting ---
        # Create snapshot based on who is active at the *end* of the year
        active_at_year_end_mask = current_df[EMP_TERM_DATE].isna() | (current_df[EMP_TERM_DATE] > end_date)
        snapshot_df = current_df[active_at_year_end_mask].copy()
        hc_end_of_year = len(snapshot_df)
        log.info(f"Year {sim_year} End: Headcount={hc_end_of_year}")

        # Store snapshot using the simulation year as the key
        hr_snapshots[sim_year] = snapshot_df
        log.info(f"Stored HR snapshot for year {sim_year}.")

        # Prepare for next year: Carry forward only those active at year end
        current_df = snapshot_df.copy()


    log.info(f"--- Finished HR Projection (project_hr). Generated {len(hr_snapshots)} snapshots. ---")
    return hr_snapshots
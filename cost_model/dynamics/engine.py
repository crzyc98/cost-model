# cost_model/dynamics/engine.py
"""
Engine for running population dynamics simulation steps (Phase 1).
Orchestrates compensation changes, terminations, and hiring.
"""

import logging
import math
import pandas as pd
import numpy as np
from typing import Union, Optional, Mapping, Any, Dict

# Use relative imports within the cost_model package
try:
    # Ensure calculate_tenure is imported correctly
    from ..utils.date_utils import calculate_tenure
    # Import specific generation functions from hiring/compensation modules
    from .hiring import generate_new_hires
    from .compensation import apply_comp_bump, apply_onboarding_bump # Assuming these exist now
    from .termination import apply_turnover # Assuming termination logic is in termination.py
    # ML imports if needed
    from ..ml.ml_utils import try_load_ml_model, predict_turnover
    # Import necessary column names
    from ..utils.columns import EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP
    # Import sampling helpers if needed directly here, or called within sub-modules
    from .sampling.salary import SalarySampler, DefaultSalarySampler
except ImportError as e:
    print(f"Error importing dynamics components: {e}")
    # Define fallbacks for static analysis if needed
    EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP = 'employee_hire_date', 'employee_termination_date', 'employee_birth_date', 'employee_gross_compensation'
    def generate_new_hires(*args, **kwargs): raise NotImplementedError()
    def apply_comp_bump(*args, **kwargs): raise NotImplementedError()
    def apply_onboarding_bump(*args, **kwargs): raise NotImplementedError()
    def apply_turnover(*args, **kwargs): raise NotImplementedError()
    def try_load_ml_model(*args, **kwargs): return None, None
    def predict_turnover(*args, **kwargs): raise NotImplementedError()
    class DefaultSalarySampler: pass


# Module-level logger
logger = logging.getLogger(__name__)


# Renamed function from project_hr to run_dynamics_for_year
def run_dynamics_for_year(
    current_df: pd.DataFrame,
    year_config: Mapping[str, Any], # Use the resolved scenario config (like GlobalParameters model)
    sim_year: int,
    # Pass RNG if needed, otherwise initialize internally based on seed in config
    # rng: Optional[np.random.Generator] = None,
    parent_logger: Optional[logging.Logger] = None # Accept logger
) -> pd.DataFrame:
    """
    Runs one year of population dynamics: compensation, terminations, hiring.

    Args:
        current_df: DataFrame with the active population at the START of the year.
        year_config: The fully resolved configuration dictionary or object for the scenario for this year.
        sim_year: The current simulation year (e.g., 2025).
        parent_logger: Optional logger instance from the calling script.

    Returns:
        A pandas DataFrame representing the population state AFTER dynamics
        but BEFORE plan rules are applied for the year. Includes new hires
        and reflects terminations that occurred during the year.
    """
    # --- Setup Logger ---
    log = parent_logger or logger # Use passed logger or module logger
    log.info(f"--- Running Dynamics for Simulation Year {sim_year} ---")

    # --- RNG Setup (Example: derive from seed in config if not passed) ---
    seed = getattr(year_config, 'random_seed', None) # Get random seed from config (now accessing Pydantic model attribute directly)
    # Create separate RNGs for different processes using SeedSequence for better isolation
    if seed is not None:
        log.debug(f"Initializing dynamics RNGs with seed: {seed}")
        master_ss = np.random.SeedSequence(seed)
        # Spawn sequences for each major random process within dynamics
        ss_bump, ss_term, ss_nh, ss_ml = master_ss.spawn(4)
        rng_bump = np.random.default_rng(ss_bump)
        rng_term = np.random.default_rng(ss_term)
        rng_nh = np.random.default_rng(ss_nh)
        rng_ml = np.random.default_rng(ss_ml)
    else:
        log.warning("No random seed provided for dynamics. Results may not be reproducible.")
        # Initialize default RNGs if no seed
        rng_bump = np.random.default_rng()
        rng_term = np.random.default_rng()
        rng_nh = np.random.default_rng()
        rng_ml = np.random.default_rng()
    log.debug("Dynamics RNGs initialized.")

    # --- Config Parameters ---
    # Extract parameters needed for this year's dynamics from year_config
    # Use getattr() with defaults for safety
    start_date = pd.Timestamp(f"{sim_year}-01-01")
    end_date = pd.Timestamp(f"{sim_year}-12-31")
    comp_increase_rate = getattr(year_config, 'annual_compensation_increase_rate', 0.0)
    termination_rate = getattr(year_config, 'annual_termination_rate', 0.0)
    new_hire_termination_rate = getattr(year_config, 'new_hire_termination_rate', termination_rate)
    growth_rate = getattr(year_config, 'annual_growth_rate', 0.0)
    maintain_headcount = getattr(year_config, 'maintain_headcount', False) # Default False if growth used
    initial_base_count = getattr(year_config, 'initial_headcount_for_target', len(current_df)) # Need a way to track base for growth/maintain

    log.info(f"Year {sim_year} Dynamics Params: CompRate={comp_increase_rate:.2%}, TermRate={termination_rate:.2%}, NHTermRate={new_hire_termination_rate:.2%}, MaintainHC={maintain_headcount}, GrowthRate={growth_rate:.2%}")

    # --- ML Model Setup ---
    model_path = getattr(year_config, 'ml_model_path', '')
    features_path = getattr(year_config, 'model_features_path', '')
    projection_model, feature_cols = None, []
    if model_path and features_path:
        ml_pair = try_load_ml_model(model_path, features_path)
        if ml_pair:
             projection_model, feature_cols = ml_pair
             log.info(f"Loaded ML turnover model from {model_path} with {len(feature_cols)} features.")
    use_ml = getattr(year_config, 'use_ml_turnover', False) and projection_model is not None
    log.info(f"Using ML for turnover: {use_ml}")

    # --- Baseline Salaries (Needed for onboarding bump/new hire sampling if applicable) ---
    # This might need to be loaded once and passed in, or handled differently
    # For now, assume it might be needed by helpers called below
    # Re-evaluate if baseline_hire_salaries is truly needed here or just within hiring.py
    baseline_hire_salaries = getattr(year_config, 'baseline_hire_salary_distribution', pd.Series(dtype=float)) # Example placeholder
    # if baseline_hire_salaries.empty:
    #     log.warning("Baseline hire salary distribution not provided or empty.")


    # --- Initial State Check ---
    if current_df.empty:
        log.warning(f"Starting dynamics for year {sim_year} with empty population.")
        # Decide behavior: return empty or attempt hiring? Let's attempt hiring.
        pass # Continue to hiring step

    # Ensure essential columns exist
    required_cols = [EMP_HIRE_DATE, EMP_GROSS_COMP, EMP_BIRTH_DATE]
    for col in required_cols:
         if col not in current_df.columns:
             log.error(f"Missing required column in current_df: '{col}'. Aborting dynamics for year {sim_year}.")
             return pd.DataFrame() # Return empty DataFrame on critical error
    if EMP_TERM_DATE not in current_df.columns:
         log.warning(f"Column '{EMP_TERM_DATE}' not found. Creating with NaT.")
         current_df[EMP_TERM_DATE] = pd.NaT

    # Initialize prev_term_salaries as Series (needed for apply_turnover)
    # Use salaries from the input DataFrame (active at start of year)
    prev_term_salaries = current_df[EMP_GROSS_COMP].dropna()


    # --- Yearly Dynamics Steps ---
    year_start_headcount = len(current_df)
    log.info(f"Year {sim_year} Start: Headcount={year_start_headcount}")

    # Calculate Tenure relative to start of current simulation year
    current_df['tenure'] = calculate_tenure(current_df[EMP_HIRE_DATE], start_date)
    log.debug(f"Year {sim_year}: Recalculated tenure.")

    # 1. Comp Bump (using compensation.py helper)
    log.debug(f"Year {sim_year}: Applying compensation bump...")
    current_df = apply_comp_bump(
        df=current_df,
        comp_col=EMP_GROSS_COMP,
        # Pass relevant config sections if needed by the helper
        # dist=year_config.get('second_year_compensation_dist', {}), # Example
        rate=comp_increase_rate,
        rng=rng_bump,
        # sampler=DefaultSalarySampler(), # If sampler logic is complex
        log=log
    )

    # --- 2. Experienced Employee Terminations --- #
    log.info(f"Year {sim_year}: Applying experienced termination rate ({termination_rate:.2%}) to initial {len(current_df)} employees.")
    # Apply annual_termination_rate ONLY to the population at the start of the year
    df_after_exp_term = apply_turnover(
        df=current_df.copy(), # Operate on a copy to avoid modifying original before counting
        hire_col=EMP_HIRE_DATE,
        probs_or_rate=termination_rate, # Use the annual rate for existing employees
        start_date=start_date,
        end_date=end_date,
        rng=rng_term,
        prev_term_salaries=prev_term_salaries, # Pass salaries from start of year
        log=log
    )

    # Identify who was terminated IN THIS STEP
    exp_terminated_mask = df_after_exp_term[EMP_TERM_DATE].between(start_date, end_date, inclusive='both')
    num_exp_terminated = exp_terminated_mask.sum()
    log.info(f"Year {sim_year}: Experienced Terminations applied = {num_exp_terminated}")

    # --- 3. Identify Survivors from the initial cohort --- #
    survivor_mask = ~exp_terminated_mask # Survivors are those NOT terminated in the above step
    survivor_df = df_after_exp_term[survivor_mask].copy()
    num_survivors = len(survivor_df)
    log.info(f"Year {sim_year}: Survivors from initial cohort = {num_survivors}")

    # Store term dates from this step before adding new hires
    terminated_this_step_df = df_after_exp_term[exp_terminated_mask].copy()

    # --- 4. New Hires --- #
    # Determine hires needed based on ACTUAL survivors
    if maintain_headcount:
        initial_headcount = len(current_df) # Headcount at the very start of the year
        target_count = getattr(year_config, 'initial_headcount_for_target', initial_headcount)
        log.debug(f"Year {sim_year}: MaintainHC=True. Target={target_count}, Survivors={num_survivors}")
        needed = max(0, target_count - num_survivors)
    else:  # maintain_headcount is False
        year_start_headcount = len(current_df) # Active employees at the start of this sim_year
        growth_rate = getattr(year_config, 'annual_growth_rate', 0.0)
        # new_hire_termination_rate is already defined from config (around line 91 in the full file)

        # Target number of active employees at the END of sim_year to achieve the year-over-year growth_rate
        target_active_eoy = math.ceil(year_start_headcount * (1 + growth_rate))
        log.info(f"Year {sim_year}: MaintainHC=False. YearStartHC={year_start_headcount}, ConfigGrowthRate={growth_rate:.2%}, TargetActiveEOY={target_active_eoy}")

        # Net hires needed (after NH term) to reach TargetActiveEOY, considering survivors from the initial cohort
        net_hires_needed_after_nh_term = max(0, target_active_eoy - num_survivors)
        log.info(f"Year {sim_year}: SurvivorsFromInitialCohort={num_survivors}, NetHiresNeededAfterNHTerm={net_hires_needed_after_nh_term} to reach TargetActiveEOY.")

        # Gross up the net_hires_needed to account for new hire terminations
        calculated_gross_hires = 0
        if 0 < new_hire_termination_rate < 1:
            divisor = (1 - new_hire_termination_rate)
            calculated_gross_hires = math.ceil(net_hires_needed_after_nh_term / divisor)
            log.info(f"Year {sim_year}: Adjusted for NHTermRate ({new_hire_termination_rate:.2%}). CalculatedGrossHires={calculated_gross_hires} (from NetHiresNeededAfterNHTerm={net_hires_needed_after_nh_term})")
        elif new_hire_termination_rate >= 1:
            if net_hires_needed_after_nh_term > 0:
                log.warning(f"Year {sim_year}: NHTermRate is {new_hire_termination_rate:.2%} (>=100%) and {net_hires_needed_after_nh_term} net hires are needed. Cannot achieve target. Setting calculated gross hires to 0.")
                calculated_gross_hires = 0
            else: # net_hires_needed_after_nh_term is 0
                calculated_gross_hires = 0
                log.info(f"Year {sim_year}: NHTermRate is {new_hire_termination_rate:.2%}. NetHiresNeededAfterNHTerm is {net_hires_needed_after_nh_term}. Setting calculated gross hires to 0.")
        else: # new_hire_termination_rate is 0 or not a valid positive fraction (e.g. negative)
            calculated_gross_hires = net_hires_needed_after_nh_term
            log.info(f"Year {sim_year}: No adjustment for NHTermRate (rate is {new_hire_termination_rate:.2%}). CalculatedGrossHires={calculated_gross_hires}")
        
        needed = calculated_gross_hires

    log.info(f"Year {sim_year}: Hires needed = {needed}")
    new_hires_df = pd.DataFrame()
    if needed > 0:
        log.info(f"Generating {needed} new hires for {sim_year}.")
        # Use all known IDs (including those terminated this year) to avoid reuse
        all_known_ids = df_after_exp_term.index.tolist()
        new_hires_df = generate_new_hires(
            num_hires=needed,
            hire_year=sim_year,
            scenario_config=year_config,
            existing_ids=all_known_ids,
            rng=rng_nh
        )
        log.info(f"Generated {len(new_hires_df)} new hire records.")

        # --- Apply Onboarding Compensation Bump --- #
        if not new_hires_df.empty:
            # (Onboarding bump logic remains the same)
            logger.debug(f"Year {sim_year}: Applying onboarding comp bump...")
            # Safely get onboarding bump config from year_config.plan_rules
            plan_rules_obj = getattr(year_config, 'plan_rules', None)
            ob_cfg = getattr(plan_rules_obj, 'onboarding_bump', {}) if plan_rules_obj else {}

            # Use getattr to check the 'apply_onboarding_bump' field on the Pydantic model
            if ob_cfg and getattr(ob_cfg, 'apply_onboarding_bump', False):
                logger.info(f"Applying onboarding bump with config: {ob_cfg}")
                new_hires_df = apply_onboarding_bump(
                    df=new_hires_df,
                    comp_col=EMP_GROSS_COMP,
                    ob_cfg=ob_cfg,
                    baseline_hire_salaries=baseline_hire_salaries, # Pass baseline salaries
                    rng=rng_nh, # Can reuse NH RNG or use bump RNG
                    log=log
                )

    # --- 5. Apply New Hire Termination Rate --- #
    if new_hire_termination_rate > 0 and not new_hires_df.empty:
        log.debug(f"Applying new hire termination rate ({new_hire_termination_rate:.1%}) to {len(new_hires_df)} new hires...")
        new_hires_df = apply_turnover(
            df=new_hires_df,
            hire_col=EMP_HIRE_DATE,
            probs_or_rate=new_hire_termination_rate, # Apply specific NH rate
            start_date=start_date, # Terminations happen *during* sim_year
            end_date=end_date,
            rng=rng_term, # Can reuse term RNG
            # sampler=DefaultSalarySampler(), # If needed
            prev_term_salaries=prev_term_salaries, # Use original term salaries base
            log=log
        )
        num_nh_termed = new_hires_df[EMP_TERM_DATE].notna().sum()
        log.info(f"Year {sim_year}: {num_nh_termed} new hires terminated within the year.")

    # 6. Combine Survivors and New Hires
    # This represents the population at year end *before* considering NH terminations
    df_for_terms = df_after_exp_term.copy()
    if 'new_hires_df' in locals() and not new_hires_df.empty:
        dynamics_output_df = pd.concat([df_for_terms, new_hires_df], ignore_index=True)
    else:
        # If no new hires were made (e.g., current_df was empty and no growth, or negative growth led to 0 hires_needed)
        dynamics_output_df = df_for_terms.copy()
    
    log.info(f"Year {sim_year}: Combined survivors and new hires. Headcount = {len(dynamics_output_df)}")

    # 7. Final DataFrame for the Year (to be passed to rules engine)
    # This DF now includes survivors, new hires, and reflects all terminations for the year
    final_dynamics_df = dynamics_output_df.copy()

    # --- LOGGING: Track Terminations of Recent Hires (Optional Refined Log) ---
    # This can be done here or in reporting module
    terminated_in_year_mask_final = final_dynamics_df[EMP_TERM_DATE].between(start_date, end_date, inclusive='both')
    terminated_ids_this_year = final_dynamics_df.loc[terminated_in_year_mask_final, EMP_GROSS_COMP]
    new_hires_terminated_count = 0
    if not terminated_ids_this_year.empty:
        terminated_df_subset = final_dynamics_df[final_dynamics_df[EMP_GROSS_COMP].isin(terminated_ids_this_year.tolist())]
        # Check if hired in or after the overall simulation start_year
        hired_during_sim_mask = terminated_df_subset[EMP_HIRE_DATE].dt.year >= getattr(year_config, 'start_year', sim_year) # Use global start year
        new_hires_terminated_count = hired_during_sim_mask.sum()
        log.info(f"Year {sim_year}: Total {terminated_in_year_mask_final.sum()} terminations. {new_hires_terminated_count} were hired >= {getattr(year_config, 'start_year', sim_year)}.")
    else:
        log.info(f"Year {sim_year}: No employees terminated in this period.")
    # --- End Termination Tracking Log ---


    log.info(f"--- Finished Dynamics for Simulation Year {sim_year} ---")
    return final_dynamics_df

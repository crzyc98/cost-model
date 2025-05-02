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

    # 2. Terminations (using termination.py helper)
    log.debug(f"Year {sim_year}: Applying terminations (ML={use_ml})...")
    if use_ml:
        log.debug("Predicting ML turnover probabilities...")
        # Ensure current_df has features needed by the model
        term_probs = predict_turnover(
            current_df, projection_model, feature_cols, random_state=rng_ml
        )
        term_rate_or_probs = term_probs
    else:
         # Rule-based: Apply different rates if configured
         log.debug(f"Applying rule-based turnover (NH Rate={new_hire_termination_rate:.2%}, Other Rate={termination_rate:.2%})")
         is_new_hire_this_year = (current_df[EMP_HIRE_DATE].dt.year == sim_year) # Check if hired *this* year (unlikely before hiring step)
         # More likely: check tenure == 0 or based on a flag
         is_new_hire_tenure_check = current_df['tenure'] < 1.0 # Example: less than 1 year tenure
         term_rate_or_probs = np.where(is_new_hire_tenure_check, new_hire_termination_rate, termination_rate)

    current_df = apply_turnover(
        df=current_df,
        hire_col=EMP_HIRE_DATE,
        probs_or_rate=term_rate_or_probs,
        start_date=start_date, # Terminations happen *during* sim_year
        end_date=end_date,
        rng=rng_term,
        # sampler=DefaultSalarySampler(), # Pass sampler if needed by apply_turnover
        prev_term_salaries=prev_term_salaries, # Pass salaries from start of year
        log=log
    )
    # Update prev_term_salaries for next year's *potential* use (if needed)
    # Note: This captures terms based on start-of-year pop. New hire terms aren't included yet.
    terminated_in_year_mask = current_df[EMP_TERM_DATE].between(start_date, end_date, inclusive='both')
    prev_term_salaries = current_df.loc[terminated_in_year_mask, EMP_GROSS_COMP].dropna()


    # 3. Identify Survivors (those not terminated *yet*)
    survivor_mask = current_df[EMP_TERM_DATE].isna() | (current_df[EMP_TERM_DATE] > end_date)
    survivor_df = current_df[survivor_mask].copy()
    num_survivors = len(survivor_df)
    log.info(f"Year {sim_year}: {num_survivors} survivors after initial termination step.")

    # 4. New Hires (using hiring.py helper)
    # Determine hires needed
    if maintain_headcount:
        target_count = getattr(year_config, 'initial_headcount_for_target', len(current_df))
        log.debug(f"Year {sim_year}: MaintainHC=True. Target={target_count}, Survivors={num_survivors}")
        needed = max(0, target_count - num_survivors)
    else:
        target_count = int(getattr(year_config, 'initial_headcount_for_target', len(current_df)) * (1 + getattr(year_config, 'annual_growth_rate', 0.0)) ** (sim_year - getattr(year_config, 'start_year', sim_year) + 1)) # Cumulative growth
        net_needed = max(0, target_count - num_survivors)
        log.debug(f"Year {sim_year}: MaintainHC=False. Target={target_count}, Survivors={num_survivors}, NetNeeded={net_needed}")
        # Gross up based on *expected* NH term rate? Or handle NH terms separately?
        # Let's assume generate_new_hires creates the target 'needed' and term logic handles NH rate.
        needed = net_needed
        # --- Alternative 'gross up' logic ---
        # if new_hire_termination_rate < 1 and net_needed > 0:
        #     needed = math.ceil(net_needed / (1 - new_hire_termination_rate))
        #     log.debug(f"Year {sim_year}: Grossing up hires needed ({net_needed}) by NH term rate ({new_hire_termination_rate:.1%}) to {needed}")
        # else: needed = net_needed

    log.info(f"Year {sim_year}: Hires needed = {needed}")
    new_hires_df = pd.DataFrame() # Initialize empty
    if needed > 0:
        log.info(f"Generating {needed} new hires for {sim_year}.")
        existing_ids = current_df.index.tolist() # Get all IDs currently in DF (active or termed this year)
        new_hires_df = generate_new_hires(
            num_hires=needed,
            hire_year=sim_year,
            scenario_config=year_config, # Pass the resolved config
            existing_ids=existing_ids,
            rng=rng_nh # Use the NH-specific RNG
        )
        log.debug(f"Generated {len(new_hires_df)} new hire records.")

        # --- Apply Onboarding Compensation Bump for New Hires (if applicable) ---
        if not new_hires_df.empty:
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

    # 5. Combine Survivors and New Hires
    # This represents the population at year end *before* considering NH terminations
    combined_df = pd.concat([survivor_df, new_hires_df], ignore_index=True)
    log.info(f"Year {sim_year}: Combined survivors and new hires. Headcount = {len(combined_df)}")

    # 6. Apply New Hire Termination Rate (Optional Step - depends on desired model flow)
    # If new_hire_termination_rate should apply *after* hires join within the same year:
    if getattr(year_config, 'new_hire_termination_rate', 0.0) > 0 and not new_hires_df.empty:
         log.debug(f"Applying new hire termination rate ({getattr(year_config, 'new_hire_termination_rate', 0.0):.1%}) to {len(new_hires_df)} new hires...")
         new_hire_mask = combined_df.index.isin(new_hires_df.index) # Identify new hires in combined DF
         # Create a rate array specific to new hires
         nh_term_rate_array = np.where(new_hire_mask, getattr(year_config, 'new_hire_termination_rate', 0.0), 0.0) # Only apply rate to new hires

         combined_df = apply_turnover(
             df=combined_df,
             hire_col=EMP_HIRE_DATE,
             probs_or_rate=nh_term_rate_array, # Apply specific NH rate
             start_date=start_date, # Terminations happen *during* sim_year
             end_date=end_date,
             rng=rng_term, # Can reuse term RNG
             # sampler=DefaultSalarySampler(), # If needed
             prev_term_salaries=prev_term_salaries, # Use original term salaries base
             log=log
         )
         num_nh_termed = combined_df.loc[new_hire_mask, EMP_TERM_DATE].notna().sum()
         log.info(f"Year {sim_year}: {num_nh_termed} new hires terminated within the year.")

    # 7. Final DataFrame for the Year (to be passed to rules engine)
    # This DF now includes survivors, new hires, and reflects all terminations for the year
    final_dynamics_df = combined_df.copy()

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

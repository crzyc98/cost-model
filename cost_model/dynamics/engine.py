# cost_model/dynamics/engine.py
"""
Engine for running population dynamics simulation steps (Phase 1).
Orchestrates compensation changes, terminations, and hiring.
"""

import logging
import math
import pandas as pd
import numpy as np
from typing import Union, Optional, Mapping, Any, Dict, List

# Use relative imports within the cost_model package
try:
    from ..utils.date_utils import calculate_tenure, get_random_dates_in_year
    from .hiring import generate_new_hires
    from .compensation import apply_comp_bump, apply_onboarding_bump
    from .termination import apply_turnover
    from ..ml.ml_utils import try_load_ml_model, predict_turnover
    from ..utils.columns import EMP_ID, EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP
    from .sampling.salary import SalarySampler, DefaultSalarySampler
except ImportError as e:
    print(f"Error importing dynamics components from engine.py: {e}")
    EMP_ID, EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP = 'employee_id', 'employee_hire_date', 'employee_termination_date', 'employee_birth_date', 'employee_gross_compensation'
    def generate_new_hires(*args, **kwargs): raise NotImplementedError("generate_new_hires not imported")
    def apply_comp_bump(*args, **kwargs): raise NotImplementedError("apply_comp_bump not imported")
    def apply_onboarding_bump(*args, **kwargs): raise NotImplementedError("apply_onboarding_bump not imported")
    def apply_turnover(*args, **kwargs): raise NotImplementedError("apply_turnover not imported")
    def try_load_ml_model(*args, **kwargs): return None, None
    def predict_turnover(*args, **kwargs): raise NotImplementedError("predict_turnover not imported")
    def get_random_dates_in_year(*args, **kwargs): raise NotImplementedError("get_random_dates_in_year not imported")
    class DefaultSalarySampler: pass

logger = logging.getLogger(__name__)

def run_dynamics_for_year(
    current_df: pd.DataFrame,
    year_config: Mapping[str, Any],
    sim_year: int,
    parent_logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    log = parent_logger or logger
    log.info(f"--- Running Dynamics for Simulation Year {sim_year} ---")

    # --- RNG Setup ---
    seed = getattr(year_config, 'random_seed', None)
    if seed is not None:
        log.debug(f"Initializing dynamics RNGs with seed: {seed}")
        master_ss = np.random.SeedSequence(seed)
        ss_bump, ss_exp_term, ss_nh_gen, ss_nh_term, ss_ml = master_ss.spawn(5)
        rng_bump = np.random.default_rng(ss_bump)
        rng_exp_term = np.random.default_rng(ss_exp_term)
        rng_nh_gen = np.random.default_rng(ss_nh_gen)
        rng_nh_term = np.random.default_rng(ss_nh_term)
        rng_ml = np.random.default_rng(ss_ml)
    else:
        log.warning("No random seed provided for dynamics. Results may not be reproducible.")
        rng_bump, rng_exp_term, rng_nh_gen, rng_nh_term, rng_ml = (np.random.default_rng() for _ in range(5))
    log.debug("Dynamics RNGs initialized.")

    # --- Config Parameters ---
    comp_increase_rate = getattr(year_config, 'annual_compensation_increase_rate', 0.0)
    termination_rate = getattr(year_config, 'annual_termination_rate', 0.0)
    new_hire_termination_rate = getattr(year_config, 'new_hire_termination_rate', termination_rate)
    new_hire_term_rate_safety_margin = getattr(year_config, 'new_hire_termination_rate_safety_margin', 0.0)
    growth_rate = getattr(year_config, 'annual_growth_rate', 0.0)
    maintain_headcount = getattr(year_config, 'maintain_headcount', False)
    use_expected_attrition = getattr(year_config, 'use_expected_attrition', False)

    log.info(
        f"Year {sim_year} Dynamics Params: CompRate={comp_increase_rate:.2%}, TermRate={termination_rate:.2%}, "
        f"NHTermRate={new_hire_termination_rate:.2%}, NHTermSafetyMargin={new_hire_term_rate_safety_margin:.2%}, "
        f"GrowthRate={growth_rate:.2%}, MaintainHC={maintain_headcount}, UseExpectedAttrition={use_expected_attrition}"
    )

    # --- Prepare df_processed, Set Index, and Calculate n0_exp ---
    df_processed = current_df.copy()
    start_date = pd.Timestamp(f"{sim_year}-01-01")
    end_date = pd.Timestamp(f"{sim_year}-12-31")

    # 1. Ensure EMP_TERM_DATE column exists
    if EMP_TERM_DATE not in df_processed.columns and not df_processed.empty:
         log.warning(f"Column '{EMP_TERM_DATE}' not found in input DataFrame for year {sim_year}. Creating with NaT.")
         df_processed[EMP_TERM_DATE] = pd.NaT
    elif df_processed.empty and EMP_TERM_DATE not in df_processed.columns:
         df_processed[EMP_TERM_DATE] = pd.NaT

    # 2. Handle EMP_ID and set index for df_processed (CRITICAL: DO THIS EARLY)
    if not df_processed.empty:
        if EMP_ID not in df_processed.columns:
            log.error(f"CRITICAL: Required column '{EMP_ID}' not found in df_processed. Indexing will be integer-based. This may lead to issues.")
        elif df_processed.index.name != EMP_ID: # If EMP_ID column exists and is not already the index
            if df_processed[EMP_ID].is_unique:
                log.debug(f"Setting '{EMP_ID}' as index for df_processed.")
                df_processed = df_processed.set_index(EMP_ID, drop=False) # Index is now EMP_ID
            else:
                log.error(f"CRITICAL: Column '{EMP_ID}' is not unique. Cannot set as index. Retaining original index. This may lead to issues.")
    
    # 3. Calculate n0_exp (truly active employees at year start) using the now stable-indexed df_processed
    exp_attrition_eligible_df = pd.DataFrame(columns=df_processed.columns, index=pd.Index([], name=df_processed.index.name))
    n0_exp = 0
    if not df_processed.empty:
        active_at_year_start_mask = (df_processed[EMP_TERM_DATE].isna()) | \
                                    (df_processed[EMP_TERM_DATE] >= start_date)
        exp_attrition_eligible_df = df_processed[active_at_year_start_mask] # Index will be of the same type as df_processed.index
        n0_exp = len(exp_attrition_eligible_df)
    log.info(f"Year {sim_year}: Total records loaded into dynamics = {len(current_df)}, Active employees eligible for exp. attrition (n0_exp) = {n0_exp}")

    # 4. Determine the base headcount for calculating growth/maintenance targets
    if n0_exp == 0 and getattr(year_config, 'initial_headcount_for_target', 0) > 0:
        year_start_headcount_for_calc = getattr(year_config, 'initial_headcount_for_target', 0)
        log.info(f"Year {sim_year}: Active headcount (n0_exp) is 0. Using initial_headcount_for_target ({year_start_headcount_for_calc}) as base for calculations.")
    else:
        year_start_headcount_for_calc = n0_exp
        if year_start_headcount_for_calc != len(current_df): # len(current_df) is total records from input for this year
             log.info(f"Year {sim_year}: Using active headcount (n0_exp = {n0_exp}) as base for growth/maintenance calculations. (Total records from input: {len(current_df)}).")
        else:
             log.info(f"Year {sim_year}: Using active headcount (n0_exp = {n0_exp}) as base for growth/maintenance calculations.")
    
    # --- Yearly Dynamics Log Start ---
    log.info(f"Year {sim_year} Start: Total Records Input={len(current_df)}, Active Base for Growth Calc={year_start_headcount_for_calc}")

    # 5. Apply comp bump and tenure to active employees
    if n0_exp > 0:
        # exp_attrition_eligible_df is already the slice of active employees from the correctly indexed df_processed.
        active_employees_work_df = exp_attrition_eligible_df.copy() # Work on a copy

        active_employees_work_df['tenure'] = calculate_tenure(active_employees_work_df[EMP_HIRE_DATE], start_date)
        log.debug(f"Year {sim_year}: Recalculated tenure for {len(active_employees_work_df)} active employees.")
        
        active_employees_work_df = apply_comp_bump(
            df=active_employees_work_df,
            comp_col=EMP_GROSS_COMP,
            rate=comp_increase_rate,
            rng=rng_bump,
            log=log
        )
        log.debug(f"Year {sim_year}: Applied compensation bump to active employees.")
        
        df_processed.update(active_employees_work_df) # Update the main DataFrame

    # --- 6. Experienced Employee Terminations ---
    terminated_exp_indices: List[Any] = []
    if n0_exp > 0: # n0_exp is the count from exp_attrition_eligible_df
        if use_expected_attrition:
            log.info(f"Year {sim_year}: Applying DETERMINISTIC experienced termination rate ({termination_rate:.2%}) to {n0_exp} eligible employees.")
            n_term_exp = math.ceil(n0_exp * termination_rate)
            if n_term_exp > n0_exp: n_term_exp = n0_exp
            
            if n_term_exp > 0:
                indices_to_choose_from = exp_attrition_eligible_df.index # This index is from the correctly indexed df_processed
                terminated_exp_indices = rng_exp_term.choice(indices_to_choose_from, size=int(n_term_exp), replace=False).tolist()
                term_dates_for_exp = get_random_dates_in_year(sim_year, count=len(terminated_exp_indices), rng=rng_exp_term, day_of_month=15)
                df_processed.loc[terminated_exp_indices, EMP_TERM_DATE] = term_dates_for_exp
            log.info(f"Year {sim_year}: Deterministically terminated {len(terminated_exp_indices)} experienced employees.")
        else: # Stochastic
            log.info(f"Year {sim_year}: Applying STOCHASTIC experienced termination rate ({termination_rate:.2%}) to {n0_exp} eligible employees.")
            # Operate on a copy of the eligible portion (exp_attrition_eligible_df)
            temp_eligible_df_for_term = exp_attrition_eligible_df.copy()
            
            temp_eligible_df_for_term = apply_turnover(
                df=temp_eligible_df_for_term,
                hire_col=EMP_HIRE_DATE,
                probs_or_rate=termination_rate,
                start_date=start_date,
                end_date=end_date,
                rng=rng_exp_term,
                prev_term_salaries=temp_eligible_df_for_term[EMP_GROSS_COMP].dropna() if not temp_eligible_df_for_term.empty else pd.Series(dtype=float),
                log=log
            )
            df_processed.update(temp_eligible_df_for_term) # Update main df_processed with results from the slice
            
            # Identify who was terminated stochastically from the original eligible indices
            terminated_this_step_mask = (df_processed.loc[exp_attrition_eligible_df.index, EMP_TERM_DATE].notna()) & \
                                        (df_processed.loc[exp_attrition_eligible_df.index, EMP_TERM_DATE] >= start_date) & \
                                        (df_processed.loc[exp_attrition_eligible_df.index, EMP_TERM_DATE] <= end_date)
            terminated_exp_indices = df_processed.loc[exp_attrition_eligible_df.index][terminated_this_step_mask].index.tolist()
            log.info(f"Year {sim_year}: Stochastically terminated {len(terminated_exp_indices)} experienced employees.")
    else:
        log.info(f"Year {sim_year}: No experienced employees eligible for termination (n0_exp is 0).")

    num_survivors_initial_cohort = 0
    if n0_exp > 0:
        survivor_mask_on_eligible = (df_processed.loc[exp_attrition_eligible_df.index, EMP_TERM_DATE].isna()) | \
                                    (df_processed.loc[exp_attrition_eligible_df.index, EMP_TERM_DATE] > end_date)
        num_survivors_initial_cohort = survivor_mask_on_eligible.sum()
    log.info(f"Year {sim_year}: Survivors from initial active cohort (n0_exp={n0_exp}, after experienced attrition): {num_survivors_initial_cohort}")

    # --- 7. New Hires Calculation ---
    if maintain_headcount:
        target_eoy_headcount = year_start_headcount_for_calc
    else: # Growth rate
        target_eoy_headcount = math.ceil(year_start_headcount_for_calc * (1 + growth_rate))
    log.info(f"Year {sim_year}: Target EOY headcount (based on active start {year_start_headcount_for_calc} and growth {growth_rate:.2%}) = {target_eoy_headcount}")
    
    needed_active_new_hires_eoy = max(0, target_eoy_headcount - num_survivors_initial_cohort)
    log.info(f"Year {sim_year}: Needed active new hires at EOY = {needed_active_new_hires_eoy} (Target: {target_eoy_headcount}, Survivors: {num_survivors_initial_cohort})")

    hires_to_make = 0
    if needed_active_new_hires_eoy > 0:
        current_nh_term_rate_for_calc = new_hire_termination_rate
        log_msg_suffix = "deterministic NH term assumption"
        if not use_expected_attrition:
            current_nh_term_rate_for_calc = min(new_hire_termination_rate + new_hire_term_rate_safety_margin, 0.99)
            log_msg_suffix = "stochastic NH term assumption w/ safety margin"
        
        if current_nh_term_rate_for_calc >= 1.0:
            log.warning(f"Year {sim_year}: NH term rate for calculation ({current_nh_term_rate_for_calc:.2%}) is >= 100%. Cannot achieve {needed_active_new_hires_eoy} active new hires by hiring.")
        else:
            hires_to_make = math.ceil(needed_active_new_hires_eoy / (1 - current_nh_term_rate_for_calc))
        log.info(f"Year {sim_year}: Calculated {hires_to_make} hires needed ({log_msg_suffix} using rate {current_nh_term_rate_for_calc:.2%}) to get {needed_active_new_hires_eoy} active new hires at EOY.")
    else:
        log.info(f"Year {sim_year}: No new hires needed as survivors ({num_survivors_initial_cohort}) meet/exceed EOY target ({target_eoy_headcount}).")
    
    hires_to_make = max(0, int(hires_to_make))

    new_hires_df = pd.DataFrame()
    if hires_to_make > 0:
        log.info(f"Generating {hires_to_make} new hires for {sim_year}.")
        existing_ids_list = []
        if not df_processed.empty:
            if EMP_ID in df_processed.columns:
                existing_ids_list = df_processed[EMP_ID].unique().tolist()
            else:
                log.warning(f"'{EMP_ID}' column not in df_processed, using its index for existing_ids list generation.")
                existing_ids_list = df_processed.index.unique().tolist()
        
        new_hires_df = generate_new_hires(
            num_hires=hires_to_make,
            hire_year=sim_year,
            scenario_config=year_config,
            existing_ids=existing_ids_list,
            rng=rng_nh_gen,
            id_col_name=EMP_ID
        )
        if not new_hires_df.empty:
            log.info(f"Generated {len(new_hires_df)} new hire records.")
            if EMP_ID not in new_hires_df.columns:
                 log.error(f"CRITICAL: generate_new_hires did not produce column '{EMP_ID}'.")
                 return pd.DataFrame()
            if not new_hires_df[EMP_ID].is_unique:
                 log.error(f"CRITICAL: Generated new hire '{EMP_ID}' values are not unique.")
            
            new_hires_df = new_hires_df.set_index(EMP_ID, drop=False)
            if EMP_TERM_DATE not in new_hires_df.columns:
                new_hires_df[EMP_TERM_DATE] = pd.NaT
            # Onboarding bump logic can be added here
        else:
            log.warning(f"generate_new_hires was expected to create {hires_to_make} hires but returned an empty DataFrame.")
            hires_to_make = 0
            
    # --- 8. Apply New Hire Termination Rate ---
    terminated_nh_indices: List[Any] = []
    if hires_to_make > 0 and not new_hires_df.empty and new_hire_termination_rate > 0:
        n0_nh = len(new_hires_df)
        # ... (deterministic or stochastic NH term logic, same as before) ...
        if use_expected_attrition:
            log.info(f"Year {sim_year}: Applying DETERMINISTIC new hire termination rate ({new_hire_termination_rate:.2%}) to {n0_nh} new hires.")
            n_term_nh = math.ceil(n0_nh * new_hire_termination_rate)
            if n_term_nh > n0_nh: n_term_nh = n0_nh
            if n_term_nh > 0:
                terminated_nh_indices = rng_nh_term.choice(new_hires_df.index, size=int(n_term_nh), replace=False).tolist()
                term_dates_for_nh = get_random_dates_in_year(sim_year, count=len(terminated_nh_indices), rng=rng_nh_term, day_of_month=28)
                new_hires_df.loc[terminated_nh_indices, EMP_TERM_DATE] = term_dates_for_nh
            log.info(f"Year {sim_year}: Deterministically terminated {len(terminated_nh_indices)} new hires.")
        else: # Stochastic
            log.info(f"Year {sim_year}: Applying STOCHASTIC new hire termination rate ({new_hire_termination_rate:.2%}) to {n0_nh} new hires.")
            temp_nh_df = new_hires_df.copy()
            temp_nh_df = apply_turnover(
                df=temp_nh_df,
                hire_col=EMP_HIRE_DATE,
                probs_or_rate=new_hire_termination_rate,
                start_date=start_date,
                end_date=end_date,
                rng=rng_nh_term,
                prev_term_salaries=temp_nh_df[EMP_GROSS_COMP].dropna() if not temp_nh_df.empty else pd.Series(dtype=float),
                log=log
            )
            new_hires_df.update(temp_nh_df)
            terminated_nh_this_step_mask = (new_hires_df[EMP_TERM_DATE].notna()) & \
                                           (new_hires_df[EMP_TERM_DATE] >= start_date) & \
                                           (new_hires_df[EMP_TERM_DATE] <= end_date)
            terminated_nh_indices = new_hires_df[terminated_nh_this_step_mask].index.tolist()
            log.info(f"Year {sim_year}: Stochastically terminated {len(terminated_nh_indices)} new hires.")

    # --- 9. Combine Population ---
    final_df_list = []
    if not df_processed.empty:
        final_df_list.append(df_processed.reset_index(drop=True)) 
    if not new_hires_df.empty:
        final_df_list.append(new_hires_df.reset_index(drop=True))

    if not final_df_list:
        final_dynamics_df = pd.DataFrame()
        log.warning(f"Year {sim_year}: Population is empty at end of dynamics.")
    else:
        all_cols = set()
        for df_item in final_df_list: all_cols.update(df_item.columns)
        processed_df_list = []
        for df_item in final_df_list:
            for col_to_add in all_cols - set(df_item.columns): df_item[col_to_add] = pd.NA
            processed_df_list.append(df_item[list(all_cols)])
        final_dynamics_df = pd.concat(processed_df_list, ignore_index=True)

    # --- Final Logging ---
    final_eoy_active_mask = (final_dynamics_df[EMP_TERM_DATE].isna()) | (final_dynamics_df[EMP_TERM_DATE] > end_date)
    final_eoy_active_headcount = final_eoy_active_mask.sum()
    calculated_target_eoy = target_eoy_headcount if 'target_eoy_headcount' in locals() else 'N/A'
    log.info(
        f"Year {sim_year}: Combined population. Total records = {len(final_dynamics_df)}. "
        f"Active EOY = {final_eoy_active_headcount} (Calculated Target EOY for hiring: {calculated_target_eoy})"
    )
    total_terminated_this_year_mask = (final_dynamics_df[EMP_TERM_DATE].notna()) & \
                                      (final_dynamics_df[EMP_TERM_DATE] >= start_date) & \
                                      (final_dynamics_df[EMP_TERM_DATE] <= end_date)
    num_total_terminated_this_year = total_terminated_this_year_mask.sum()
    log.info(f"Year {sim_year}: Total terminations recorded in {sim_year} = {num_total_terminated_this_year} (Exp: {len(terminated_exp_indices)}, NH: {len(terminated_nh_indices)})")

    log.info(f"--- Finished Dynamics for Simulation Year {sim_year} ---")
    return final_dynamics_df
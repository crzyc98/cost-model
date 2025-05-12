# cost_model/dynamics/engine.py
"""
Engine for running population dynamics simulation steps (Phase 1).
Orchestrates compensation changes, terminations, and hiring.

## QuickStart

To run workforce dynamics simulations programmatically:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from cost_model.dynamics.engine import run_dynamics_for_year

# Create a sample initial workforce snapshot
initial_snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
    'employee_role': ['Engineer', 'Manager', 'Engineer', 'Analyst'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0, 55000.0],
    'active': [True, True, True, True],
    'employee_hire_date': [
        pd.Timestamp('2024-06-01'),
        pd.Timestamp('2023-03-15'),
        pd.Timestamp('2025-01-10'),
        pd.Timestamp('2024-11-05')
    ],
    'employee_birth_date': [
        pd.Timestamp('1990-05-12'),
        pd.Timestamp('1985-08-23'),
        pd.Timestamp('1992-11-30'),
        pd.Timestamp('1995-03-15')
    ],
    'employee_termination_date': [None, None, None, None],
    'tenure_band': ['0-1', '1-3', '0-1', '0-1']
}).set_index('employee_id')

# Define simulation year and configuration
sim_year = 2025

# Create a configuration for the simulation year
year_config = {
    # Termination settings
    'term_rate': 0.12,                  # 12% annual termination rate
    'new_hire_term_rate': 0.25,         # 25% new hire termination rate
    'use_ml_turnover': False,           # Don't use ML for turnover prediction
    
    # Compensation settings
    'comp_raise_pct': 0.03,             # 3% annual compensation increase
    'cola_pct': 0.02,                   # 2% cost of living adjustment
    'promotion_pct': 0.10,              # 10% of employees get promoted
    'promotion_raise_pct': 0.15,        # 15% raise for promotions
    
    # Hiring settings
    'headcount_target': 5,              # Target 5 employees by end of year
    'headcount_floor': 3,               # Minimum 3 employees
    'new_hire_roles': {                 # Distribution of new hire roles
        'Engineer': 0.6,
        'Analyst': 0.3,
        'Manager': 0.1
    },
    'new_hire_comp': {                  # Base compensation by role
        'Engineer': 70000,
        'Analyst': 55000,
        'Manager': 90000
    }
}

# Run the dynamics simulation for the year
updated_snapshot, events_df = run_dynamics_for_year(
    current_df=initial_snapshot,
    year_config=year_config,
    sim_year=sim_year
)

# Analyze the results
print(f"Initial workforce: {len(initial_snapshot)} employees")
print(f"Updated workforce: {len(updated_snapshot)} employees")
print(f"Active employees: {updated_snapshot['active'].sum()}")

# Analyze terminations
terminated = updated_snapshot[~updated_snapshot['active']]
print(f"\nTerminated employees: {len(terminated)}")
if len(terminated) > 0:
    print("Termination details:")
    for idx, emp in terminated.iterrows():
        term_date = emp['employee_termination_date']
        print(f"  {idx}: Terminated on {term_date.strftime('%Y-%m-%d')}")

# Analyze new hires
new_hires = updated_snapshot[updated_snapshot.index.str.startswith('NH_')]
print(f"\nNew hires: {len(new_hires)}")
if len(new_hires) > 0:
    print("New hire details:")
    for idx, emp in new_hires.iterrows():
        hire_date = emp['employee_hire_date']
        role = emp['employee_role']
        comp = emp['employee_gross_compensation']
        print(f"  {idx}: {role} hired on {hire_date.strftime('%Y-%m-%d')} at ${comp:,.2f}")

# Analyze compensation changes
if 'employee_gross_compensation' in initial_snapshot.columns:
    # Only compare employees present in both snapshots
    common_employees = set(initial_snapshot.index) & set(updated_snapshot.index)
    common_employees = [emp for emp in common_employees if emp in initial_snapshot.index and emp in updated_snapshot.index]
    
    if common_employees:
        initial_comp = initial_snapshot.loc[common_employees, 'employee_gross_compensation']
        updated_comp = updated_snapshot.loc[common_employees, 'employee_gross_compensation']
        
        # Calculate changes
        comp_change = updated_comp - initial_comp
        comp_pct_change = (comp_change / initial_comp) * 100
        
        print(f"\nCompensation changes for existing employees:")
        print(f"  Average increase: ${comp_change.mean():,.2f} ({comp_pct_change.mean():.1f}%)")
        print(f"  Total compensation change: ${comp_change.sum():,.2f}")

# Analyze events
print(f"\nGenerated {len(events_df)} events:")
event_counts = events_df['event_type'].value_counts()
for event_type, count in event_counts.items():
    print(f"  {event_type}: {count}")

# Save the results
output_dir = Path('output/dynamics')
output_dir.mkdir(parents=True, exist_ok=True)

# Save the updated snapshot
updated_snapshot.to_parquet(output_dir / f'snapshot_{sim_year}.parquet')

# Save the events
events_df.to_parquet(output_dir / f'events_{sim_year}.parquet')
```

This demonstrates how to run a workforce dynamics simulation for a year, including terminations, compensation changes, and new hires.
"""

import logging
import math
import pandas as pd
import numpy as np
from typing import Optional, Mapping, Any, List

# Use relative imports within the cost_model package
try:
    from ..utils.date_utils import calculate_tenure, get_random_dates_in_year
    from .hiring import generate_new_hires
    from .compensation import apply_comp_bump, apply_onboarding_bump
    from .termination import apply_turnover
    from ..ml.ml_utils import try_load_ml_model, predict_turnover
    from ..utils.columns import (
        EMP_ID,
        EMP_HIRE_DATE,
        EMP_TERM_DATE,
        EMP_BIRTH_DATE,
        EMP_GROSS_COMP,
    )
    from .sampling.salary import DefaultSalarySampler
except ImportError as e:
    print(f"Error importing dynamics components from engine.py: {e}")
    EMP_ID, EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP = (
        "employee_id",
        "employee_hire_date",
        "employee_termination_date",
        "employee_birth_date",
        "employee_gross_compensation",
    )

    def generate_new_hires(*args, **kwargs):
        raise NotImplementedError("generate_new_hires not imported")

    def apply_comp_bump(*args, **kwargs):
        raise NotImplementedError("apply_comp_bump not imported")

    def apply_onboarding_bump(*args, **kwargs):
        raise NotImplementedError("apply_onboarding_bump not imported")

    def apply_turnover(*args, **kwargs):
        raise NotImplementedError("apply_turnover not imported")

    def try_load_ml_model(*args, **kwargs):
        return None, None

    def predict_turnover(*args, **kwargs):
        raise NotImplementedError("predict_turnover not imported")

    def get_random_dates_in_year(*args, **kwargs):
        raise NotImplementedError("get_random_dates_in_year not imported")

    class DefaultSalarySampler:
        pass


logger = logging.getLogger(__name__)


def run_dynamics_for_year(
    current_df: pd.DataFrame,
    year_config: Mapping[str, Any],
    sim_year: int,
    parent_logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    log = parent_logger or logger
    log.info(f"--- Running Dynamics for Simulation Year {sim_year} ---")

    # --- RNG Setup ---
    seed = getattr(year_config, "random_seed", None)
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
        log.warning(
            "No random seed provided for dynamics. Results may not be reproducible."
        )
        rng_bump, rng_exp_term, rng_nh_gen, rng_nh_term, rng_ml = (
            np.random.default_rng() for _ in range(5)
        )
    log.debug("Dynamics RNGs initialized.")

    # --- Config Parameters ---
    comp_increase_rate = getattr(year_config, "annual_compensation_increase_rate", 0.0)
    termination_rate = getattr(year_config, "annual_termination_rate", 0.0)
    new_hire_termination_rate = getattr(
        year_config, "new_hire_termination_rate", termination_rate
    )
    new_hire_term_rate_safety_margin = getattr(
        year_config, "new_hire_termination_rate_safety_margin", 0.0
    )
    growth_rate = getattr(year_config, "annual_growth_rate", 0.0)
    maintain_headcount = getattr(year_config, "maintain_headcount", False)
    use_expected_attrition = getattr(year_config, "use_expected_attrition", False)

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
        log.warning(
            f"Column '{EMP_TERM_DATE}' not found in input DataFrame for year {sim_year}. Creating with NaT."
        )
        df_processed[EMP_TERM_DATE] = pd.NaT
    elif df_processed.empty and EMP_TERM_DATE not in df_processed.columns:
        df_processed[EMP_TERM_DATE] = pd.NaT

    # 2. Handle EMP_ID and set index for df_processed (CRITICAL: DO THIS EARLY)
    if not df_processed.empty:
        if EMP_ID not in df_processed.columns:
            log.error(
                f"CRITICAL: Required column '{EMP_ID}' not found in df_processed. Indexing will be integer-based. This may lead to issues."
            )
        elif (
            df_processed.index.name != EMP_ID
        ):  # If EMP_ID column exists and is not already the index
            if df_processed[EMP_ID].is_unique:
                log.debug(f"Setting '{EMP_ID}' as index for df_processed.")
                df_processed = df_processed.set_index(
                    EMP_ID, drop=False
                )  # Index is now EMP_ID
            else:
                log.error(
                    f"CRITICAL: Column '{EMP_ID}' is not unique. Cannot set as index. Retaining original index. This may lead to issues."
                )

    # 3. Calculate n0_exp (truly active employees at year start) using the now stable-indexed df_processed
    exp_attrition_eligible_df = pd.DataFrame(
        columns=df_processed.columns, index=pd.Index([], name=df_processed.index.name)
    )
    n0_exp = 0
    if not df_processed.empty:
        active_at_year_start_mask = (df_processed[EMP_TERM_DATE].isna()) | (
            df_processed[EMP_TERM_DATE] >= start_date
        )
        exp_attrition_eligible_df = df_processed[
            active_at_year_start_mask
        ]  # Index will be of the same type as df_processed.index
        n0_exp = len(exp_attrition_eligible_df)
    log.info(
        f"Year {sim_year}: Total records loaded into dynamics = {len(current_df)}, Active employees eligible for exp. attrition (n0_exp) = {n0_exp}"
    )

    # 4. Determine the base headcount for calculating growth/maintenance targets
    if n0_exp == 0 and getattr(year_config, "initial_headcount_for_target", 0) > 0:
        year_start_headcount_for_calc = getattr(
            year_config, "initial_headcount_for_target", 0
        )
        log.info(
            f"Year {sim_year}: Active headcount (n0_exp) is 0. Using initial_headcount_for_target ({year_start_headcount_for_calc}) as base for calculations."
        )
    else:
        year_start_headcount_for_calc = n0_exp
        if year_start_headcount_for_calc != len(
            current_df
        ):  # len(current_df) is total records from input for this year
            log.info(
                f"Year {sim_year}: Using active headcount (n0_exp = {n0_exp}) as base for growth/maintenance calculations. (Total records from input: {len(current_df)})."
            )
        else:
            log.info(
                f"Year {sim_year}: Using active headcount (n0_exp = {n0_exp}) as base for growth/maintenance calculations."
            )

    # --- Yearly Dynamics Log Start ---
    log.info(
        f"Year {sim_year} Start: Total Records Input={len(current_df)}, Active Base for Growth Calc={year_start_headcount_for_calc}"
    )

    # 5. Apply comp bump and tenure to active employees
    if n0_exp > 0:
        # exp_attrition_eligible_df is already the slice of active employees from the correctly indexed df_processed.
        active_employees_work_df = exp_attrition_eligible_df.copy()  # Work on a copy

        active_employees_work_df["tenure"] = calculate_tenure(
            active_employees_work_df[EMP_HIRE_DATE], start_date
        )
        log.debug(
            f"Year {sim_year}: Recalculated tenure for {len(active_employees_work_df)} active employees."
        )

        active_employees_work_df = apply_comp_bump(
            df=active_employees_work_df,
            comp_col=EMP_GROSS_COMP,
            rate=comp_increase_rate,
            rng=rng_bump,
            log=log,
        )
        log.debug(f"Year {sim_year}: Applied compensation bump to active employees.")

        df_processed.update(active_employees_work_df)  # Update the main DataFrame

    # --- 6. Experienced Employee Terminations ---
    terminated_exp_indices: List[Any] = []
    if n0_exp > 0:  # n0_exp is the count from exp_attrition_eligible_df
        if use_expected_attrition:
            log.info(
                f"Year {sim_year}: Applying DETERMINISTIC experienced termination rate ({termination_rate:.2%}) to {n0_exp} eligible employees."
            )
            n_term_exp = math.ceil(n0_exp * termination_rate)
            if n_term_exp > n0_exp:
                n_term_exp = n0_exp

            if n_term_exp > 0:
                indices_to_choose_from = (
                    exp_attrition_eligible_df.index
                )  # This index is from the correctly indexed df_processed
                terminated_exp_indices = rng_exp_term.choice(
                    indices_to_choose_from, size=int(n_term_exp), replace=False
                ).tolist()
                term_dates_for_exp = get_random_dates_in_year(
                    sim_year,
                    count=len(terminated_exp_indices),
                    rng=rng_exp_term,
                    day_of_month=15,
                )
                df_processed.loc[terminated_exp_indices, EMP_TERM_DATE] = (
                    term_dates_for_exp
                )
            log.info(
                f"Year {sim_year}: Deterministically terminated {len(terminated_exp_indices)} experienced employees."
            )
        else:  # Stochastic
            log.info(
                f"Year {sim_year}: Applying STOCHASTIC experienced termination rate ({termination_rate:.2%}) to {n0_exp} eligible employees."
            )
            # Operate on a copy of the eligible portion (exp_attrition_eligible_df)
            temp_eligible_df_for_term = exp_attrition_eligible_df.copy()

            temp_eligible_df_for_term = apply_turnover(
                df=temp_eligible_df_for_term,
                hire_col=EMP_HIRE_DATE,
                probs_or_rate=termination_rate,
                start_date=start_date,
                end_date=end_date,
                rng=rng_exp_term,
                prev_term_salaries=(
                    temp_eligible_df_for_term[EMP_GROSS_COMP].dropna()
                    if not temp_eligible_df_for_term.empty
                    else pd.Series(dtype=float)
                ),
                log=log,
            )
            df_processed.update(
                temp_eligible_df_for_term
            )  # Update main df_processed with results from the slice

            # Identify who was terminated stochastically from the original eligible indices
            terminated_this_step_mask = (
                (
                    df_processed.loc[
                        exp_attrition_eligible_df.index, EMP_TERM_DATE
                    ].notna()
                )
                & (
                    df_processed.loc[exp_attrition_eligible_df.index, EMP_TERM_DATE]
                    >= start_date
                )
                & (
                    df_processed.loc[exp_attrition_eligible_df.index, EMP_TERM_DATE]
                    <= end_date
                )
            )
            terminated_exp_indices = df_processed.loc[exp_attrition_eligible_df.index][
                terminated_this_step_mask
            ].index.tolist()
            log.info(
                f"Year {sim_year}: Stochastically terminated {len(terminated_exp_indices)} experienced employees."
            )
    else:
        log.info(
            f"Year {sim_year}: No experienced employees eligible for termination (n0_exp is 0)."
        )

    num_survivors_initial_cohort = 0
    if n0_exp > 0:
        survivor_mask_on_eligible = (
            df_processed.loc[exp_attrition_eligible_df.index, EMP_TERM_DATE].isna()
        ) | (
            df_processed.loc[exp_attrition_eligible_df.index, EMP_TERM_DATE] > end_date
        )
        num_survivors_initial_cohort = survivor_mask_on_eligible.sum()
    log.info(
        f"Year {sim_year}: Survivors from initial active cohort (n0_exp={n0_exp}, after experienced attrition): {num_survivors_initial_cohort}"
    )

    # --- 7. New Hires Calculation ---
    if maintain_headcount:
        target_eoy_headcount = year_start_headcount_for_calc
    else:  # Growth rate
        target_eoy_headcount = math.ceil(
            year_start_headcount_for_calc * (1 + growth_rate)
        )
    log.info(
        f"Year {sim_year}: Target EOY headcount (based on active start {year_start_headcount_for_calc} and growth {growth_rate:.2%}) = {target_eoy_headcount}"
    )

    # Calculate expected terminations
    expected_terminations = math.ceil(n0_exp * termination_rate)
    log.info(
        f"Year {sim_year}: Expected terminations: {expected_terminations} ({termination_rate:.2%} of {n0_exp} active employees)"
    )

    # Calculate net growth needed
    net_growth_needed = target_eoy_headcount - n0_exp + expected_terminations
    log.info(
        f"Year {sim_year}: Net growth needed: {net_growth_needed} (Target: {target_eoy_headcount}, Start: {n0_exp}, Expected Terminations: {expected_terminations})"
    )

    # Calculate hires needed, accounting for new hire termination rate
    if net_growth_needed > 0:
        hires_to_make = math.ceil(net_growth_needed * (1 + new_hire_termination_rate))
        log.info(
            f"Year {sim_year}: Calculated {hires_to_make} hires needed to achieve net growth of {net_growth_needed} (including {new_hire_termination_rate:.2%} new hire terminations)"
        )
    else:
        hires_to_make = 0
        log.info(
            f"Year {sim_year}: No new hires needed as net growth is not required ({net_growth_needed} <= 0)"
        )

    hires_to_make = max(0, int(hires_to_make))

    new_hires_df = pd.DataFrame()
    if hires_to_make > 0:
        log.info(f"Generating {hires_to_make} new hires for {sim_year}.")
        existing_ids_list = []
        if not df_processed.empty:
            if EMP_ID in df_processed.columns:
                existing_ids_list = df_processed[EMP_ID].unique().tolist()
            else:
                log.warning(
                    f"'{EMP_ID}' column not in df_processed, using its index for existing_ids list generation."
                )
                existing_ids_list = df_processed.index.unique().tolist()

        new_hires_df = generate_new_hires(
            num_hires=hires_to_make,
            hire_year=sim_year,
            scenario_config=year_config,
            existing_ids=existing_ids_list,
            rng=rng_nh_gen,
            id_col_name=EMP_ID,
        )
        if not new_hires_df.empty:
            log.info(f"Generated {len(new_hires_df)} new hire records.")
            if EMP_ID not in new_hires_df.columns:
                log.error(
                    f"CRITICAL: generate_new_hires did not produce column '{EMP_ID}'."
                )
                return pd.DataFrame()
            if not new_hires_df[EMP_ID].is_unique:
                log.error(
                    f"CRITICAL: Generated new hire '{EMP_ID}' values are not unique."
                )

            new_hires_df = new_hires_df.set_index(EMP_ID, drop=False)
            if EMP_TERM_DATE not in new_hires_df.columns:
                new_hires_df[EMP_TERM_DATE] = pd.NaT
            # Onboarding bump logic can be added here
        else:
            log.warning(
                f"generate_new_hires was expected to create {hires_to_make} hires but returned an empty DataFrame."
            )
            hires_to_make = 0

    # --- 8. Apply New Hire Termination Rate ---
    terminated_nh_indices: List[Any] = []
    if hires_to_make > 0 and not new_hires_df.empty and new_hire_termination_rate > 0:
        n0_nh = len(new_hires_df)
        # ... (deterministic or stochastic NH term logic, same as before) ...
        if use_expected_attrition:
            log.info(
                f"Year {sim_year}: Applying DETERMINISTIC new hire termination rate ({new_hire_termination_rate:.2%}) to {n0_nh} new hires."
            )
            n_term_nh = math.ceil(n0_nh * new_hire_termination_rate)
            if n_term_nh > n0_nh:
                n_term_nh = n0_nh
            if n_term_nh > 0:
                terminated_nh_indices = rng_nh_term.choice(
                    new_hires_df.index, size=int(n_term_nh), replace=False
                ).tolist()
                term_dates_for_nh = get_random_dates_in_year(
                    sim_year,
                    count=len(terminated_nh_indices),
                    rng=rng_nh_term,
                    day_of_month=28,
                )
                new_hires_df.loc[terminated_nh_indices, EMP_TERM_DATE] = (
                    term_dates_for_nh
                )
            log.info(
                f"Year {sim_year}: Deterministically terminated {len(terminated_nh_indices)} new hires."
            )
        else:  # Stochastic
            log.info(
                f"Year {sim_year}: Applying STOCHASTIC new hire termination rate ({new_hire_termination_rate:.2%}) to {n0_nh} new hires."
            )
            temp_nh_df = new_hires_df.copy()
            temp_nh_df = apply_turnover(
                df=temp_nh_df,
                hire_col=EMP_HIRE_DATE,
                probs_or_rate=new_hire_termination_rate,
                start_date=start_date,
                end_date=end_date,
                rng=rng_nh_term,
                prev_term_salaries=(
                    temp_nh_df[EMP_GROSS_COMP].dropna()
                    if not temp_nh_df.empty
                    else pd.Series(dtype=float)
                ),
                log=log,
            )
            new_hires_df.update(temp_nh_df)
            terminated_nh_this_step_mask = (
                (new_hires_df[EMP_TERM_DATE].notna())
                & (new_hires_df[EMP_TERM_DATE] >= start_date)
                & (new_hires_df[EMP_TERM_DATE] <= end_date)
            )
            terminated_nh_indices = new_hires_df[
                terminated_nh_this_step_mask
            ].index.tolist()
            log.info(
                f"Year {sim_year}: Stochastically terminated {len(terminated_nh_indices)} new hires."
            )

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
        for df_item in final_df_list:
            all_cols.update(df_item.columns)
        processed_df_list = []
        for df_item in final_df_list:
            for col_to_add in all_cols - set(df_item.columns):
                df_item[col_to_add] = pd.NA
            processed_df_list.append(df_item[list(all_cols)])
        processed_df_list = [df for df in processed_df_list if not df.empty]
        if processed_df_list:
            final_dynamics_df = pd.concat(processed_df_list, ignore_index=True)
        else:
            final_dynamics_df = pd.DataFrame()

    # --- Final Logging ---
    final_eoy_active_mask = (final_dynamics_df[EMP_TERM_DATE].isna()) | (
        final_dynamics_df[EMP_TERM_DATE] > end_date
    )
    final_eoy_active_headcount = final_eoy_active_mask.sum()
    calculated_target_eoy = (
        target_eoy_headcount if "target_eoy_headcount" in locals() else "N/A"
    )
    log.info(
        f"Year {sim_year}: Combined population. Total records = {len(final_dynamics_df)}. "
        f"Active EOY = {final_eoy_active_headcount} (Calculated Target EOY for hiring: {calculated_target_eoy})"
    )
    total_terminated_this_year_mask = (
        (final_dynamics_df[EMP_TERM_DATE].notna())
        & (final_dynamics_df[EMP_TERM_DATE] >= start_date)
        & (final_dynamics_df[EMP_TERM_DATE] <= end_date)
    )
    num_total_terminated_this_year = total_terminated_this_year_mask.sum()
    log.info(
        f"Year {sim_year}: Total terminations recorded in {sim_year} = {num_total_terminated_this_year} (Exp: {len(terminated_exp_indices)}, NH: {len(terminated_nh_indices)})"
    )

    log.info(f"--- Finished Dynamics for Simulation Year {sim_year} ---")
    return final_dynamics_df

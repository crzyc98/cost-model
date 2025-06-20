# cost_model/engines/hire.py
"""
Engine for generating hire events during workforce simulations.
QuickStart: see docs/cost_model/engines/hire.md
"""

import datetime
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from logging_config import get_diagnostic_logger, get_logger

from cost_model.dynamics.sampling.new_hires import sample_new_hire_compensation
from cost_model.dynamics.sampling.salary import DefaultSalarySampler
from cost_model.state.event_log import EVENT_COLS, EVT_COMP, EVT_HIRE, EVT_TERM, create_event
from cost_model.state.schema import (
    EMP_GROSS_COMP,
    EMP_HIRE_DATE,
    EMP_ID,
    EMP_LEVEL,
    EMP_TERM_DATE,
    SIMULATION_YEAR,
)

logger = get_logger(__name__)
diag_logger = get_diagnostic_logger(__name__)

from types import SimpleNamespace

from cost_model.state.schema import EMP_BIRTH_DATE, EMP_TENURE


def run(
    snapshot: pd.DataFrame,
    hires_to_make: int,
    hazard_slice: pd.DataFrame,
    rng: np.random.Generator,
    census_template_path: str,
    global_params: SimpleNamespace,
    terminated_events: pd.DataFrame = None,
) -> List[pd.DataFrame]:
    """
    Determine how many hires to generate so that the *expected* active headcount at
    year-end meets `target_eoy`. Gross-up using the new-hire term rate from hazard_slice.
    Returns two DataFrames (in the list):
      1. A 'hire' events DataFrame
      2. A first 'comp' event DataFrame for those hires (their starting comp)
    """
    if hazard_slice.empty:
        logger.warning(
            "[HIRE.RUN] Hazard slice is empty. Cannot determine simulation year or new hire term rates. Returning no hires."
        )
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]

    simulation_year = int(
        hazard_slice[SIMULATION_YEAR].iloc[0]
    )  # Use constant and ensure it's an int
    logger.info(f"[HIRE.RUN YR={simulation_year}] Hires to make (passed-in): {hires_to_make}")

    # No need to recalculate hires_to_make - the caller has already done this math
    # including the gross-up for new hire termination rate if needed

    # Get role and default compensation parameters from various possible locations
    role_comp_params = {}
    default_params = {}

    # Try to get role_compensation_params from different possible locations
    # 1. Try from global_params.compensation.roles (nested structure in dev_tiny.yaml)
    if hasattr(global_params, "compensation") and hasattr(global_params.compensation, "roles"):
        logger.info(
            f"[HIRE.RUN YR={simulation_year}] Using role_comp_params from global_params.compensation.roles"
        )
        role_comp_params = global_params.compensation.roles
        if not isinstance(role_comp_params, dict):
            role_comp_params = (
                vars(role_comp_params) if hasattr(role_comp_params, "__dict__") else {}
            )
    # 2. Try direct attribute on global_params
    elif hasattr(global_params, "role_compensation_params"):
        logger.info(
            f"[HIRE.RUN YR={simulation_year}] Using role_comp_params from global_params.role_compensation_params"
        )
        role_comp_params = global_params.role_compensation_params
        if not isinstance(role_comp_params, dict):
            role_comp_params = (
                vars(role_comp_params) if hasattr(role_comp_params, "__dict__") else {}
            )

    # Try to get new_hire_compensation_params from different possible locations
    # 1. Try from global_params.compensation.new_hire (nested structure in dev_tiny.yaml)
    if hasattr(global_params, "compensation") and hasattr(global_params.compensation, "new_hire"):
        logger.info(
            f"[HIRE.RUN YR={simulation_year}] Using default_params from global_params.compensation.new_hire"
        )
        default_params = global_params.compensation.new_hire
        if not isinstance(default_params, dict):
            default_params = vars(default_params) if hasattr(default_params, "__dict__") else {}
    # 2. Try direct attribute on global_params
    elif hasattr(global_params, "new_hire_compensation_params"):
        diag_logger.debug(
            f"[HIRE.RUN YR={simulation_year}] Using default_params from global_params.new_hire_compensation_params"
        )
        default_params = global_params.new_hire_compensation_params
        if not isinstance(default_params, dict):
            default_params = vars(default_params) if hasattr(default_params, "__dict__") else {}

    # Log available roles for debugging
    diag_logger.debug(
        f"[HIRE.RUN YR={simulation_year}] Available roles in role_comp_params: {list(role_comp_params.keys()) if isinstance(role_comp_params, dict) else 'None'}"
    )

    if hires_to_make <= 0:
        diag_logger.debug(
            f"[HIRE.RUN YR={simulation_year}] No hires to make as passed-in value is zero or negative."
        )
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]
    # Assign hires to levels according to proportions
    # Choose levels based on current distribution in snapshot
    # Fall back to level 1 if snapshot is empty
    level_counts = snapshot[EMP_LEVEL].value_counts(normalize=True)
    levels = level_counts.index.tolist() or [1]
    probs = level_counts.values.tolist() if not level_counts.empty else [1.0]
    level_choices = rng.choice(levels, size=hires_to_make, p=probs)
    # Generate unique employee_ids (assume string IDs)
    existing_ids = set(snapshot[EMP_ID]) if EMP_ID in snapshot.columns else set(snapshot.index)
    new_ids = []
    i = 1
    # Maximum attempts to prevent potential infinite loops if existing_ids is extremely dense
    # or if f-string somehow generates problematic IDs repeatedly (unlikely for f-string itself).
    max_attempts = hires_to_make * 2 + 20  # Allow a generous number of retries

    if hires_to_make > 0:  # Only attempt to generate IDs if hires are needed
        while len(new_ids) < hires_to_make and i <= max_attempts:
            candidate_eid = f"NH_{simulation_year}_{i:04d}"

            # Validate the candidate_eid string before adding it
            # It should be non-empty, not a representation of NA, and unique.
            is_problematic_string = candidate_eid.lower() in [
                "nan",
                "na",
                "<na>",
                "none",
                "null",
                "",
            ]

            if not is_problematic_string and candidate_eid not in existing_ids:
                new_ids.append(candidate_eid)
            # else: # Optional: log if a candidate was skipped
            # logger.debug(f"[HIRE.RUN] Skipped candidate EID: '{candidate_eid}'. Problematic: {is_problematic_string}, Exists: {candidate_eid in existing_ids}")
            i += 1

        if len(new_ids) < hires_to_make:
            diag_logger.debug(
                f"[HIRE.RUN YR={simulation_year}] Could only generate {len(new_ids)} unique/valid new EIDs out of {hires_to_make} requested "
                f"(attempted {i-1} times). Proceeding with {len(new_ids)} hires."
            )
            # Crucial: Adjust hires_to_make to the actual number of IDs successfully generated.
            # This ensures subsequent list creations (role_choices, hire_dates, etc.) are sized correctly
            # and the zip operation in the event creation loop doesn't misalign.
            hires_to_make = len(new_ids)
    # Generate hire dates uniformly in the year
    start = pd.Timestamp(f"{simulation_year}-01-01")
    end = pd.Timestamp(f"{simulation_year}-12-31")
    days = (end - start).days + 1
    hire_dates = [
        start + pd.Timedelta(days=int(d)) for d in rng.integers(0, days, size=hires_to_make)
    ]

    # Role assignment removed as part of schema refactoring
    # ----- Termination-based sampling with parameterized premium and age jitter -----
    ext_prem = getattr(global_params, "replacement_hire_premium", 0.02)
    age_sd = getattr(global_params, "replacement_hire_age_sd", 2)
    pool = None
    if terminated_events is not None and not terminated_events.empty:
        terms = terminated_events[terminated_events.event_type == EVT_TERM]
        # Ensure EMP_ID is only a column, not both index and column
        if EMP_ID in snapshot.index.names and EMP_ID in snapshot.columns:
            snap = snapshot.reset_index(drop=True)
        elif EMP_ID in snapshot.index.names:
            snap = snapshot.reset_index()
        else:
            snap = snapshot
        terms = terms.merge(
            snap[[EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP]], on=EMP_ID, how="left"
        ).drop_duplicates(subset=EMP_ID)
        pool = terms[[EMP_ID, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_BIRTH_DATE]]
    if pool is not None and len(pool) >= hires_to_make:
        choice_idx = rng.choice(pool.index, size=hires_to_make, replace=True)
        clones = pool.loc[choice_idx].copy().reset_index(drop=True)
        # bump salary by premium
        clones[EMP_GROSS_COMP] *= 1 + ext_prem
        clones["clone_of"] = pool.loc[choice_idx, EMP_ID].values
        # jitter birth_date ± age_sd years
        bd = pd.to_datetime(clones[EMP_BIRTH_DATE])
        jitter_days = rng.normal(0, age_sd * 365.25, size=len(bd)).astype(int)
        clones[EMP_BIRTH_DATE] = bd + pd.to_timedelta(jitter_days, unit="D")
        # keep hire_date same or optionally reset to uniform in year
        clones[EMP_HIRE_DATE] = clones[EMP_HIRE_DATE]
        starting_comps = clones[EMP_GROSS_COMP].values
        birth_dates = pd.to_datetime(clones[EMP_BIRTH_DATE]).dt.strftime("%Y-%m-%d").values
        clone_of = clones["clone_of"].tolist()
    else:
        # Fix: Generate realistic birth dates based on parameters rather than hardcoding
        # --- Use DefaultSalarySampler for config-driven salary sampling ---
        # Try to get age parameters from multiple possible locations
        # 1. From default_params dict
        new_hire_age_mean = default_params.get("new_hire_average_age", None)
        new_hire_age_std = default_params.get("new_hire_age_std_dev", None)
        new_hire_age_min = default_params.get("new_hire_age_min", None)
        new_hire_age_max = default_params.get("new_hire_age_max", None)

        # 2. If not found, try direct attributes on global_params
        if new_hire_age_mean is None:
            new_hire_age_mean = getattr(global_params, "new_hire_average_age", 30)
        if new_hire_age_std is None:
            new_hire_age_std = getattr(global_params, "new_hire_age_std_dev", 5)
        if new_hire_age_min is None:
            new_hire_age_min = getattr(global_params, "new_hire_age_min", 22)
        if new_hire_age_max is None:
            new_hire_age_max = getattr(global_params, "new_hire_age_max", 45)

        # 3. Try nested compensation.new_hire parameters (for auto-tuning compatibility)
        if hasattr(global_params, "compensation") and hasattr(
            global_params.compensation, "new_hire"
        ):
            comp_new_hire = global_params.compensation.new_hire
            if new_hire_age_mean is None or new_hire_age_mean == 30:  # Use nested if available
                new_hire_age_mean = getattr(comp_new_hire, "age_mean", new_hire_age_mean)
            if new_hire_age_std is None or new_hire_age_std == 5:  # Use nested if available
                new_hire_age_std = getattr(comp_new_hire, "age_std", new_hire_age_std)
        # Generate ages using normal distribution with truncation
        ages = rng.normal(new_hire_age_mean, new_hire_age_std, size=hires_to_make)
        ages = np.clip(ages, new_hire_age_min, new_hire_age_max)

        # Prepare level-based compensation parameters
        sampler = DefaultSalarySampler(rng)
        starting_comps = []
        for idx, level in enumerate(level_choices):
            # Look up level-specific compensation parameters from job_levels configuration
            level_params = default_params.copy()  # Start with defaults

            # Find the job level configuration for this specific level
            if hasattr(global_params, "job_levels") and global_params.job_levels:
                job_levels = global_params.job_levels
                # Convert to list if it's not already
                if not isinstance(job_levels, list):
                    job_levels = vars(job_levels) if hasattr(job_levels, "__dict__") else []

                # Find matching level configuration
                level_config = None
                for job_level in job_levels:
                    if hasattr(job_level, "level_id") and job_level.level_id == level:
                        level_config = job_level
                        break
                    elif isinstance(job_level, dict) and job_level.get("level_id") == level:
                        level_config = job_level
                        break

                if level_config:
                    # Update params with level-specific values
                    if hasattr(level_config, "comp_base_salary"):
                        level_params["comp_base_salary"] = level_config.comp_base_salary
                    elif isinstance(level_config, dict) and "comp_base_salary" in level_config:
                        level_params["comp_base_salary"] = level_config["comp_base_salary"]

                    # Also update other compensation parameters if available
                    for param_name in [
                        "comp_age_factor",
                        "comp_stochastic_std_dev",
                        "min_compensation",
                        "max_compensation",
                    ]:
                        if hasattr(level_config, param_name):
                            level_params[param_name.replace("comp_", "comp_")] = getattr(
                                level_config, param_name
                            )
                        elif isinstance(level_config, dict) and param_name in level_config:
                            level_params[param_name.replace("comp_", "comp_")] = level_config[
                                param_name
                            ]

                    # Map min/max_compensation to the expected parameter names
                    if hasattr(level_config, "min_compensation"):
                        level_params["comp_min_salary"] = level_config.min_compensation
                    elif isinstance(level_config, dict) and "min_compensation" in level_config:
                        level_params["comp_min_salary"] = level_config["min_compensation"]

                    if hasattr(level_config, "max_compensation"):
                        level_params["comp_max_salary"] = level_config.max_compensation
                    elif isinstance(level_config, dict) and "max_compensation" in level_config:
                        level_params["comp_max_salary"] = level_config["max_compensation"]

                    logger.info(
                        f"[HIRE.RUN YR={simulation_year}] Using level-specific params for level {level}: base_salary=${level_params.get('comp_base_salary', 'N/A')}"
                    )
                else:
                    logger.warning(
                        f"[HIRE.RUN YR={simulation_year}] No job level configuration found for level {level}, using defaults"
                    )
            else:
                logger.warning(
                    f"[HIRE.RUN YR={simulation_year}] No job_levels configuration available, using default params for level {level}"
                )

            comp = sampler.sample_new_hires(
                size=1, params=level_params, ages=np.array([ages[idx]]), rng=rng
            ).iloc[0]
            starting_comps.append(comp)

        # Convert ages to birth dates based on hire dates
        birth_dates = []
        for hire_date, age in zip(hire_dates, ages):
            # Calculate birth year by subtracting age from hire date year
            birth_year = hire_date.year - int(age)

            # Random month and day
            month = rng.integers(1, 13)  # 1-12
            max_days = 28 if month == 2 else 30 if month in [4, 6, 9, 11] else 31
            day = rng.integers(1, max_days + 1)  # 1-28/30/31

            # Create birth date
            birth_date = pd.Timestamp(f"{birth_year}-{month:02d}-{day:02d}")

            # Adjust if the birth date would make them older/younger than intended
            # by checking if their birthday has occurred yet this year
            actual_age = (
                hire_date.year
                - birth_date.year
                - ((hire_date.month, hire_date.day) < (birth_date.month, birth_date.day))
            )
            if actual_age != int(age):
                # Adjust the year up or down by 1 to get the correct age
                birth_year += int(age) - actual_age
                birth_date = pd.Timestamp(f"{birth_year}-{month:02d}-{day:02d}")

            birth_dates.append(birth_date.strftime("%Y-%m-%d"))

        clone_of = [""] * hires_to_make
    # Role assignment removed as part of schema refactoring

    # Build a DataFrame with one row per new hire for output
    hires_df = pd.DataFrame(
        {
            EMP_ID: new_ids,
            "sampled_comp": starting_comps,
            EMP_HIRE_DATE: hire_dates,
            EMP_BIRTH_DATE: birth_dates,
            "clone_of": clone_of,
        }
    )

    logger.info(
        f"[HIRE.RUN YR={simulation_year}] Sampled new hire salaries: mean=${np.mean(starting_comps):,.0f}, min=${np.min(starting_comps):,.0f}, max=${np.max(starting_comps):,.0f}"
    )
    # Log detailed salary distribution statistics
    logger.info(f"[HIRE.RUN YR={simulation_year}] Salary distribution stats:")
    logger.info(
        f"[HIRE.RUN YR={simulation_year}]   25th percentile: ${np.percentile(starting_comps, 25):,.0f}"
    )
    logger.info(
        f"[HIRE.RUN YR={simulation_year}]   Median: ${np.percentile(starting_comps, 50):,.0f}"
    )
    logger.info(
        f"[HIRE.RUN YR={simulation_year}]   75th percentile: ${np.percentile(starting_comps, 75):,.0f}"
    )

    # 1. Derive year from hazard_slice[SIMULATION_YEAR]
    simulation_year = int(hazard_slice[SIMULATION_YEAR].iloc[0])
    # Again, use EOY to filter terminations for placeholder logic
    as_of = pd.Timestamp(f"{simulation_year}-12-31")

    # 3. (Optional) Gross up by new-hire term rate (not implemented here)
    # 4. (Optional) Read census template for realistic new hire sampling (scaffold only)
    # import pandas as pd
    # census_df = pd.read_parquet(census_template_path)
    # TODO: Sample from census_df by role, etc.

    # 5. (Optional) Pull compensation defaults from plan_rules_config if available (scaffold only)
    # base_comp = plan_rules_config.new_hire_compensation_params.comp_base_salary
    # comp_std = plan_rules_config.new_hire_compensation_params.comp_std
    # ...

    # Generate hire events
    hire_events = []
    for i, (eid, dt, bd_raw, co) in enumerate(zip(new_ids, hire_dates, birth_dates, clone_of)):
        # Debug: Log birth date values and types
        logger.debug(f"Processing birth date for {eid}: type={type(bd_raw)}, value={bd_raw}")

        # Ensure birth date is properly converted to datetime
        try:
            if pd.isna(bd_raw):
                bd = pd.NaT
                bd_str = None
            elif isinstance(bd_raw, (pd.Timestamp, datetime.date, datetime.datetime)):
                bd = pd.to_datetime(bd_raw)
                bd_str = bd.strftime("%Y-%m-%d")
            else:
                # Try to parse string as datetime
                bd = pd.to_datetime(bd_raw, errors="coerce")
                bd_str = bd.strftime("%Y-%m-%d") if not pd.isna(bd) else None

            # Debug: Log final birth date value
            logger.debug(f"Processed birth date for {eid}: final={bd}, str={bd_str}")

            # Convert NumPy types to native Python types for JSON serialization
            if isinstance(co, (np.integer, np.floating)):
                co = int(co) if isinstance(co, np.integer) else float(co)

            # Include compensation in the JSON payload to avoid providing both value_num and value_json
            payload = {
                # role removed as part of schema refactoring
                "birth_date": bd_str,
                "clone_of": str(co) if co is not None else "",
                "compensation": float(starting_comps[i]),  # Include compensation in the payload
            }

            hire_events.append(
                create_event(
                    event_time=dt,
                    employee_id=eid,
                    event_type=EVT_HIRE,
                    value_num=None,  # Setting to None as we're using value_json
                    value_json=payload,  # Let create_event handle the JSON serialization
                    meta=f"Hire event for {eid} in {simulation_year}",
                )
            )
        except Exception as e:
            logger.error(f"Error processing birth date for {eid}: {str(e)}")
            raise
    hire_df = pd.DataFrame(hire_events, columns=EVENT_COLS).sort_values(
        "event_time", ignore_index=True
    )
    # Only return hire events; compensation will be sampled in run_one_year

    # Rename for clarity from general 'hire_df' to 'hires_events_df' if this is the final events DF
    hires_events_df = hire_df

    if not hires_events_df.empty:
        # Check 1: EMP_ID column must exist
        if EMP_ID not in hires_events_df.columns:
            logger.error(
                f"[HIRE.RUN VALIDATION] CRITICAL: Returned hires_events DataFrame is MISSING the '{EMP_ID}' column. Columns present: {hires_events_df.columns.tolist()}"
            )
            # Consider raising an error here to halt execution if this critical column is missing.
            # raise ValueError(f"Hires events DataFrame missing '{EMP_ID}' column.")
        else:
            # Check 2: No actual pd.NA objects in EMP_ID column
            if hires_events_df[EMP_ID].isna().any():
                num_na = hires_events_df[EMP_ID].isna().sum()
                logger.error(
                    f"[HIRE.RUN VALIDATION] CRITICAL: EMP_ID column in returned hires_events CONTAINS {num_na} pd.NA VALUES. This should not happen if create_event stringifies IDs."
                )
                # Consider raising an error.
                # raise ValueError(f"EMP_ID column in hires_events contains {num_na} pd.NA values.")

            # Check 3: All non-NA values in EMP_ID must be non-empty strings
            non_na_ids = hires_events_df[EMP_ID].dropna()
            if not non_na_ids.empty:
                all_valid_strings = True
                for idx, val in non_na_ids.items():
                    if not isinstance(val, str) or not val.strip():
                        all_valid_strings = False
                        logger.error(
                            f"[HIRE.RUN VALIDATION] CRITICAL: EMP_ID column in returned hires_events contains invalid string at index {idx}: Value='{val}', Type={type(val)}."
                        )
                        # Break or collect all problematic ones
                        break
                if not all_valid_strings:
                    # Consider raising an error
                    # raise ValueError("EMP_ID column in hires_events contains non-string or empty string values.")
                    pass  # Logged above

    return [hires_events_df]

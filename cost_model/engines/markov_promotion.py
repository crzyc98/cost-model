import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from logging_config import get_logger, get_diagnostic_logger
from typing import Tuple, Optional, List
from cost_model.state.job_levels.sampling import apply_promotion_markov, load_markov_matrix
from cost_model.state.event_log import EVENT_COLS, EVT_PROMOTION, EVT_RAISE, EVT_TERM, create_event
from cost_model.state.schema import EMP_ID, EMP_LEVEL, EMP_EXITED, EMP_LEVEL_SOURCE, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_BIRTH_DATE, SIMULATION_YEAR
from cost_model.utils.date_utils import age_to_band

logger = get_logger(__name__)


def _extract_promotion_hazard_config(global_params) -> dict:
    """Extract promotion hazard configuration from global_params.

    This replaces the old _load_hazard_defaults() function to support centralized configuration
    for auto-tuning. Falls back to environment variable override for testing compatibility.

    Args:
        global_params: Global parameters object containing promotion_hazard configuration

    Returns:
        Dictionary containing promotion hazard configuration
    """
    import os

    # Check for environment variable override for testing (backward compatibility)
    hazard_file = os.environ.get('HAZARD_CONFIG_FILE')
    if hazard_file:
        config_path = Path(__file__).parent.parent.parent / "config" / hazard_file
        logger.debug(f"[PROMOTION] Using environment override, loading hazard configuration from: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.debug(f"[PROMOTION] Successfully loaded hazard config from {hazard_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"[PROMOTION] Hazard defaults file not found at {config_path}. Falling back to global_params.")
        except Exception as e:
            logger.warning(f"[PROMOTION] Error loading hazard defaults: {e}. Falling back to global_params.")

    # Extract promotion hazard configuration from global_params
    if hasattr(global_params, 'promotion_hazard'):
        promotion_config = global_params.promotion_hazard
        logger.debug("[PROMOTION] Using promotion hazard configuration from global_params")

        # Convert to the expected dictionary format
        config = {
            'promotion': {
                'base_rate': getattr(promotion_config, 'base_rate', 0.10),
                'tenure_multipliers': getattr(promotion_config, 'tenure_multipliers', {}),
                'level_dampener_factor': getattr(promotion_config, 'level_dampener_factor', 0.15),
                'age_multipliers': getattr(promotion_config, 'age_multipliers', {})
            }
        }
        return config

    logger.warning("[PROMOTION] No promotion_hazard configuration found in global_params. Age multipliers will not be applied.")
    return {}


def _apply_promotion_age_multipliers(
    snapshot: pd.DataFrame,
    promotion_matrix: pd.DataFrame,
    year: int,
    hazard_defaults: dict
) -> pd.DataFrame:
    """Apply age-based multipliers to promotion transition probabilities.

    This function modifies the promotion matrix probabilities based on employee age,
    similar to how age multipliers are applied in the termination engine.

    Args:
        snapshot: Employee snapshot DataFrame containing birth dates
        promotion_matrix: Original Markov transition matrix
        year: Current simulation year for age calculation
        hazard_defaults: Hazard configuration containing age multipliers

    Returns:
        Modified promotion matrix with age-adjusted probabilities per employee
    """
    if not hazard_defaults or 'promotion' not in hazard_defaults:
        logger.debug(f"[PROMOTION] Year {year}: No age multipliers found in hazard defaults.")
        return promotion_matrix

    age_multipliers = hazard_defaults.get('promotion', {}).get('age_multipliers', {})
    if not age_multipliers:
        logger.debug(f"[PROMOTION] Year {year}: No age multipliers configured for promotion.")
        return promotion_matrix

    # Check if birth date column exists
    if EMP_BIRTH_DATE not in snapshot.columns:
        logger.warning(f"[PROMOTION] Year {year}: {EMP_BIRTH_DATE} column not found. Age multipliers will not be applied.")
        return promotion_matrix

    # Calculate ages as of year-end (consistent with termination engine)
    as_of_date = pd.Timestamp(f"{year}-12-31")
    birth_dates = pd.to_datetime(snapshot[EMP_BIRTH_DATE], errors='coerce')
    ages = ((as_of_date - birth_dates).dt.days / 365.25).round().astype('Int64')

    # Map ages to age bands and then to multipliers
    emp_ages = {}
    for idx, age in ages.items():
        if pd.isna(age):
            continue
        age_band = age_to_band(int(age))
        emp_ages[snapshot.loc[idx, EMP_ID]] = age_band

    logger.info(f"[PROMOTION] Year {year}: Calculated ages for {len(emp_ages)} employees. "
               f"Age bands: {set(emp_ages.values())}")

    # For Markov promotion, we need to create employee-specific matrices
    # Since the current implementation uses a single matrix for all employees,
    # we'll store the age multipliers to be applied during sampling
    # This is a design decision - we could create per-employee matrices but that's complex

    # Store age multipliers in the snapshot for use during Markov sampling
    snapshot = snapshot.copy()
    snapshot['_promotion_age_multiplier'] = 1.0  # Default multiplier

    for emp_id, age_band in emp_ages.items():
        if age_band in age_multipliers:
            multiplier = age_multipliers[age_band]
            emp_mask = snapshot[EMP_ID] == emp_id
            snapshot.loc[emp_mask, '_promotion_age_multiplier'] = multiplier

    # Log the impact of age adjustments
    non_default_multipliers = (snapshot['_promotion_age_multiplier'] != 1.0).sum()
    if non_default_multipliers > 0:
        logger.info(f"[PROMOTION] Year {year}: Applied age multipliers to {non_default_multipliers} employees. "
                   f"Multiplier range: {snapshot['_promotion_age_multiplier'].min():.2f} - {snapshot['_promotion_age_multiplier'].max():.2f}")

    return snapshot  # Return modified snapshot with age multipliers


def create_promotion_raise_events(
    snapshot: pd.DataFrame,
    promoted: pd.DataFrame,
    promo_time: pd.Timestamp,
    promo_raise_config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create promotion and raise events for promoted employees with level-specific raise percentages.

    Args:
        snapshot: Original snapshot before promotions
        promoted: DataFrame of promoted employees with new levels
        promo_time: Timestamp for the promotion/raise events
        promo_raise_config: Dictionary mapping "{from_level}_to_{to_level}" to raise percentage
                          Example: {"1_to_2": 0.05, "2_to_3": 0.08, "3_to_4": 0.10}

    Returns:
        Tuple of (promotions_df, raises_df) DataFrames
    """
    promotion_events = []
    raise_events = []

    for _, row in promoted.iterrows():
        emp_id = row[EMP_ID]
        # Robustly handle possible duplicate index (row.name)
        from_level_val = snapshot.loc[row.name, EMP_LEVEL]
        if isinstance(from_level_val, pd.Series):
            logger = get_logger(__name__)
            logger.warning(f"[PROMOTION] Duplicate index for employee {emp_id} (row.name={row.name}); skipping promotion/raise event.")
            continue
        from_level = int(from_level_val)
        to_level = int(row[EMP_LEVEL])
        try:
            comp_val = snapshot.loc[row.name, EMP_GROSS_COMP]
            if pd.isna(comp_val) or comp_val is pd.NA:
                # This should be rare now that we filter at the start, but keep as a safeguard
                hire_date = snapshot.loc[row.name, EMP_HIRE_DATE]
                hire_date_str = hire_date.strftime('%Y-%m-%d') if not pd.isna(hire_date) else 'unknown hire date'
                logger = get_logger(__name__)
                logger.warning(
                    f"[PROMOTION] Employee {emp_id} (hired {hire_date_str}) missing {EMP_GROSS_COMP}; "
                    "promotion/raise event skipped. This should have been caught earlier in the process."
                )
                continue
            current_comp = float(comp_val)
        except KeyError as e:
            logger = get_logger(__name__)
            logger.error(
                f"[PROMOTION] Error accessing compensation data for employee {emp_id}: {str(e)}. "
                "This suggests a data integrity issue. Skipping promotion/raise event."
            )
            continue

        # Get the raise percentage for this promotion level
        level_key = f"{from_level}_to_{to_level}"
        raise_pct = promo_raise_config.get(level_key, 0.10)  # Default to 10% if not specified
        raise_amount = current_comp * raise_pct

        # Create promotion event
        promo_event = create_event(
            event_time=promo_time,
            employee_id=emp_id,
            event_type=EVT_PROMOTION,
            value_json=json.dumps({
                "from_level": from_level,
                "to_level": to_level,
                "previous_comp": current_comp,
                "raise_pct": raise_pct
            }),
            meta=f"Promotion from level {from_level} to {to_level} with {raise_pct:.0%} raise"
        )
        promotion_events.append(promo_event)

        # Create raise event with all details in value_json
        # Promotion raises at 00:00:30 (before merit at 00:01, before COLA at 00:02)
        raise_event = create_event(
            event_time=promo_time + pd.Timedelta(seconds=30),  # Promotion raises at 00:00:30
            employee_id=emp_id,
            event_type=EVT_RAISE,
            value_json=json.dumps({
                "amount": raise_amount,
                "previous_comp": current_comp,
                "new_comp": current_comp * (1 + raise_pct),
                "raise_pct": raise_pct,
                "reason": f"promotion_{level_key}",
                "from_level": from_level,
                "to_level": to_level
            }),
            meta=f"{raise_pct:.1%} raise for promotion from level {from_level} to {to_level}"
        )
        raise_events.append(raise_event)


    # Convert to DataFrames with proper schema
    promotions_df = pd.DataFrame(promotion_events, columns=EVENT_COLS) if promotion_events else pd.DataFrame(columns=EVENT_COLS)
    raises_df = pd.DataFrame(raise_events, columns=EVENT_COLS) if raise_events else pd.DataFrame(columns=EVENT_COLS)

    return promotions_df, raises_df

from logging_config import get_logger


import sys
from cost_model.state.job_levels.sampling import load_markov_matrix

def apply_markov_promotions(
    snapshot: pd.DataFrame,
    promo_time: pd.Timestamp,
    rng: Optional[np.random.RandomState] = None,
    promotion_raise_config: Optional[dict] = None,
    simulation_year: Optional[int] = None,
    promotion_matrix: Optional[pd.DataFrame] = None,
    global_params=None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply Markov-chain based promotions to the workforce with associated raises.

    Args:
        snapshot: Current workforce snapshot DataFrame
        promo_time: Timestamp for when promotions occur
        rng: Optional random number generator for reproducibility
        promotion_raise_config: Dictionary mapping "{from_level}_to_{to_level}" to raise percentage
                             Example: {"1_to_2": 0.05, "2_to_3": 0.08, "3_to_4": 0.10}
                             If None, uses default 10% for all promotions
        simulation_year: Optional simulation year for promotions. If not provided, will be inferred from promo_time or snapshot

    Returns:
        Tuple of (promotions_df, raises_df, exits_df) where:
        - promotions_df: DataFrame of promotion events
        - raises_df: DataFrame of raise events associated with promotions
        - exits_df: DataFrame of exit events
    """
    # Set up loggers
    logger = get_logger(__name__)
    diag_logger = get_diagnostic_logger(__name__)

    # Load promotion matrix dynamically if not provided
    if promotion_matrix is None:
        allow_default = getattr(global_params, "dev_mode", False)
        promotion_matrix_path = getattr(global_params, "promotion_matrix_path", None)

        try:
            promotion_matrix = load_markov_matrix(
                promotion_matrix_path,
                allow_default
            )
        except (FileNotFoundError, ValueError) as e:
            logger.error(
                "Promotion matrix load/validation failed: %s", e
            )
            sys.exit(1)

    # Set default raise config if not provided
    if promotion_raise_config is None:
        promotion_raise_config = {
            "1_to_2": 0.05,  # 5% raise for 1→2
            "2_to_3": 0.08,  # 8% raise for 2→3
            "3_to_4": 0.10,  # 10% raise for 3→4
            "default": 0.10  # Default 10% for any other promotions
        }

    # Create a copy to avoid modifying the input
    snapshot = snapshot.copy()

    # Add diagnostics for EMP_LEVEL at start of apply_markov_promotions
    year_val = simulation_year or promo_time.year
    diag_logger.debug(f"[MARKOV_PROMOTION DIAGNOSTIC YR={year_val}] Snapshot analysis at start of apply_markov_promotions:")
    if EMP_LEVEL in snapshot.columns:
        na_count = snapshot[EMP_LEVEL].isna().sum()
        diag_logger.debug(f"  Input snapshot['{EMP_LEVEL}'] NA count: {na_count}")
        diag_logger.debug(f"  Input snapshot['{EMP_LEVEL}'] dtype: {snapshot[EMP_LEVEL].dtype}")

        # If there are any NaN values, log the affected employee IDs for debugging
        if na_count > 0:
            na_employees = snapshot[snapshot[EMP_LEVEL].isna()][[EMP_ID, EMP_HIRE_DATE]].copy()
            if not na_employees.empty:
                na_employees[EMP_HIRE_DATE] = na_employees[EMP_HIRE_DATE].dt.strftime('%Y-%m-%d')
                diag_logger.debug(f"  Employees with NaN {EMP_LEVEL}: {na_employees.to_dict('records')}")

        # Log level distribution if there are no NaNs or after removing NaNs
        non_na_snapshot = snapshot.dropna(subset=[EMP_LEVEL])
        if not non_na_snapshot.empty:
            level_counts = non_na_snapshot[EMP_LEVEL].value_counts().to_dict()
            diag_logger.debug(f"  Level distribution in input snapshot: {level_counts}")
    else:
        diag_logger.warning(f"{EMP_LEVEL} column not found in input snapshot")

    # Check for missing compensation and log details before filtering
    missing_comp_mask = snapshot[EMP_GROSS_COMP].isna()
    if missing_comp_mask.any():
        missing_employees = snapshot[missing_comp_mask][[EMP_ID, EMP_HIRE_DATE]].copy()
        missing_employees[EMP_HIRE_DATE] = missing_employees[EMP_HIRE_DATE].dt.strftime('%Y-%m-%d')

        # Log summary
        diag_logger.debug(
            f"Found {missing_comp_mask.sum()} employees missing {EMP_GROSS_COMP} in promotion processing. "
            f"These employees will be excluded from promotion consideration."
        )

        # Log first few examples for debugging
        for _, emp in missing_employees.head(5).iterrows():
            logger.debug(
                f"Employee {emp[EMP_ID]} (hired {emp[EMP_HIRE_DATE]}) is missing compensation data; "
                "skipping promotion consideration."
            )

        if len(missing_employees) > 5:
            logger.debug(f"... and {len(missing_employees) - 5} more employees with missing compensation.")

        # Filter out employees with missing compensation
        snapshot = snapshot[~missing_comp_mask].copy()

        if snapshot.empty:
            logger.debug("No employees with valid compensation data remaining after filtering.")
            return (
                pd.DataFrame(columns=EVENT_COLS),
                pd.DataFrame(columns=EVENT_COLS),
                pd.DataFrame(columns=snapshot.columns)
            )

    # Get the simulation year from the parameter, promo_time, or the snapshot
    if simulation_year is None:
        simulation_year = promo_time.year if hasattr(promo_time, 'year') else None
        if simulation_year is None and 'simulation_year' in snapshot.columns:
            simulation_year = snapshot['simulation_year'].iloc[0] if not snapshot.empty else None

    # Extract promotion hazard configuration from global_params
    hazard_defaults = _extract_promotion_hazard_config(global_params)

    # Apply age multipliers to promotion probabilities
    # Note: This modifies the snapshot to include age multipliers for use in Markov sampling
    snapshot = _apply_promotion_age_multipliers(snapshot, promotion_matrix, simulation_year, hazard_defaults)

    # Apply Markov promotions with termination date handling
    out = apply_promotion_markov(
        snapshot,
        rng=rng,
        simulation_year=simulation_year,
        matrix=promotion_matrix  # Always pass the loaded promotion matrix
    )

    # DIAGNOSTIC: Immediately check for NaNs after apply_promotion_markov
    if pd.isna(out[EMP_LEVEL]).any():
        nan_count = pd.isna(out[EMP_LEVEL]).sum()
        logger.debug(f"[MARKOV_PROMOTION DIAGNOSTIC YR={year_val}] IMMEDIATELY after apply_promotion_markov: Found {nan_count} NaN values in {EMP_LEVEL}")

        # Get some details about the employees with NaN levels
        nan_emps = out[pd.isna(out[EMP_LEVEL])]
        if not nan_emps.empty and EMP_ID in nan_emps.columns:
            nan_emp_ids = nan_emps[EMP_ID].tolist()
            logger.debug(f"[MARKOV_PROMOTION DIAGNOSTIC YR={year_val}] Employees with NaN {EMP_LEVEL}: {nan_emp_ids[:5]}{'...' if len(nan_emp_ids) > 5 else ''}")

            # Check if these employees have exited
            nan_exited = nan_emps[EMP_EXITED].sum() if EMP_EXITED in nan_emps.columns else 'N/A'
            logger.debug(f"[MARKOV_PROMOTION DIAGNOSTIC YR={year_val}] Out of {len(nan_emps)} employees with NaN {EMP_LEVEL}, {nan_exited} have exited=True")
    else:
        logger.debug(f"[MARKOV_PROMOTION DIAGNOSTIC YR={year_val}] IMMEDIATELY after apply_promotion_markov: No NaN values in {EMP_LEVEL}, all good!")

    # Create promotion events for level changes
    promoted_mask = (out[EMP_LEVEL] != snapshot[EMP_LEVEL]) & ~out[EMP_EXITED]
    promoted = out[promoted_mask].copy()

    if promoted.empty:
        return (
            pd.DataFrame(columns=EVENT_COLS),
            pd.DataFrame(columns=EVENT_COLS),
            out[out[EMP_EXITED]].copy()
        )

    # Create promotion and raise events for those who were promoted
    promotions_df, raises_df = create_promotion_raise_events(
        snapshot,
        promoted,
        promo_time,
        promo_raise_config=promotion_raise_config
    )

    # Update job_level_source for promoted employees
    if not promoted.empty:
        # Ensure the category exists before setting the value
        if EMP_LEVEL_SOURCE in out.columns and pd.api.types.is_categorical_dtype(out[EMP_LEVEL_SOURCE]):
            # Add 'markov-promo' to the categories if it's not already there
            if 'markov-promo' not in out[EMP_LEVEL_SOURCE].cat.categories:
                out[EMP_LEVEL_SOURCE] = out[EMP_LEVEL_SOURCE].cat.add_categories(['markov-promo'])

        # Now it's safe to set the value
        out.loc[promoted.index, EMP_LEVEL_SOURCE] = 'markov-promo'

        # DIAGNOSTIC: Check if the level_source update introduced any NaNs
        if pd.isna(out[EMP_LEVEL]).any():
            nan_count = pd.isna(out[EMP_LEVEL]).sum()
            logger.debug(f"[MARKOV_PROMOTION DIAGNOSTIC YR={year_val}] After EMP_LEVEL_SOURCE update: Found {nan_count} NaN values in {EMP_LEVEL}")

    # Update the levels in the output snapshot
    # Fill NaN values before converting to int to avoid IntCastingNaNError
    if pd.isna(out[EMP_LEVEL]).any():
        # Log how many NaN values we found
        nan_count = pd.isna(out[EMP_LEVEL]).sum()
        logger.debug(f"[MARKOV_PROMOTION DIAGNOSTIC YR={year_val}] Final check: Found {nan_count} NaN values in {EMP_LEVEL}, filling with default level 1")

        # Get details about the employees with NaNs at this stage
        nan_indices = out.index[pd.isna(out[EMP_LEVEL])].tolist()
        if len(nan_indices) > 0:
            nan_emps = out.loc[nan_indices]
            if EMP_ID in nan_emps.columns:
                nan_emp_ids = nan_emps[EMP_ID].tolist()
                logger.debug(f"[MARKOV_PROMOTION DIAGNOSTIC YR={year_val}] Final check - Employees with NaN {EMP_LEVEL}: {nan_emp_ids[:5]}{'...' if len(nan_emp_ids) > 5 else ''}")

                # Check if these employees have exited flag set
                if EMP_EXITED in nan_emps.columns:
                    exited_count = nan_emps[EMP_EXITED].sum()
                    logger.debug(f"[MARKOV_PROMOTION DIAGNOSTIC YR={year_val}] Out of {nan_count} employees with NaN {EMP_LEVEL}, {exited_count} have exited=True")

        # Fill NaN values with level 1 (or another appropriate default)
        out[EMP_LEVEL] = out[EMP_LEVEL].fillna(1)

    # Now convert to int safely
    out[EMP_LEVEL] = out[EMP_LEVEL].astype(int)

    # Get employees who exited during Markov promotions
    exited_employees = out[out[EMP_EXITED]].copy()

    # Create proper exit events for employees who exited
    exit_events = []

    for _, row in exited_employees.iterrows():
        emp_id = row[EMP_ID]

        # Get termination date if it exists, or use promo_time as fallback
        term_date = row.get('employee_termination_date', promo_time)
        if pd.isna(term_date):
            term_date = promo_time

        # Create exit event with proper schema using standard EVT_TERM type
        exit_event = create_event(
            event_time=term_date,
            employee_id=emp_id,
            event_type=EVT_TERM,  # Using standard termination event type
            value_json=json.dumps({
                "previous_level": int(row.get(EMP_LEVEL, 0)),
                "exit_source": "markov_promotion"
            }),
            meta=f"Employee exited via Markov promotion chain"
        )
        exit_events.append(exit_event)

    # Convert to DataFrame with proper event schema
    exited_df = pd.DataFrame(exit_events, columns=EVENT_COLS) if exit_events else pd.DataFrame(columns=EVENT_COLS)

    # Add simulation_year column if it's not already there
    if not exited_df.empty and SIMULATION_YEAR not in exited_df.columns:
        exited_df[SIMULATION_YEAR] = simulation_year

    return promotions_df, raises_df, exited_df

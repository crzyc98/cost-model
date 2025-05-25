import json
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from cost_model.state.job_levels.sampling import apply_promotion_markov, load_markov_matrix
from cost_model.state.event_log import EVENT_COLS, EVT_PROMOTION, EVT_RAISE, create_event
from cost_model.state.schema import EMP_ID, EMP_LEVEL, EMP_ROLE, EMP_EXITED, EMP_LEVEL_SOURCE, EMP_GROSS_COMP, EMP_HIRE_DATE


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
            logging.warning(f"[PROMOTION] Duplicate index for employee {emp_id} (row.name={row.name}); skipping promotion/raise event.")
            continue
        from_level = int(from_level_val)
        to_level = int(row[EMP_LEVEL])
        try:
            comp_val = snapshot.loc[row.name, EMP_GROSS_COMP]
            if pd.isna(comp_val) or comp_val is pd.NA:
                # This should be rare now that we filter at the start, but keep as a safeguard
                hire_date = snapshot.loc[row.name, EMP_HIRE_DATE]
                hire_date_str = hire_date.strftime('%Y-%m-%d') if not pd.isna(hire_date) else 'unknown hire date'
                logging.warning(
                    f"[PROMOTION] Employee {emp_id} (hired {hire_date_str}) missing {EMP_GROSS_COMP}; "
                    "promotion/raise event skipped. This should have been caught earlier in the process."
                )
                continue
            current_comp = float(comp_val)
        except KeyError as e:
            logging.error(
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
        raise_event = create_event(
            event_time=promo_time + pd.Timedelta(days=1),  # Raise happens day after promotion
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

import logging
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
    # Set up logger at function start
    logger = logging.getLogger(__name__)
    
    # Load promotion matrix dynamically if not provided
    if promotion_matrix is None:
        allow_default = getattr(global_params, "dev_mode", False)
        try:
            promotion_matrix = load_markov_matrix(
                getattr(global_params, "promotion_matrix_path", None),
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
    
    # Check for missing compensation and log details before filtering
    missing_comp_mask = snapshot[EMP_GROSS_COMP].isna()
    if missing_comp_mask.any():
        missing_employees = snapshot[missing_comp_mask][[EMP_ID, EMP_HIRE_DATE]].copy()
        missing_employees[EMP_HIRE_DATE] = missing_employees[EMP_HIRE_DATE].dt.strftime('%Y-%m-%d')
        
        # Log summary
        logger.warning(
            f"Found {missing_comp_mask.sum()} employees missing {EMP_GROSS_COMP} in promotion processing. "
            f"These employees will be excluded from promotion consideration."
        )
        
        # Log first few examples for debugging
        for _, emp in missing_employees.head(5).iterrows():
            logger.warning(
                f"Employee {emp[EMP_ID]} (hired {emp[EMP_HIRE_DATE]}) is missing compensation data; "
                "skipping promotion consideration."
            )
        
        if len(missing_employees) > 5:
            logger.warning(f"... and {len(missing_employees) - 5} more employees with missing compensation.")
        
        # Filter out employees with missing compensation
        snapshot = snapshot[~missing_comp_mask].copy()
        
        if snapshot.empty:
            logger.warning("No employees with valid compensation data remaining after filtering.")
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
    
    # Apply Markov promotions with termination date handling
    out = apply_promotion_markov(
        snapshot, 
        rng=rng, 
        simulation_year=simulation_year,
        matrix=promotion_matrix  # Always pass the loaded promotion matrix
    )
    
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
    
    # Update the levels in the output snapshot
    # Fill NaN values before converting to int to avoid IntCastingNaNError
    if pd.isna(out[EMP_LEVEL]).any():
        # Log how many NaN values we found
        nan_count = pd.isna(out[EMP_LEVEL]).sum()
        logger = logging.getLogger(__name__)
        logger.warning(f"Found {nan_count} NaN values in EMP_LEVEL, filling with default level 1")
        
        # Fill NaN values with level 1 (or another appropriate default)
        out[EMP_LEVEL] = out[EMP_LEVEL].fillna(1)
    
    # Now convert to int safely
    out[EMP_LEVEL] = out[EMP_LEVEL].astype(int)
    
    return promotions_df, raises_df, out[out[EMP_EXITED]].copy()

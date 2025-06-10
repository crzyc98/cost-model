"""
Hiring logic module for run_one_year package.

Handles new hire generation, compensation assignment, and termination.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cost_model.engines import hire
from cost_model.state.event_log import EVENT_COLS, create_event
from cost_model.state.job_levels.sampling import sample_mixed_new_hires
from cost_model.state.schema import (
    EMP_ACTIVE,
    EMP_BIRTH_DATE,
    EMP_GROSS_COMP,
    EMP_HIRE_DATE,
    EMP_ID,
    EMP_LEVEL,
    EMP_LEVEL_SOURCE,
    EMP_TERM_DATE,
    EVT_COMP,
    EVT_HIRE,
)
from cost_model.state.snapshot_update import _apply_new_hires

# Set up logging
logger = logging.getLogger(__name__)


def compute_hire_counts(
    start_count: int, global_params: Any, hazard_slice: pd.DataFrame, year: int
) -> Tuple[int, int]:
    """
    Computes the number of hires needed based on growth targets.

    Args:
        start_count: Starting headcount
        global_params: Global configuration parameters
        hazard_slice: Hazard table slice for the current year
        year: Current simulation year

    Returns:
        Tuple containing (net_hires, gross_hires)
    """
    logger = logging.getLogger(__name__)

    # Determine new hire rate
    new_hire_rate = getattr(global_params, "new_hire_rate", 0.0)
    logger.info(
        f"[RUN_ONE_YEAR YR={year}] Using new_hire_rate={new_hire_rate} from global_params.new_hire_rate"
    )

    # Determine new hire termination rate
    new_hire_term_rate = getattr(global_params, "new_hire_termination_rate", 0.0)
    logger.info(
        f"[RUN_ONE_YEAR YR={year}] Using new_hire_termination_rate={new_hire_term_rate} from global_params.new_hire_termination_rate"
    )

    # Calculate net hires based on start count and growth
    net_hires = int(start_count * new_hire_rate)

    # Calculate gross hires (accounting for expected attrition)
    gross_hires = net_hires
    if new_hire_term_rate > 0:
        gross_hires = int(net_hires / (1 - new_hire_term_rate))

    logger.info(
        f"[RUN_ONE_YEAR YR={year}] start={start_count}, net_hires={net_hires} "
        f"({new_hire_rate*100:.1f}%), gross_hires={gross_hires} "
        f"(to allow for {new_hire_term_rate*100:.1f}% NH term)"
    )

    return net_hires, gross_hires


def generate_hire_events(
    temp_snapshot: pd.DataFrame,
    gross_hires: int,
    hazard_slice: pd.DataFrame,
    year_rng: np.random.Generator,
    census_template_path: str,
    global_params: Any,
    term_events: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """
    Generates new hire events.

    Args:
        temp_snapshot: Current snapshot DataFrame
        gross_hires: Number of gross hires to generate
        hazard_slice: Hazard table slice for the current year
        year_rng: Random number generator
        census_template_path: Path to census template
        global_params: Global configuration parameters
        term_events: Termination events DataFrame
        year: Current simulation year

    Returns:
        DataFrame containing hire events
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[HIRE.RUN YR={year}] Hires to make (passed-in): {gross_hires}")

    if gross_hires <= 0:
        return pd.DataFrame()

    # Generate hire events using hire.run
    hire_events_dfs = hire.run(
        snapshot=temp_snapshot,
        hires_to_make=gross_hires,
        hazard_slice=hazard_slice,
        rng=year_rng,
        census_template_path=census_template_path,
        global_params=global_params,
        terminated_events=term_events,
    )

    # hire.run() returns a list of DataFrames - the first one contains the hire events
    if not hire_events_dfs or len(hire_events_dfs) == 0:
        logger.warning(f"[YR={year}] No hire events generated")
        return pd.DataFrame()

    hire_events_df = hire_events_dfs[0] if len(hire_events_dfs) > 0 else pd.DataFrame()

    if hire_events_df is None or hire_events_df.empty:
        logger.warning(f"[YR={year}] No hire events in the result")
        return pd.DataFrame()

    logger.info(f"[RUN_ONE_YEAR YR={year}] Generated {len(hire_events_df)} new-hire events")
    return hire_events_df


def sample_and_build_hire_comp_events(
    prev_snapshot: pd.DataFrame,
    new_hire_events: pd.DataFrame,
    gross_hires: int,
    year_rng: np.random.Generator,
    year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Samples compensation for new hires and builds compensation events.

    Args:
        prev_snapshot: Previous snapshot DataFrame
        new_hire_events: New hire events DataFrame
        gross_hires: Number of gross hires
        year_rng: Random number generator
        year: Current simulation year

    Returns:
        Tuple containing (comp_events_df, new_hires_df)
    """
    logger = logging.getLogger(__name__)

    # Skip if no gross hires
    if gross_hires <= 0 or new_hire_events.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 1. Derive distribution of levels from previous snapshot
    if EMP_LEVEL in prev_snapshot.columns:
        # Convert level IDs to integers and filter valid ones
        valid_levels = pd.to_numeric(prev_snapshot[EMP_LEVEL], errors="coerce").dropna()
        curr_dist = valid_levels.value_counts(normalize=True).to_dict()
    else:
        curr_dist = {}

    # If no current distribution, use default distribution from job levels config
    if not curr_dist:
        logger.warning(
            f"[YR={year}] No level distribution found in snapshot, using default job levels"
        )
        curr_dist = {1: 0.5, 2: 0.3, 3: 0.15, 4: 0.05}  # Default distribution for job levels 1-4

    # Ensure we have integer level IDs
    level_counts = {int(lvl): int(dist * gross_hires) for lvl, dist in curr_dist.items()}

    # Ensure we have the exact number of hires
    assigned = sum(level_counts.values())
    if assigned < gross_hires and level_counts:
        # Distribute remaining hires according to the same distribution
        remaining = gross_hires - assigned
        levels = list(level_counts.keys())
        probs = [curr_dist[lvl] for lvl in levels]
        probs = [p / sum(probs) for p in probs]  # Normalize

        # Sample remaining hires
        additional = np.random.choice(levels, size=remaining, p=probs)
        for lvl in additional:
            level_counts[lvl] += 1

    # 2. Sample new hires with levels & compensation
    new_hires = sample_mixed_new_hires(level_counts, age_range=(25, 55), random_state=year_rng)

    # Rename columns to match schema
    new_hires = new_hires.rename(columns={"level_id": EMP_LEVEL, "compensation": EMP_GROSS_COMP})
    new_hires["job_level_source"] = "hire"

    # 3. Align hire IDs from hire events
    if not new_hire_events.empty:
        new_hires[EMP_ID] = new_hire_events[EMP_ID].reset_index(drop=True)

    # 4. Build compensation events for new hires
    comp_events = []
    logger.info(f"[YR={year}] Processing {len(new_hires)} new hires for compensation events")

    for idx, r in new_hires.iterrows():
        # Get EMP_ID and ensure it's valid
        emp_id = r.get(EMP_ID)
        if pd.isna(emp_id) or emp_id == "":
            logger.warning(
                f"[YR={year}] Row {idx}: Skipping EVT_COMP creation - missing employee ID"
            )
            continue

        # Extract compensation and validate it
        comp_value = None

        # Try different possible field names for compensation
        if EMP_GROSS_COMP in r and not pd.isna(r[EMP_GROSS_COMP]):
            comp_value = r[EMP_GROSS_COMP]
            level = r.get(EMP_LEVEL, "N/A") if not pd.isna(r.get(EMP_LEVEL)) else "N/A"
            logger.info(
                f"[YR={year}] Using compensation for {emp_id} (level={level}): {comp_value}"
            )

            # Get level-based comp if available for comparison
            if (
                EMP_LEVEL in new_hires.columns
                and not new_hires[new_hires[EMP_LEVEL] == level].empty
            ):
                level_comp = new_hires[
                    (new_hires[EMP_LEVEL] == level) & (new_hires[EMP_GROSS_COMP].notna())
                ][EMP_GROSS_COMP].mean()
                if not np.isnan(level_comp):
                    logger.info(f"[YR={year}] Avg compensation for level {level}: {level_comp:.2f}")

        # If still no value, use a default based on job level if available
        if comp_value is None or pd.isna(comp_value):
            level = r.get(EMP_LEVEL, 1)
            if pd.isna(level):
                level = 1

            # Default compensation based on job level (can be adjusted)
            level_based_comp = {1: 50000.0, 2: 75000.0, 3: 100000.0, 4: 150000.0}.get(
                int(level), 75000.0
            )

            comp_value = level_based_comp
            logger.warning(
                f"[YR={year}] Row {idx}: Using level-based default compensation "
                f"for employee {emp_id} (level={level}): {comp_value}"
            )

        # Final check for valid numeric compensation
        if not isinstance(comp_value, (int, float)) or pd.isna(comp_value):
            logger.warning(
                f"[YR={year}] Row {idx}: Invalid compensation value "
                f"for employee {emp_id}: {comp_value}. Using default."
            )
            comp_value = 75000.0

        # Create the compensation event
        event = create_event(
            event_time=pd.Timestamp(f"{year}-01-01"),
            employee_id=emp_id,
            event_type=EVT_COMP,
            value_num=float(comp_value),
            meta=f"new_hire;before=0.00;after={float(comp_value):.2f}",
        )
        comp_events.append(event)

    # Check if we created valid events
    if not comp_events:
        logger.warning(f"[YR={year}] No valid compensation events created for new hires")
        return pd.DataFrame(), new_hires

    comp_events_df = pd.DataFrame(comp_events)
    logger.info(f"[YR={year}] Created {len(comp_events_df)} compensation events for new hires")

    return comp_events_df, new_hires


def process_new_hires(
    temp_snapshot: pd.DataFrame,
    gross_hires: int,
    hazard_slice: pd.DataFrame,
    year_rng: np.random.Generator,
    census_template_path: str,
    global_params: Any,
    term_events: pd.DataFrame,
    year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to handle all new hire processing:
    - Generate hire events
    - Sample compensation
    - Create compensation events
    - Apply new hires to snapshot

    Args:
        temp_snapshot: Current snapshot DataFrame
        gross_hires: Number of gross hires
        hazard_slice: Hazard table slice for the current year
        year_rng: Random number generator
        census_template_path: Path to census template
        global_params: Global configuration parameters
        term_events: Termination events DataFrame
        year: Current simulation year

    Returns:
        Tuple containing (combined_events_df, updated_snapshot)
    """
    logger = logging.getLogger(__name__)

    # Skip if no hires
    if gross_hires <= 0:
        return pd.DataFrame(), temp_snapshot

    # Generate hire events
    hire_events_df = generate_hire_events(
        temp_snapshot=temp_snapshot,
        gross_hires=gross_hires,
        hazard_slice=hazard_slice,
        year_rng=year_rng,
        census_template_path=census_template_path,
        global_params=global_params,
        term_events=term_events,
        year=year,
    )

    if hire_events_df.empty:
        return pd.DataFrame(), temp_snapshot

    # Sample and build compensation events
    comp_events_df, new_hires_df = sample_and_build_hire_comp_events(
        prev_snapshot=temp_snapshot,
        new_hire_events=hire_events_df,
        gross_hires=gross_hires,
        year_rng=year_rng,
        year=year,
    )

    # Apply new hires to snapshot
    snapshot_with_hires = temp_snapshot.copy()
    if not new_hires_df.empty:
        # Ensure the new_hires_df has the required columns
        if "event_type" not in new_hires_df.columns:
            new_hires_df = new_hires_df.copy()
            new_hires_df["event_type"] = EVT_HIRE

        # Ensure the new_hires_df has the 'value_num' column for compensation
        if "value_num" not in new_hires_df.columns and EMP_GROSS_COMP in new_hires_df.columns:
            new_hires_df["value_num"] = new_hires_df[EMP_GROSS_COMP]

        # Ensure the new_hires_df has the 'event_time' column
        if "event_time" not in new_hires_df.columns:
            new_hires_df["event_time"] = pd.Timestamp(f"{year}-01-01")

        snapshot_with_hires = _apply_new_hires(
            current=snapshot_with_hires, new_events=new_hires_df, year=year
        )

        logger.info(
            f"[YR={year}] With hires snapshot: {len(snapshot_with_hires[EMP_ID].unique())} unique EMP_IDs, {snapshot_with_hires.shape[0]} rows"
        )
        logger.debug(
            f"[YR={year}] With hires snapshot details: {snapshot_with_hires.head().to_dict()}"
        )

        active_count = (
            snapshot_with_hires["active"].sum() if "active" in snapshot_with_hires.columns else 0
        )
        logger.info(
            f"[YR={year}] After Hires     = {active_count}  (added {len(new_hires_df)} hires)"
        )

    # Combine all events
    all_events = []
    if not hire_events_df.empty:
        all_events.append(hire_events_df)
    if not comp_events_df.empty:
        all_events.append(comp_events_df)

    if all_events:
        combined_events = pd.concat(all_events, ignore_index=True)
        return combined_events, snapshot_with_hires

    return pd.DataFrame(), snapshot_with_hires

# cost_model/projections/snapshot.py
"""
Refactored snapshot module for creating and updating workforce snapshots during projections.
QuickStart: see docs/cost_model/projections/snapshot.md

This module has been refactored to use modular components for better maintainability.
The main functionality is now provided by the SnapshotBuilder class and supporting components.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union, Dict, Tuple, List

from cost_model.state.snapshot.constants import SNAPSHOT_COLS as SNAPSHOT_COL_NAMES, SNAPSHOT_DTYPES
from cost_model.state.schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE,
    EMP_GROSS_COMP, EMP_DEFERRAL_RATE, EMP_TENURE_BAND, EMP_TENURE,
    EMP_TERM_DATE, EMP_ACTIVE, EMP_LEVEL, EMP_LEVEL_SOURCE, EMP_EXITED, SIMULATION_YEAR
)

# Import the new modular components
from .snapshot.snapshot_builder import SnapshotBuilder

logger = logging.getLogger(__name__)

# Create a module-level instance for backward compatibility
_snapshot_builder = SnapshotBuilder()


def create_initial_snapshot(start_year: int, census_path: Union[str, Path]) -> pd.DataFrame:
    """
    Create the initial employee snapshot from census data.
    
    This function now delegates to the modular SnapshotBuilder for improved maintainability.

    Args:
        start_year: The starting year for the simulation
        census_path: Path to the census data file (Parquet format)

    Returns:
        DataFrame containing the initial employee snapshot

    Raises:
        FileNotFoundError: If the census file doesn't exist
        ValueError: If the census data is invalid or missing required columns
    """
    return _snapshot_builder.create_initial_snapshot(start_year, census_path)


def update_snapshot_with_events(
    prev_snapshot: pd.DataFrame,
    events_df: pd.DataFrame,
    as_of: pd.Timestamp,
    event_priority: Dict[str, int]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply events up to 'as_of' date to the previous snapshot.
    Returns the updated snapshot and list of active employee IDs as of 'as_of'.
    """
    # Filter events by timestamp
    if 'event_time' in events_df.columns:
        from cost_model.state.schema import EVENT_TIME
        filtered = events_df[events_df[EVENT_TIME] <= as_of]
    else:
        filtered = events_df.copy()
    
    # Apply update from state snapshot
    from cost_model.state.snapshot import update as _update_snapshot

    updated_snapshot = _update_snapshot(prev_snapshot, filtered, as_of.year)
    
    # Determine active employees at year-end
    if 'active' in updated_snapshot.columns:
        year_end_employee_ids = updated_snapshot[updated_snapshot[EMP_ACTIVE]].index.tolist()
    else:
        year_end_employee_ids = []
    
    return updated_snapshot, year_end_employee_ids


def _extract_compensation_for_employee(
    emp_id: str,
    year_events: pd.DataFrame,
    simulation_year: int,
    logger: logging.Logger
) -> float:
    """
    Extract compensation for a terminated new hire employee using priority-based approach.

    Priority order:
    1. Most recent EVT_COMP event within the simulation year up to termination date
    2. Compensation from EVT_HIRE event value_json payload
    3. Level-based default compensation
    4. Global default compensation (50000.0)

    Args:
        emp_id: Employee ID to extract compensation for
        year_events: Events that occurred during the simulation year
        simulation_year: The simulation year being processed
        logger: Logger instance for debugging

    Returns:
        float: The compensation value for the employee
    """
    # This function maintains the original complex logic
    # TODO: Consider refactoring this into a compensation extraction service
    
    from cost_model.state.schema import (
        EMP_ID, EVT_TYPE, EVT_COMP, EVT_HIRE, EVENT_TIME, VALUE_JSON, EMP_LEVEL
    )
    import json
    
    DEFAULT_COMPENSATION = 50000.0
    
    logger.debug(f"Extracting compensation for employee {emp_id}")
    
    # Get events for this specific employee
    employee_events = year_events[year_events[EMP_ID] == emp_id].copy()
    
    if employee_events.empty:
        logger.info(f"No events found for employee {emp_id}, using default compensation: {DEFAULT_COMPENSATION}")
        return DEFAULT_COMPENSATION
    
    # Priority 1: Most recent EVT_COMP event
    comp_events = employee_events[employee_events[EVT_TYPE] == EVT_COMP]
    if not comp_events.empty:
        # Sort by event time and get the most recent
        comp_events_sorted = comp_events.sort_values(EVENT_TIME, ascending=False)
        latest_comp_event = comp_events_sorted.iloc[0]
        
        if VALUE_JSON in latest_comp_event and pd.notna(latest_comp_event[VALUE_JSON]):
            try:
                value_data = json.loads(latest_comp_event[VALUE_JSON])
                if 'new_compensation' in value_data:
                    compensation = float(value_data['new_compensation'])
                    logger.info(f"Found compensation from EVT_COMP event for employee {emp_id}: {compensation}")
                    return compensation
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Could not parse compensation from EVT_COMP event for employee {emp_id}: {e}")
    
    # Priority 2: Compensation from EVT_HIRE event
    hire_events = employee_events[employee_events[EVT_TYPE] == EVT_HIRE]
    if not hire_events.empty:
        hire_event = hire_events.iloc[0]  # Should be only one hire event
        
        if VALUE_JSON in hire_event and pd.notna(hire_event[VALUE_JSON]):
            try:
                value_data = json.loads(hire_event[VALUE_JSON])
                if 'compensation' in value_data:
                    compensation = float(value_data['compensation'])
                    logger.info(f"Found compensation from EVT_HIRE event for employee {emp_id}: {compensation}")
                    return compensation
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Could not parse compensation from EVT_HIRE event for employee {emp_id}: {e}")
    
    # Priority 3: Level-based default (if we have level information)
    if EMP_LEVEL in employee_events.columns:
        levels = employee_events[EMP_LEVEL].dropna()
        if not levels.empty:
            level = levels.iloc[0]
            # Simple level-based compensation mapping
            level_compensation_map = {
                1: 40000, 2: 50000, 3: 65000, 4: 85000, 5: 110000,
                6: 140000, 7: 180000, 8: 230000, 9: 300000, 10: 400000
            }
            if level in level_compensation_map:
                compensation = level_compensation_map[level]
                logger.info(f"Using level-based compensation for employee {emp_id} (level {level}): {compensation}")
                return compensation
    
    # Priority 4: Global default
    logger.info(f"Using global default compensation for employee {emp_id}: {DEFAULT_COMPENSATION}")
    return DEFAULT_COMPENSATION


def build_enhanced_yearly_snapshot(
    start_of_year_snapshot: pd.DataFrame,
    end_of_year_snapshot: pd.DataFrame,
    year_events: pd.DataFrame,
    simulation_year: int
) -> pd.DataFrame:
    """
    Build an enhanced yearly snapshot that includes all employees who were active
    at any point during the specified simulation year.
    
    This function delegates to the SnapshotBuilder for the core logic but maintains
    the original implementation for now to ensure compatibility.
    """
    return _snapshot_builder.build_enhanced_yearly_snapshot(
        start_of_year_snapshot,
        end_of_year_snapshot,
        year_events,
        simulation_year
    )


def _determine_employee_status_eoy(employee_row: pd.Series, simulation_year: int) -> str:
    """
    Determine the end-of-year status for an employee based on their data.
    
    Args:
        employee_row: Row containing employee data
        simulation_year: The simulation year
        
    Returns:
        Status string: 'Active', 'Terminated', or 'Unknown'
    """
    from cost_model.state.schema import EMP_ACTIVE, EMP_TERM_DATE
    
    # Check if employee is marked as active
    if EMP_ACTIVE in employee_row and pd.notna(employee_row[EMP_ACTIVE]):
        return 'Active' if employee_row[EMP_ACTIVE] else 'Terminated'
    
    # Check termination date
    if EMP_TERM_DATE in employee_row and pd.notna(employee_row[EMP_TERM_DATE]):
        term_date = pd.to_datetime(employee_row[EMP_TERM_DATE])
        if term_date.year == simulation_year:
            return 'Terminated'
    
    return 'Active'  # Default assumption


def consolidate_snapshots_to_parquet(snapshots_dir: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Consolidate multiple yearly snapshot files into a single Parquet file.
    
    Args:
        snapshots_dir: Directory containing yearly snapshot files
        output_path: Path for the consolidated output file
    """
    snapshots_dir = Path(snapshots_dir)
    output_path = Path(output_path)
    
    logger.info(f"Consolidating snapshots from {snapshots_dir} to {output_path}")
    
    # Find all snapshot files
    snapshot_files = list(snapshots_dir.glob("*snapshot*.parquet"))
    
    if not snapshot_files:
        logger.warning(f"No snapshot files found in {snapshots_dir}")
        return
    
    # Read and concatenate all snapshots
    all_snapshots = []
    for file_path in sorted(snapshot_files):
        try:
            snapshot_df = pd.read_parquet(file_path)
            all_snapshots.append(snapshot_df)
            logger.debug(f"Loaded snapshot from {file_path}: {len(snapshot_df)} records")
        except Exception as e:
            logger.error(f"Error reading snapshot file {file_path}: {e}")
    
    if not all_snapshots:
        logger.error("No valid snapshot files could be loaded")
        return
    
    # Concatenate all snapshots
    consolidated_df = pd.concat(all_snapshots, ignore_index=True)
    
    # Save consolidated file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_df.to_parquet(output_path, index=False)
    
    logger.info(f"Consolidated {len(snapshot_files)} snapshots into {output_path}: {len(consolidated_df)} total records")


# Backward compatibility exports
__all__ = [
    'create_initial_snapshot',
    'build_enhanced_yearly_snapshot', 
    'update_snapshot_with_events',
    'consolidate_snapshots_to_parquet',
    '_extract_compensation_for_employee',
    '_determine_employee_status_eoy'
]
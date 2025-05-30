# cost_model/engines/run_one_year/orchestrator/hiring.py
"""
Hiring orchestration module.

Handles all hiring-related logic including headcount target calculation,
hire event generation, and snapshot updates with new employees.
"""
import json
import logging
from typing import List, Optional, Tuple
import pandas as pd

from cost_model.engines import hire
from cost_model.state.schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP, EMP_LEVEL,
    EMP_ACTIVE, EMP_TENURE_BAND, EMP_DEFERRAL_RATE, EMP_TENURE,
    EMP_LEVEL_SOURCE, EMP_EXITED, SIMULATION_YEAR, EVENT_COLS
)
from cost_model.state.event_log import EVENT_PANDAS_DTYPES
from cost_model.utils.tenure_utils import standardize_tenure_band
from ..utils import compute_headcount_targets
from .base import YearContext, safe_get_meta, filter_valid_employee_ids


class HiringOrchestrator:
    """
    Orchestrates all hiring-related activities for a simulation year.

    This includes:
    - Computing headcount targets based on growth and attrition
    - Generating hire events through the hire engine
    - Creating new employee records in the snapshot
    - Handling compensation events for new hires
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the hiring orchestrator.

        Args:
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger or logging.getLogger(__name__)

    def get_events(
        self,
        snapshot: pd.DataFrame,
        year_context: YearContext,
        terminated_events: pd.DataFrame = None,
        start_count: Optional[int] = None
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        Generate hiring events and update the snapshot with new employees.

        Args:
            snapshot: Current workforce snapshot (survivors after terminations)
            year_context: Year-specific context and parameters
            terminated_events: Events from terminations (for hire engine context)

        Returns:
            Tuple of (event_list, updated_snapshot) where:
            - event_list: List of DataFrames [hire_events, comp_events]
            - updated_snapshot: Snapshot with new hires added
        """
        self.logger.info("[STEP] Computing headcount targets and generating hires")

        # Calculate headcount targets
        start_count_calc, survivor_count, target_eoy, net_hires, gross_hires = self._compute_targets(
            snapshot, year_context, start_count
        )

        self.logger.info(
            f"[HIRING] Start: {start_count_calc}, Survivors: {survivor_count}, "
            f"Net Hires: {net_hires}, Gross Hires: {gross_hires}, Target EOY: {target_eoy}"
        )

        if gross_hires <= 0:
            self.logger.info("[HIRING] No hires needed")
            return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)], snapshot

        # Generate hire events
        hire_events, comp_events = self._generate_hire_events(
            snapshot, gross_hires, year_context, terminated_events
        )

        # Update snapshot with new hires
        updated_snapshot = self._update_snapshot_with_hires(
            snapshot, hire_events, comp_events, year_context
        )

        self.logger.info(f"[HIRING] Generated {len(hire_events)} hires with {len(comp_events)} comp events")

        return [hire_events, comp_events], updated_snapshot

    def _compute_targets(
        self,
        snapshot: pd.DataFrame,
        year_context: YearContext,
        start_count: Optional[int] = None
    ) -> Tuple[int, int, int, int, int]:
        """
        Compute hiring targets based on growth parameters and attrition.

        Args:
            snapshot: Current workforce snapshot
            year_context: Year-specific context

        Returns:
            Tuple of (start_count, survivor_count, target_eoy, net_hires, gross_hires)
        """
        # Get parameters from global_params
        target_growth = getattr(year_context.global_params, 'target_growth', 0.0)
        nh_term_rate = getattr(year_context.global_params, 'new_hire_termination_rate', 0.0)

        # Calculate counts
        survivor_count = snapshot[EMP_ACTIVE].sum() if EMP_ACTIVE in snapshot.columns else len(snapshot)

        # Use provided start count or estimate it
        if start_count is None:
            # Estimate start count (this is a simplification - ideally passed from caller)
            start_count = int(survivor_count / (1 - 0.1))  # Assume ~10% attrition for estimation

        target_eoy, net_hires, gross_hires = compute_headcount_targets(
            start_count, survivor_count, target_growth, nh_term_rate
        )

        return start_count, survivor_count, target_eoy, net_hires, gross_hires

    def _generate_hire_events(
        self,
        snapshot: pd.DataFrame,
        gross_hires: int,
        year_context: YearContext,
        terminated_events: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate hire and compensation events using the hire engine.

        Args:
            snapshot: Current workforce snapshot
            gross_hires: Number of gross hires to generate
            year_context: Year-specific context
            terminated_events: Termination events for context

        Returns:
            Tuple of (hire_events, comp_events)
        """
        # Standardize tenure bands in hazard slice
        hz_slice = year_context.hazard_slice.copy()
        if EMP_TENURE_BAND in hz_slice.columns:
            hz_slice[EMP_TENURE_BAND] = hz_slice[EMP_TENURE_BAND].apply(standardize_tenure_band)
            self.logger.info(
                f"[HIRING] Standardized hazard slice tenure bands: "
                f"{hz_slice[EMP_TENURE_BAND].unique().tolist()}"
            )

        # Call the hire engine
        # Handle terminated_events properly to avoid DataFrame boolean evaluation
        if terminated_events is None or terminated_events.empty:
            term_events_param = pd.DataFrame()
        else:
            term_events_param = terminated_events

        hires_result = hire.run(
            snapshot=snapshot,
            hires_to_make=gross_hires,
            hazard_slice=hz_slice,
            rng=year_context.year_rng,
            census_template_path=year_context.census_template_path,
            global_params=year_context.global_params,
            terminated_events=term_events_param
        )

        # Extract events from result
        if len(hires_result) >= 2:
            hire_events = hires_result[0]
            comp_events = hires_result[1]
        else:
            hire_events = hires_result[0] if hires_result else pd.DataFrame(columns=EVENT_COLS)
            comp_events = pd.DataFrame(columns=EVENT_COLS)

        # Validate and clean events
        hire_events = self._validate_events(hire_events, "hire")
        comp_events = self._validate_events(comp_events, "compensation")

        return hire_events, comp_events

    def _validate_events(self, events: pd.DataFrame, event_type: str) -> pd.DataFrame:
        """
        Validate and clean event DataFrames.

        Args:
            events: Events DataFrame to validate
            event_type: Type of events for logging

        Returns:
            Cleaned events DataFrame
        """
        if events.empty:
            return events

        # Filter out invalid employee IDs
        events = filter_valid_employee_ids(events, self.logger)

        # Ensure proper dtypes
        for col, dtype in EVENT_PANDAS_DTYPES.items():
            if col in events.columns:
                try:
                    events[col] = events[col].astype(dtype)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Could not convert {event_type} events column {col} to {dtype}: {e}")

        return events

    def _update_snapshot_with_hires(
        self,
        snapshot: pd.DataFrame,
        hire_events: pd.DataFrame,
        comp_events: pd.DataFrame,
        year_context: YearContext
    ) -> pd.DataFrame:
        """
        Update the snapshot with new hire records.

        Args:
            snapshot: Current workforce snapshot
            hire_events: Hire events DataFrame
            comp_events: Compensation events DataFrame
            year_context: Year-specific context

        Returns:
            Updated snapshot with new hires
        """
        if hire_events.empty:
            return snapshot

        # Extract compensation values from comp events
        compensation_map = {}
        if not comp_events.empty:
            for _, row in comp_events.iterrows():
                emp_id = row.get(EMP_ID)
                comp_value = row.get('value_num')
                if emp_id and pd.notna(comp_value):
                    compensation_map[emp_id] = comp_value

        # Extract employee data from hire events
        employee_ids = hire_events[EMP_ID].tolist()

        # Get compensation values, with fallback logic
        compensation_values = []
        for emp_id in employee_ids:
            comp = compensation_map.get(emp_id)
            if comp is None:
                # Try to extract from hire event meta or value_json
                hire_row = hire_events[hire_events[EMP_ID] == emp_id].iloc[0]
                comp = self._extract_compensation_from_hire_event(hire_row)
            compensation_values.append(comp or 50000.0)  # Default fallback

        # Extract job levels from hire events
        extracted_levels = []
        for _, row in hire_events.iterrows():
            level = self._extract_level_from_hire_event(row)
            extracted_levels.append(level)

        extracted_levels = pd.Series(extracted_levels)

        # Create new hire records
        new_hires_data = {
            EMP_ID: employee_ids,
            EMP_HIRE_DATE: pd.to_datetime(hire_events['event_time']),
            EMP_BIRTH_DATE: pd.to_datetime('1990-01-01'),  # Default
            # EMP_ROLE removed as part of schema refactoring
            EMP_GROSS_COMP: compensation_values,
            'employee_termination_date': pd.NaT,
            EMP_ACTIVE: True,
            EMP_DEFERRAL_RATE: 0.0,
            EMP_TENURE_BAND: '<1',  # New hires
            EMP_TENURE: 0.0,
            EMP_LEVEL: pd.to_numeric(extracted_levels, errors='coerce').fillna(1).astype('Int64'),
            EMP_LEVEL_SOURCE: 'new_hire',
            EMP_EXITED: False,
            SIMULATION_YEAR: year_context.year
        }

        # Try to get birth date from meta if available (role removed as part of schema refactoring)
        if 'meta' in hire_events.columns:
            new_hires_data[EMP_BIRTH_DATE] = hire_events['meta'].apply(
                lambda x: pd.to_datetime(safe_get_meta(x, 'birth_date', '1990-01-01'))
            )
            # Role extraction removed as part of schema refactoring

        # Create new hires DataFrame
        new_hires = pd.DataFrame(new_hires_data)
        new_hires = new_hires.set_index(EMP_ID, drop=False)

        # Combine with existing snapshot
        snapshot_copy = snapshot.copy()
        if snapshot_copy.index.name != EMP_ID:
            if EMP_ID in snapshot_copy.columns:
                snapshot_copy = snapshot_copy.set_index(EMP_ID, drop=False)

        # Concatenate snapshots
        combined_snapshot = pd.concat([snapshot_copy, new_hires], ignore_index=False)

        # Ensure no duplicate employee IDs
        if combined_snapshot.index.duplicated().any():
            self.logger.warning("Found duplicate employee IDs after adding hires, removing duplicates")
            combined_snapshot = combined_snapshot[~combined_snapshot.index.duplicated(keep='first')]

        return combined_snapshot

    def _extract_compensation_from_hire_event(self, hire_row: pd.Series) -> Optional[float]:
        """Extract compensation from a hire event row."""
        # Try value_num first
        if 'value_num' in hire_row and pd.notna(hire_row['value_num']):
            return float(hire_row['value_num'])

        # Try value_json
        if 'value_json' in hire_row and pd.notna(hire_row['value_json']):
            try:
                value_data = json.loads(hire_row['value_json'])
                if isinstance(value_data, dict):
                    return value_data.get('compensation') or value_data.get('salary')
                elif isinstance(value_data, (int, float)):
                    return float(value_data)
            except (json.JSONDecodeError, TypeError):
                pass

        # Try meta
        if 'meta' in hire_row:
            comp = safe_get_meta(hire_row['meta'], 'compensation')
            if comp is not None:
                return float(comp)

        return None

    def _extract_level_from_hire_event(self, hire_row: pd.Series) -> Optional[int]:
        """Extract job level from a hire event row."""
        # Try value_json first
        if 'value_json' in hire_row and pd.notna(hire_row['value_json']):
            try:
                value_data = json.loads(hire_row['value_json'])
                if isinstance(value_data, dict):
                    level = value_data.get('level') or value_data.get('job_level')
                    if level is not None:
                        return int(level)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # Try meta
        if 'meta' in hire_row:
            level = safe_get_meta(hire_row['meta'], 'level')
            if level is not None:
                try:
                    return int(level)
                except (ValueError, TypeError):
                    pass

        return None
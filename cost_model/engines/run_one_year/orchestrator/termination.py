# cost_model/engines/run_one_year/orchestrator/termination.py
"""
Termination orchestration module.

Handles both experienced employee terminations and new hire terminations,
generating appropriate termination and compensation events.
"""
import logging
from typing import List, Optional, Tuple
import pandas as pd

from cost_model.engines import term
from cost_model.engines.nh_termination import run_new_hires
from cost_model.state.schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_TENURE_BAND, EVENT_COLS
)
from cost_model.state.event_log import EVENT_PANDAS_DTYPES
from cost_model.utils.tenure_utils import standardize_tenure_band
from .base import YearContext, filter_valid_employee_ids


class TerminationOrchestrator:
    """
    Orchestrates all termination-related activities for a simulation year.

    This includes:
    - Experienced employee terminations using hazard-based rates
    - New hire terminations using deterministic rates
    - Generating termination and compensation events
    - Updating snapshots to remove terminated employees
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the termination orchestrator.

        Args:
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger or logging.getLogger(__name__)

    def get_experienced_termination_events(
        self,
        snapshot: pd.DataFrame,
        year_context: YearContext
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        Generate termination events for experienced employees and update snapshot.

        Args:
            snapshot: Current workforce snapshot
            year_context: Year-specific context and parameters

        Returns:
            Tuple of (event_list, updated_snapshot) where:
            - event_list: List of termination event DataFrames
            - updated_snapshot: Snapshot with terminated employees removed
        """
        self.logger.info("[STEP] Hazard-based terminations (experienced only)")

        # Filter to experienced employees only (hired before this year)
        experienced_mask = snapshot[EMP_HIRE_DATE] < year_context.as_of
        experienced = snapshot[experienced_mask].copy()

        if experienced.empty:
            self.logger.info("[TERMINATION] No experienced employees to process")
            return [pd.DataFrame(columns=EVENT_COLS)], snapshot

        self.logger.info(f"[TERMINATION] Processing {len(experienced)} experienced employees")

        # Prepare hazard slice
        hz_slice = self._prepare_hazard_slice(year_context.hazard_slice)

        # Run termination engine
        term_event_dfs = term.run(
            snapshot=experienced,
            hazard_slice=hz_slice,
            rng=year_context.year_rng,
            deterministic=year_context.deterministic_term
        )

        # Validate and process events
        term_events = self._process_termination_events(term_event_dfs)

        # Update snapshot to remove terminated employees
        updated_snapshot = self._remove_terminated_employees(snapshot, term_events)

        self.logger.info(f"[TERMINATION] Generated {len(term_events)} termination events")

        return [term_events], updated_snapshot

    def get_new_hire_termination_events(
        self,
        snapshot: pd.DataFrame,
        year_context: YearContext
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        Generate termination events for new hires and update snapshot.

        Args:
            snapshot: Current workforce snapshot (with new hires)
            year_context: Year-specific context and parameters

        Returns:
            Tuple of (event_list, updated_snapshot) where:
            - event_list: List of [termination_events, compensation_events]
            - updated_snapshot: Snapshot with terminated new hires removed
        """
        self.logger.info("[STEP] New hire terminations")

        # Prepare hazard slice
        hz_slice = self._prepare_hazard_slice(year_context.hazard_slice)

        # Run new hire termination engine
        nh_term_events, nh_term_comp_events = run_new_hires(
            snapshot, hz_slice, year_context.year_rng, year_context.year, deterministic=True
        )

        # Validate events
        nh_term_events = self._validate_events(nh_term_events, "new hire termination")
        nh_term_comp_events = self._validate_events(nh_term_comp_events, "new hire compensation")

        # Update snapshot to remove terminated new hires
        terminated_ids = set(nh_term_events[EMP_ID]) if not nh_term_events.empty else set()

        # Handle both index-based and column-based employee ID filtering
        if EMP_ID in snapshot.columns:
            updated_snapshot = snapshot[~snapshot[EMP_ID].isin(terminated_ids)].copy()
        else:
            # If EMP_ID is in the index
            updated_snapshot = snapshot.loc[~snapshot.index.isin(terminated_ids)].copy()

        self.logger.info(f"[NH-TERM] Terminated {len(terminated_ids)} new hires")

        return [nh_term_events, nh_term_comp_events], updated_snapshot

    def _prepare_hazard_slice(self, hazard_slice: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the hazard slice for termination processing.

        Args:
            hazard_slice: Raw hazard slice for the year

        Returns:
            Processed hazard slice with standardized tenure bands
        """
        hz_slice = hazard_slice.copy()

        # Standardize tenure bands if present
        if EMP_TENURE_BAND in hz_slice.columns:
            hz_slice[EMP_TENURE_BAND] = hz_slice[EMP_TENURE_BAND].apply(standardize_tenure_band)
            self.logger.info(
                f"[TERMINATION] Standardized hazard slice tenure bands: "
                f"{hz_slice[EMP_TENURE_BAND].unique().tolist()}"
            )

        return hz_slice

    def _process_termination_events(self, term_event_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Process and consolidate termination events from the term engine.

        Args:
            term_event_dfs: List of event DataFrames from term.run()

        Returns:
            Consolidated termination events DataFrame
        """
        if not term_event_dfs:
            self.logger.debug("[TERMINATION] No event DataFrames list provided to process.")
            return pd.DataFrame(columns=EVENT_COLS)

        # Enhanced filtering with explicit logging for debugging
        valid_dfs_to_concat = []
        for i, df in enumerate(term_event_dfs):
            if not isinstance(df, pd.DataFrame):
                self.logger.debug(f"[TERMINATION] Skipping non-DataFrame at index {i}")
                continue
            if df.empty:
                self.logger.debug(f"[TERMINATION] Skipping empty DataFrame at index {i}")
                continue
            if df.isna().all().all():
                self.logger.debug(f"[TERMINATION] Skipping all-NA DataFrame at index {i}")
                continue
                
            # Clean the DataFrame by removing all-NA columns BEFORE adding to the list
            df_clean = df.copy()
            all_na_cols = df_clean.columns[df_clean.isna().all()]
            if not all_na_cols.empty:
                self.logger.debug(f"[TERMINATION] Dropping {len(all_na_cols)} all-NA columns from DataFrame at index {i}")
                df_clean = df_clean.drop(columns=all_na_cols)
                
            # Only add if the cleaned DataFrame is not empty
            if not df_clean.empty:
                valid_dfs_to_concat.append(df_clean)
            else:
                self.logger.debug(f"[TERMINATION] DataFrame at index {i} became empty after cleaning, skipping")
                
        self.logger.debug(f"[TERMINATION] {len(valid_dfs_to_concat)} DataFrames will be concatenated.")

        if valid_dfs_to_concat:
            try:
                # Use concat with sort=False to avoid the FutureWarning about column ordering
                # and ignore_index=True for clean row indexing
                term_events = pd.concat(valid_dfs_to_concat, ignore_index=True, sort=False)
                self.logger.debug(f"[TERMINATION] Concatenated {len(valid_dfs_to_concat)} non-empty event DataFrame(s).")
            except Exception as e:
                self.logger.error(f"Error during concatenation of termination event DataFrames: {e}", exc_info=True)
                # Fallback to an empty DataFrame with defined columns
                term_events = pd.DataFrame(columns=EVENT_COLS)
        else:
            self.logger.debug("[TERMINATION] No valid DataFrames to concatenate; returning empty DataFrame.")
            term_events = pd.DataFrame(columns=EVENT_COLS)

        
        # Validate and clean the (potentially empty) concatenated DataFrame
        term_events = self._validate_events(term_events, "termination")

        return term_events

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
            return events # Return the empty DataFrame as is

        # Filter out invalid employee IDs
        # Ensure a fresh copy if filter_valid_employee_ids modifies or returns a view
        events = filter_valid_employee_ids(events, self.logger).copy()


        # Ensure proper dtypes
        for col, dtype in EVENT_PANDAS_DTYPES.items():
            if col in events.columns:
                try:
                    # Ensure column is not all NA before astype if dtype is non-nullable
                    # and NA handling for that dtype is problematic
                    if events[col].isna().all() and pd.api.types.is_integer_dtype(dtype):
                         # Potentially skip astype for all-NA int columns or handle differently
                         # For now, let astype attempt it; it might go to float if NA remains.
                         # Or use nullable int types like 'Int64' if that's the target.
                         pass # Let it try, or add specific handling if 'Int64' etc. are used

                    events[col] = events[col].astype(dtype)
                except (ValueError, TypeError, pd.errors.IntCastingNaNError) as e: # Added IntCastingNaNError
                    self.logger.warning(
                        f"Could not convert {event_type} events column '{col}' to {dtype}: {e}. "
                        f"Column head: {events[col].head().tolist()}"
                    )
        return events

    def _remove_terminated_employees(
        self,
        snapshot: pd.DataFrame,
        term_events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Remove terminated employees from the snapshot.

        Args:
            snapshot: Current workforce snapshot
            term_events: Termination events DataFrame

        Returns:
            Updated snapshot with terminated employees removed
        """
        if term_events.empty or EMP_ID not in term_events.columns:
            # If no termination events or no EMP_ID column in events, return original snapshot
            if EMP_ID not in term_events.columns and not term_events.empty:
                self.logger.warning(f"[TERMINATION] EMP_ID column missing in term_events. Cannot remove employees.")
            return snapshot

        # Get terminated employee IDs
        terminated_ids = set(term_events[EMP_ID].unique()) # .unique() handles potential duplicates

        if not terminated_ids:
            return snapshot # No valid IDs to remove

        # Filter out terminated employees - assumes EMP_ID is a column in snapshot
        if EMP_ID not in snapshot.columns:
            self.logger.error(f"[TERMINATION] EMP_ID column missing in snapshot. Cannot remove terminated employees.")
            return snapshot # Or handle if EMP_ID could be index

        survivors = snapshot[~snapshot[EMP_ID].isin(terminated_ids)].copy()

        self.logger.info(
            f"[TERMINATION] Processed removal of {len(terminated_ids)} unique terminated employee IDs. "
            f"Snapshot size before: {len(snapshot)}, after: {len(survivors)}"
        )

        return survivors
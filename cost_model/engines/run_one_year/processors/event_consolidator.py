# cost_model/engines/run_one_year/processors/event_consolidator.py
"""
Processor for consolidating and managing events during yearly simulation.
"""
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cost_model.state.event_log import EVENT_COLS, EVENT_PANDAS_DTYPES
from cost_model.state.schema import EMP_ID, EVENT_TIME, SIMULATION_YEAR

from .base import BaseProcessor, ProcessorResult


class EventConsolidator(BaseProcessor):
    """Handles consolidation and management of simulation events."""

    def consolidate_events(
        self,
        promotion_events: pd.DataFrame,
        termination_events: pd.DataFrame,
        hiring_events: pd.DataFrame,
        contribution_events: pd.DataFrame,
        new_hire_termination_events: pd.DataFrame,
        year: int,
        **kwargs,
    ) -> ProcessorResult:
        """
        Consolidate all events from different processors into a unified event log.

        Args:
            promotion_events: Events from promotion processor
            termination_events: Events from termination processor
            hiring_events: Events from hiring processor
            contribution_events: Events from contribution processor
            new_hire_termination_events: Events from new hire termination processor
            year: Simulation year

        Returns:
            ProcessorResult with consolidated event log
        """
        self.log_step_start(
            "Build event log",
            promotion_events=len(promotion_events),
            termination_events=len(termination_events),
            hiring_events=len(hiring_events),
            contribution_events=len(contribution_events),
            new_hire_termination_events=len(new_hire_termination_events),
            year=year,
        )

        result = ProcessorResult()

        try:
            # Collect all event DataFrames
            event_dataframes = [
                ("promotion", promotion_events),
                ("termination", termination_events),
                ("hiring", hiring_events),
                ("contribution", contribution_events),
                ("new_hire_termination", new_hire_termination_events),
            ]

            # Filter out empty DataFrames and log counts
            valid_events = []
            total_events = 0

            for event_type, events_df in event_dataframes:
                if not events_df.empty:
                    valid_events.append(events_df)
                    total_events += len(events_df)
                    self.logger.info(f"  {event_type}: {len(events_df)} events")
                else:
                    self.logger.info(f"  {event_type}: 0 events")

            if not valid_events:
                self.logger.info("No events to consolidate")
                result.data = self._create_empty_event_log()
                return result

            # Concatenate all events
            consolidated_events = pd.concat(valid_events, ignore_index=True)

            # Standardize event log format
            standardized_events = self._standardize_event_format(consolidated_events, year)

            # Validate event log
            validation_result = self._validate_event_log(standardized_events)
            if not validation_result["valid"]:
                for warning in validation_result["warnings"]:
                    result.add_warning(warning)

            self.logger.info(f"Consolidated {total_events} events into unified event log")

            result.data = standardized_events
            result.add_metadata("total_events", total_events)
            result.add_metadata(
                "event_types_included", [et for et, df in event_dataframes if not df.empty]
            )

        except Exception as e:
            self.logger.error(f"Error during event consolidation: {e}", exc_info=True)
            result.add_error(f"Event consolidation failed: {str(e)}")
            result.data = self._create_empty_event_log()

        self.log_step_end(
            "Build event log",
            total_events=len(result.data) if isinstance(result.data, pd.DataFrame) else 0,
            success=result.success,
        )

        return result

    def _standardize_event_format(self, events: pd.DataFrame, year: int) -> pd.DataFrame:
        """
        Standardize event format to match expected schema.

        Args:
            events: Raw consolidated events
            year: Simulation year

        Returns:
            Standardized event DataFrame
        """
        self.logger.info("Standardizing event format")

        if events.empty:
            return self._create_empty_event_log()

        # Ensure required columns exist
        standardized = events.copy()

        # Add missing columns with defaults
        for col in EVENT_COLS:
            if col not in standardized.columns:
                if col == SIMULATION_YEAR:
                    standardized[col] = year
                elif col == EVENT_TIME:
                    # Set default event time to end of year
                    standardized[col] = pd.Timestamp(f"{year}-12-31")
                elif col == "event_id":
                    # Generate unique event IDs
                    standardized[col] = [str(uuid.uuid4()) for _ in range(len(standardized))]
                else:
                    # Set appropriate default based on column name
                    if "id" in col.lower():
                        standardized[col] = ""
                    elif "time" in col.lower() or "date" in col.lower():
                        standardized[col] = pd.NaT
                    elif any(
                        numeric_word in col.lower() for numeric_word in ["amount", "rate", "value"]
                    ):
                        standardized[col] = 0.0
                    else:
                        standardized[col] = pd.NA

        # Apply correct data types
        for col, dtype in EVENT_PANDAS_DTYPES.items():
            if col in standardized.columns:
                try:
                    standardized[col] = standardized[col].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"Could not apply dtype {dtype} to column {col}: {e}")

        # Reorder columns to match expected schema
        final_columns = [col for col in EVENT_COLS if col in standardized.columns]
        extra_columns = [col for col in standardized.columns if col not in EVENT_COLS]

        if extra_columns:
            self.logger.info(f"Preserving extra columns: {extra_columns}")
            final_columns.extend(extra_columns)

        standardized = standardized[final_columns]

        self.logger.info(
            f"Standardized {len(standardized)} events with {len(final_columns)} columns"
        )

        return standardized

    def _validate_event_log(self, events: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the consolidated event log.

        Args:
            events: Event DataFrame to validate

        Returns:
            Validation result dictionary
        """
        validation_result = {"valid": True, "warnings": [], "errors": []}

        if events.empty:
            return validation_result

        # Check for required columns
        missing_cols = [col for col in ["event_id", EMP_ID] if col not in events.columns]
        if missing_cols:
            validation_result["errors"].append(f"Missing required columns: {missing_cols}")
            validation_result["valid"] = False

        # Check for null employee IDs
        if EMP_ID in events.columns:
            null_emp_ids = events[EMP_ID].isna().sum()
            if null_emp_ids > 0:
                validation_result["warnings"].append(
                    f"Found {null_emp_ids} events with null employee IDs"
                )

        # Check for duplicate event IDs
        if "event_id" in events.columns:
            duplicate_ids = events["event_id"].duplicated().sum()
            if duplicate_ids > 0:
                validation_result["warnings"].append(f"Found {duplicate_ids} duplicate event IDs")

        # Check event time consistency
        if EVENT_TIME in events.columns:
            invalid_times = events[EVENT_TIME].isna().sum()
            if invalid_times > 0:
                validation_result["warnings"].append(
                    f"Found {invalid_times} events with invalid times"
                )

        return validation_result

    def _create_empty_event_log(self) -> pd.DataFrame:
        """Create an empty event log with proper schema."""
        empty_events = pd.DataFrame(columns=EVENT_COLS)

        # Apply correct data types
        for col, dtype in EVENT_PANDAS_DTYPES.items():
            if col in empty_events.columns:
                empty_events[col] = empty_events[col].astype(dtype)

        return empty_events

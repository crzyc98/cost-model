# cost_model/engines/run_one_year/orchestrator/event_consolidation.py
"""
Event consolidation utilities for orchestrator.
Handles the complex logic of combining and validating simulation events.
"""
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from cost_model.state.event_log import EVENT_COLS
from cost_model.state.schema import EMP_ID, EVENT_TIME, SIMULATION_YEAR


def consolidate_events(
    promotion_events: pd.DataFrame,
    termination_events: pd.DataFrame,
    hiring_events: pd.DataFrame,
    nh_termination_events: pd.DataFrame,
    compensation_events: pd.DataFrame,
    contribution_events: pd.DataFrame,
    year: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Consolidate all simulation events into a unified event log.

    Args:
        promotion_events: Events from promotion processing
        termination_events: Events from experienced termination processing
        hiring_events: Events from hiring processing
        nh_termination_events: Events from new hire termination processing
        compensation_events: Events from compensation processing
        contribution_events: Events from contribution processing
        year: Simulation year
        logger: Logger for consolidation information

    Returns:
        Consolidated event DataFrame with standardized format
    """
    logger.info("[EVENT CONSOLIDATION] Consolidating simulation events")

    # Collect all event DataFrames with their types
    event_sources = [
        ("promotion", promotion_events),
        ("termination", termination_events),
        ("hiring", hiring_events),
        ("nh_termination", nh_termination_events),
        ("compensation", compensation_events),
        ("contribution", contribution_events),
    ]

    # Filter out empty DataFrames and log counts
    valid_events = []
    total_event_count = 0

    for event_type, events_df in event_sources:
        if not events_df.empty:
            valid_events.append(events_df)
            event_count = len(events_df)
            total_event_count += event_count
            logger.info(f"  {event_type}: {event_count} events")
        else:
            logger.info(f"  {event_type}: 0 events")

    if not valid_events:
        logger.info("No events to consolidate, creating empty event log")
        return create_empty_event_log()

    # Concatenate all events
    try:
        consolidated_events = pd.concat(valid_events, ignore_index=True)
        logger.info(f"Concatenated {total_event_count} events from {len(valid_events)} sources")
    except Exception as e:
        logger.error(f"Error concatenating events: {e}")
        return create_empty_event_log()

    # Standardize event format
    standardized_events = standardize_event_format(consolidated_events, year, logger)

    # Validate consolidated events
    validation_result = validate_event_log(standardized_events, logger)
    if not validation_result["valid"]:
        logger.warning(f"Event validation warnings: {validation_result['warnings']}")

    logger.info(f"Successfully consolidated {len(standardized_events)} events")
    return standardized_events


def standardize_event_format(
    events: pd.DataFrame, year: int, logger: logging.Logger
) -> pd.DataFrame:
    """
    Standardize event format to match expected schema.

    Args:
        events: Raw consolidated events
        year: Simulation year
        logger: Logger for standardization information

    Returns:
        Standardized event DataFrame
    """
    if events.empty:
        return create_empty_event_log()

    logger.info("Standardizing event format")
    standardized = events.copy()

    # Ensure required columns exist with appropriate defaults
    column_defaults = {
        SIMULATION_YEAR: year,
        EVENT_TIME: pd.Timestamp(f"{year}-12-31"),
        "event_id": lambda: str(uuid.uuid4()),
    }

    for col in EVENT_COLS:
        if col not in standardized.columns:
            if col in column_defaults:
                default_value = column_defaults[col]
                if callable(default_value):
                    # Generate unique values for each row
                    standardized[col] = [default_value() for _ in range(len(standardized))]
                else:
                    standardized[col] = default_value
            else:
                # Set appropriate default based on column characteristics
                standardized[col] = get_default_value_for_column(col)

    # Reorder columns to match expected schema
    final_columns = [col for col in EVENT_COLS if col in standardized.columns]
    extra_columns = [col for col in standardized.columns if col not in EVENT_COLS]

    if extra_columns:
        logger.debug(f"Preserving extra columns: {extra_columns}")
        final_columns.extend(extra_columns)

    standardized = standardized[final_columns]

    logger.info(f"Standardized events: {len(standardized)} rows, {len(final_columns)} columns")
    return standardized


def get_default_value_for_column(col_name: str) -> Any:
    """
    Get appropriate default value for a column based on its name.

    Args:
        col_name: Name of the column

    Returns:
        Appropriate default value
    """
    col_lower = col_name.lower()

    if "id" in col_lower:
        return ""
    elif any(time_word in col_lower for time_word in ["time", "date"]):
        return pd.NaT
    elif any(
        numeric_word in col_lower for numeric_word in ["amount", "rate", "value", "compensation"]
    ):
        return 0.0
    elif "active" in col_lower or "eligible" in col_lower:
        return False
    else:
        return pd.NA


def validate_event_log(events: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """
    Validate the consolidated event log for integrity and consistency.

    Args:
        events: Event DataFrame to validate
        logger: Logger for validation messages

    Returns:
        Dictionary with validation results
    """
    validation_result = {"valid": True, "errors": [], "warnings": []}

    if events.empty:
        logger.info("Empty event log - validation passed")
        return validation_result

    logger.info(f"Validating event log with {len(events)} events")

    # Check for required columns
    required_columns = ["event_id", EMP_ID]
    missing_columns = [col for col in required_columns if col not in events.columns]
    if missing_columns:
        error_msg = f"Missing required columns: {missing_columns}"
        validation_result["errors"].append(error_msg)
        validation_result["valid"] = False
        logger.error(error_msg)

    # Check for null employee IDs
    if EMP_ID in events.columns:
        null_emp_ids = events[EMP_ID].isna().sum()
        if null_emp_ids > 0:
            warning_msg = f"Found {null_emp_ids} events with null employee IDs"
            validation_result["warnings"].append(warning_msg)
            logger.warning(warning_msg)

    # Check for duplicate event IDs
    if "event_id" in events.columns:
        duplicate_ids = events["event_id"].duplicated().sum()
        if duplicate_ids > 0:
            warning_msg = f"Found {duplicate_ids} duplicate event IDs"
            validation_result["warnings"].append(warning_msg)
            logger.warning(warning_msg)

    # Check event time consistency
    if EVENT_TIME in events.columns:
        invalid_times = events[EVENT_TIME].isna().sum()
        if invalid_times > 0:
            warning_msg = f"Found {invalid_times} events with invalid times"
            validation_result["warnings"].append(warning_msg)
            logger.warning(warning_msg)

    # Check simulation year consistency
    if SIMULATION_YEAR in events.columns:
        year_values = events[SIMULATION_YEAR].unique()
        if len(year_values) > 1:
            warning_msg = f"Multiple simulation years in events: {year_values}"
            validation_result["warnings"].append(warning_msg)
            logger.warning(warning_msg)

    # Log validation summary
    if validation_result["valid"] and not validation_result["warnings"]:
        logger.info("Event log validation passed successfully")
    elif validation_result["valid"]:
        logger.info(
            f"Event log validation passed with {len(validation_result['warnings'])} warnings"
        )
    else:
        logger.error(f"Event log validation failed with {len(validation_result['errors'])} errors")

    return validation_result


def create_empty_event_log() -> pd.DataFrame:
    """
    Create an empty event log with proper schema.

    Returns:
        Empty DataFrame with correct columns and types
    """
    return pd.DataFrame(columns=EVENT_COLS)


def merge_event_logs(
    base_log: pd.DataFrame, new_events: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    """
    Merge new events into existing event log.

    Args:
        base_log: Existing event log
        new_events: New events to add
        logger: Logger for merge information

    Returns:
        Merged event log
    """
    if new_events.empty:
        logger.info("No new events to merge")
        return base_log.copy()

    if base_log.empty:
        logger.info(f"Creating new event log with {len(new_events)} events")
        return new_events.copy()

    try:
        merged_log = pd.concat([base_log, new_events], ignore_index=True)
        logger.info(
            f"Merged event logs: {len(base_log)} + {len(new_events)} = {len(merged_log)} events"
        )
        return merged_log
    except Exception as e:
        logger.error(f"Error merging event logs: {e}")
        return base_log.copy()


class EventConsolidationManager:
    """
    Manager class for handling event consolidation across orchestrator steps.
    """

    def __init__(self, logger: logging.Logger, year: int):
        self.logger = logger
        self.year = year
        self.accumulated_events = []

    def add_events(self, events: Union[pd.DataFrame, List[pd.DataFrame]], event_type: str) -> None:
        """
        Add events from a processing step.

        Args:
            events: Events DataFrame or list of DataFrames
            event_type: Description of event type/source
        """
        # Handle both single DataFrame and list of DataFrames
        if isinstance(events, list):
            # If it's a list, add each DataFrame separately
            total_events = 0
            for i, event_df in enumerate(events):
                if isinstance(event_df, pd.DataFrame) and not event_df.empty:
                    sub_type = f"{event_type}_{i}" if len(events) > 1 else event_type
                    self.accumulated_events.append((sub_type, event_df))
                    total_events += len(event_df)

            if total_events > 0:
                self.logger.info(f"Added {total_events} {event_type} events from {len(events)} DataFrames")
            else:
                self.logger.info(f"No {event_type} events to add (empty list or all empty DataFrames)")
        else:
            # Handle single DataFrame
            if isinstance(events, pd.DataFrame):
                if not events.empty:
                    self.accumulated_events.append((event_type, events))
                    self.logger.info(f"Added {len(events)} {event_type} events")
                else:
                    self.logger.info(f"No {event_type} events to add")
            else:
                self.logger.warning(f"Invalid events type for {event_type}: {type(events)}")
                self.logger.info(f"No {event_type} events to add")

    def consolidate_all(self) -> pd.DataFrame:
        """
        Consolidate all accumulated events.

        Returns:
            Consolidated event DataFrame
        """
        if not self.accumulated_events:
            self.logger.info("No events accumulated for consolidation")
            return create_empty_event_log()

        # Separate events by type
        event_dict = {}
        for event_type, events in self.accumulated_events:
            if event_type not in event_dict:
                event_dict[event_type] = []
            event_dict[event_type].append(events)

        # Concatenate events of same type
        consolidated_by_type = {}
        for event_type, events_list in event_dict.items():
            if len(events_list) == 1:
                consolidated_by_type[event_type] = events_list[0]
            else:
                consolidated_by_type[event_type] = pd.concat(events_list, ignore_index=True)

        # Consolidate sub-types back to main types (e.g., "hiring_0", "hiring_1" -> "hiring")
        main_event_types = {
            "promotion": [],
            "termination": [],
            "hiring": [],
            "nh_termination": [],
            "compensation": [],
            "contribution": [],
        }
        
        # Group events by main type, including sub-types
        for event_type, events in consolidated_by_type.items():
            # Extract main type from sub-types (e.g., "hiring_0" -> "hiring")
            main_type = event_type.split('_')[0]
            if main_type in main_event_types:
                main_event_types[main_type].append(events)
            else:
                # Handle unknown event types by treating them as their own type
                main_event_types[event_type] = [events]

        # Concatenate events of the same main type
        final_consolidated = {}
        for main_type, events_list in main_event_types.items():
            if events_list:
                if len(events_list) == 1:
                    final_consolidated[main_type] = events_list[0]
                else:
                    final_consolidated[main_type] = pd.concat(events_list, ignore_index=True)
                    self.logger.info(f"Consolidated {len(events_list)} sub-types for {main_type}: {sum(len(df) for df in events_list)} total events")
            else:
                final_consolidated[main_type] = pd.DataFrame()

        # Final consolidation
        return consolidate_events(
            promotion_events=final_consolidated.get("promotion", pd.DataFrame()),
            termination_events=final_consolidated.get("termination", pd.DataFrame()),
            hiring_events=final_consolidated.get("hiring", pd.DataFrame()),
            nh_termination_events=final_consolidated.get("nh_termination", pd.DataFrame()),
            compensation_events=final_consolidated.get("compensation", pd.DataFrame()),
            contribution_events=final_consolidated.get("contribution", pd.DataFrame()),
            year=self.year,
            logger=self.logger,
        )

    def clear(self) -> None:
        """Clear all accumulated events."""
        self.accumulated_events.clear()
        self.logger.info("Cleared accumulated events")

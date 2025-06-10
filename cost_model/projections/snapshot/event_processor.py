"""
Event processing functionality for snapshot updates.

Handles the processing of events (hiring, termination, promotion, etc.)
to update snapshots and extract employee information.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import (
    COMPENSATION_EXTRACTION_PRIORITIES,
    DEFAULT_COMPENSATION,
    LEVEL_BASED_DEFAULTS,
)
from .exceptions import CompensationExtractionError, EventProcessingError
from .models import CompensationExtractionResult

logger = logging.getLogger(__name__)


class EventProcessor:
    """Handles processing of events for snapshot updates."""

    def __init__(self):
        self.compensation_priorities = COMPENSATION_EXTRACTION_PRIORITIES

    def update_snapshot_with_events(
        self, prev_snapshot: pd.DataFrame, events: pd.DataFrame, target_date: datetime
    ) -> pd.DataFrame:
        """
        Apply events to previous snapshot up to a given date.

        This is a wrapper around the existing snapshot update functionality
        for backward compatibility.

        Args:
            prev_snapshot: Previous snapshot DataFrame
            events: Events to apply
            target_date: Apply events up to this date

        Returns:
            Updated snapshot DataFrame
        """
        logger.info(f"Updating snapshot with events up to {target_date}")

        try:
            # Import and use existing functionality
            from cost_model.state.snapshot_update import (
                update_snapshot_with_events as original_func,
            )

            return original_func(prev_snapshot, events, target_date)

        except Exception as e:
            logger.error(f"Failed to update snapshot with events: {str(e)}")
            raise EventProcessingError(f"Snapshot update failed: {str(e)}") from e

    def process_hire_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Process hiring events for snapshot reconstruction.

        Args:
            events: DataFrame containing hiring events

        Returns:
            Processed hiring events
        """
        # Filter for hiring events
        hire_events = events[events["event_type"] == "hire"].copy()

        if hire_events.empty:
            logger.debug("No hiring events found")
            return hire_events

        logger.debug(f"Processing {len(hire_events)} hiring events")

        # Add any additional processing logic here
        # For now, return as-is
        return hire_events

    def process_termination_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Process termination events for snapshot reconstruction.

        Args:
            events: DataFrame containing termination events

        Returns:
            Processed termination events
        """
        # Filter for termination events
        term_events = events[events["event_type"] == "termination"].copy()

        if term_events.empty:
            logger.debug("No termination events found")
            return term_events

        logger.debug(f"Processing {len(term_events)} termination events")

        # Add any additional processing logic here
        # For now, return as-is
        return term_events

    def extract_compensation_for_employee(
        self, emp_id: str, events: pd.DataFrame, default_compensation: Optional[float] = None
    ) -> CompensationExtractionResult:
        """
        Extract compensation for a specific employee using priority-based approach.

        Args:
            emp_id: Employee ID
            events: Events DataFrame
            default_compensation: Default compensation if none found

        Returns:
            CompensationExtractionResult with extracted compensation
        """
        logger.debug(f"Extracting compensation for employee {emp_id}")

        if default_compensation is None:
            default_compensation = DEFAULT_COMPENSATION

        # Filter events for this employee
        emp_events = events[events["EMP_ID"] == emp_id].copy()

        if emp_events.empty:
            logger.debug(f"No events found for employee {emp_id}")
            return CompensationExtractionResult(
                compensation=default_compensation,
                source="global_default",
                confidence="low",
                details={"reason": "no_events_found"},
            )

        # Try each priority source
        for priority in self.compensation_priorities:
            result = self._extract_by_priority(emp_id, emp_events, priority)
            if result.compensation is not None:
                return result

        # If nothing found, use default
        logger.warning(f"No compensation found for employee {emp_id}, using default")
        return CompensationExtractionResult(
            compensation=default_compensation,
            source="global_default",
            confidence="low",
            details={"reason": "no_compensation_in_events"},
        )

    def extract_employee_from_events(self, emp_id: str, events: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract complete employee information from events.

        Args:
            emp_id: Employee ID
            events: Events DataFrame

        Returns:
            Dictionary with extracted employee data
        """
        logger.debug(f"Extracting employee data from events for {emp_id}")

        emp_events = events[events["EMP_ID"] == emp_id].copy()

        if emp_events.empty:
            raise EventProcessingError(f"No events found for employee {emp_id}")

        # Sort events by date to get most recent information
        emp_events = emp_events.sort_values("event_date")

        # Extract basic information
        employee_data = {
            "EMP_ID": emp_id,
            "EMP_HIRE_DATE": None,
            "EMP_TERM_DATE": None,
            "EMP_GROSS_COMP": None,
            "EMP_LEVEL": None,
            "EMP_ACTIVE": True,
            "EMP_EXITED": False,
        }

        # Process events to extract information
        for _, event in emp_events.iterrows():
            event_type = event.get("event_type", "")

            if event_type == "hire":
                employee_data["EMP_HIRE_DATE"] = event.get("event_date")
                if "compensation" in event:
                    employee_data["EMP_GROSS_COMP"] = event["compensation"]
                if "job_level" in event:
                    employee_data["EMP_LEVEL"] = event["job_level"]

            elif event_type == "termination":
                employee_data["EMP_TERM_DATE"] = event.get("event_date")
                employee_data["EMP_ACTIVE"] = False
                employee_data["EMP_EXITED"] = True

            elif event_type == "promotion":
                if "job_level" in event:
                    employee_data["EMP_LEVEL"] = event["job_level"]
                if "compensation" in event:
                    employee_data["EMP_GROSS_COMP"] = event["compensation"]

            elif event_type == "compensation":
                if "compensation" in event:
                    employee_data["EMP_GROSS_COMP"] = event["compensation"]

        # Fill in defaults for missing values
        if employee_data["EMP_GROSS_COMP"] is None:
            comp_result = self.extract_compensation_for_employee(emp_id, events)
            employee_data["EMP_GROSS_COMP"] = comp_result.compensation

        return employee_data

    def _extract_by_priority(
        self, emp_id: str, emp_events: pd.DataFrame, priority: str
    ) -> CompensationExtractionResult:
        """Extract compensation using specific priority method."""

        if priority == "hire_events":
            return self._extract_from_hire_events(emp_id, emp_events)
        elif priority == "promotion_events":
            return self._extract_from_promotion_events(emp_id, emp_events)
        elif priority == "compensation_events":
            return self._extract_from_compensation_events(emp_id, emp_events)
        elif priority == "default_by_level":
            return self._extract_by_level_default(emp_id, emp_events)
        elif priority == "global_default":
            return CompensationExtractionResult(
                compensation=DEFAULT_COMPENSATION,
                source="global_default",
                confidence="low",
                details={"priority": priority},
            )
        else:
            return CompensationExtractionResult(
                compensation=None,
                source="unknown",
                confidence="low",
                details={"error": f"unknown_priority_{priority}"},
            )

    def _extract_from_hire_events(
        self, emp_id: str, emp_events: pd.DataFrame
    ) -> CompensationExtractionResult:
        """Extract compensation from hiring events."""
        hire_events = emp_events[emp_events["event_type"] == "hire"]

        if hire_events.empty:
            return CompensationExtractionResult(
                compensation=None, source="hire_events", confidence="low", details={}
            )

        # Get most recent hire event with compensation
        for _, event in hire_events.sort_values("event_date", ascending=False).iterrows():
            if "compensation" in event and pd.notna(event["compensation"]):
                return CompensationExtractionResult(
                    compensation=float(event["compensation"]),
                    source="hire_events",
                    confidence="high",
                    details={"event_date": event["event_date"]},
                )

        return CompensationExtractionResult(
            compensation=None, source="hire_events", confidence="low", details={}
        )

    def _extract_from_promotion_events(
        self, emp_id: str, emp_events: pd.DataFrame
    ) -> CompensationExtractionResult:
        """Extract compensation from promotion events."""
        promotion_events = emp_events[emp_events["event_type"] == "promotion"]

        if promotion_events.empty:
            return CompensationExtractionResult(
                compensation=None, source="promotion_events", confidence="low", details={}
            )

        # Get most recent promotion event with compensation
        for _, event in promotion_events.sort_values("event_date", ascending=False).iterrows():
            if "compensation" in event and pd.notna(event["compensation"]):
                return CompensationExtractionResult(
                    compensation=float(event["compensation"]),
                    source="promotion_events",
                    confidence="high",
                    details={"event_date": event["event_date"]},
                )

        return CompensationExtractionResult(
            compensation=None, source="promotion_events", confidence="low", details={}
        )

    def _extract_from_compensation_events(
        self, emp_id: str, emp_events: pd.DataFrame
    ) -> CompensationExtractionResult:
        """Extract compensation from compensation change events."""
        comp_events = emp_events[emp_events["event_type"] == "compensation"]

        if comp_events.empty:
            return CompensationExtractionResult(
                compensation=None, source="compensation_events", confidence="low", details={}
            )

        # Get most recent compensation event
        for _, event in comp_events.sort_values("event_date", ascending=False).iterrows():
            if "compensation" in event and pd.notna(event["compensation"]):
                return CompensationExtractionResult(
                    compensation=float(event["compensation"]),
                    source="compensation_events",
                    confidence="high",
                    details={"event_date": event["event_date"]},
                )

        return CompensationExtractionResult(
            compensation=None, source="compensation_events", confidence="low", details={}
        )

    def _extract_by_level_default(
        self, emp_id: str, emp_events: pd.DataFrame
    ) -> CompensationExtractionResult:
        """Extract compensation using level-based defaults."""
        # Try to find job level from events
        for _, event in emp_events.iterrows():
            if "job_level" in event and pd.notna(event["job_level"]):
                level = event["job_level"]
                if level in LEVEL_BASED_DEFAULTS:
                    return CompensationExtractionResult(
                        compensation=LEVEL_BASED_DEFAULTS[level],
                        source="default_by_level",
                        confidence="medium",
                        details={"job_level": level},
                    )

        return CompensationExtractionResult(
            compensation=None, source="default_by_level", confidence="low", details={}
        )

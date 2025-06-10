# cost_model/state/snapshot/event_processors/hire_processor.py
"""
Event processor for handling hire events in snapshot updates.
"""
from typing import List

import pandas as pd

from cost_model.state.schema import (
    EMP_ACTIVE,
    EMP_BIRTH_DATE,
    EMP_EXITED,
    EMP_GROSS_COMP,
    EMP_HIRE_DATE,
    EMP_ID,
    EVT_HIRE,
    SIMULATION_YEAR,
)

from .base import BaseEventProcessor, EventProcessorResult


class HireEventProcessor(BaseEventProcessor):
    """Processes hire events to add new employees to the snapshot."""

    def get_event_types(self) -> List[str]:
        """Return event types handled by this processor."""
        return [EVT_HIRE]

    def get_required_columns(self) -> List[str]:
        """Return required columns for hire events."""
        return [EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP]

    def process_events(
        self, snapshot: pd.DataFrame, events: pd.DataFrame, snapshot_year: int
    ) -> EventProcessorResult:
        """
        Process hire events to add new employees.

        Args:
            snapshot: Current snapshot DataFrame
            events: Hire events DataFrame
            snapshot_year: Current simulation year

        Returns:
            EventProcessorResult with updated snapshot
        """
        result = EventProcessorResult()

        # Filter to hire events
        hire_events = self.filter_events(events)

        if hire_events.empty:
            result.updated_snapshot = snapshot.copy()
            return result

        self.log_processing_start(len(hire_events), len(snapshot))

        # Validate events
        validation_errors = self.validate_events(hire_events)
        if validation_errors:
            for error in validation_errors:
                result.add_error(error)
            result.updated_snapshot = snapshot.copy()
            return result

        try:
            # Start with copy of current snapshot
            updated_snapshot = snapshot.copy()

            # Process each hire event
            for _, hire_event in hire_events.iterrows():
                emp_id = hire_event[EMP_ID]

                # Check if employee already exists
                if not updated_snapshot.empty and EMP_ID in updated_snapshot.columns:
                    existing_employee = updated_snapshot[updated_snapshot[EMP_ID] == emp_id]
                    if not existing_employee.empty:
                        result.add_warning(f"Employee {emp_id} already exists, skipping hire")
                        continue

                # Create new employee record
                new_employee = self._create_employee_record(hire_event, snapshot_year)

                # Add to snapshot
                if updated_snapshot.empty:
                    updated_snapshot = pd.DataFrame([new_employee])
                else:
                    # Use pd.concat instead of append (deprecated)
                    new_row_df = pd.DataFrame([new_employee])
                    updated_snapshot = pd.concat([updated_snapshot, new_row_df], ignore_index=True)

                result.add_affected_employee(emp_id)
                self.logger.debug(f"Added new employee {emp_id}")

            result.updated_snapshot = updated_snapshot
            result.metadata["employees_hired"] = len(result.employees_affected)

            # Check integrity
            integrity_issues = self.ensure_snapshot_integrity(
                snapshot, updated_snapshot, hire_events
            )
            for issue in integrity_issues:
                result.add_warning(f"Integrity issue: {issue}")

        except Exception as e:
            result.add_error(f"Error processing hire events: {str(e)}")
            result.updated_snapshot = snapshot.copy()
            self.logger.error(f"Error in hire processing: {e}", exc_info=True)

        self.log_processing_end(result)
        return result

    def _create_employee_record(self, hire_event: pd.Series, snapshot_year: int) -> dict:
        """
        Create a new employee record from hire event.

        Args:
            hire_event: Hire event Series
            snapshot_year: Current simulation year

        Returns:
            Dictionary with new employee data
        """
        # Create basic employee record
        employee_record = {
            EMP_ID: hire_event[EMP_ID],
            EMP_HIRE_DATE: hire_event[EMP_HIRE_DATE],
            EMP_BIRTH_DATE: hire_event[EMP_BIRTH_DATE],
            EMP_GROSS_COMP: hire_event[EMP_GROSS_COMP],
            EMP_ACTIVE: True,
            EMP_EXITED: False,
            SIMULATION_YEAR: snapshot_year,
        }

        # Add optional fields if present in event
        optional_fields = [
            "employee_deferral_rate",
            "employee_level",
            "employee_level_source",
            "employee_tenure_band",
            "employee_tenure",
        ]

        for field in optional_fields:
            if field in hire_event and pd.notna(hire_event[field]):
                employee_record[field] = hire_event[field]

        return employee_record

    def validate_events(self, events: pd.DataFrame) -> List[str]:
        """
        Validate hire events.

        Args:
            events: Hire events DataFrame

        Returns:
            List of validation errors
        """
        errors = super().validate_events(events)

        if events.empty:
            return errors

        # Check for required hire-specific columns
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in events.columns]
        if missing_columns:
            errors.append(f"Missing required hire columns: {missing_columns}")

        # Validate compensation values
        if EMP_GROSS_COMP in events.columns:
            invalid_comp = events[EMP_GROSS_COMP].isna() | (events[EMP_GROSS_COMP] <= 0)
            if invalid_comp.any():
                errors.append(f"Found {invalid_comp.sum()} hire events with invalid compensation")

        # Validate dates
        date_columns = [EMP_HIRE_DATE, EMP_BIRTH_DATE]
        for col in date_columns:
            if col in events.columns:
                invalid_dates = events[col].isna()
                if invalid_dates.any():
                    errors.append(f"Found {invalid_dates.sum()} hire events with invalid {col}")

        return errors

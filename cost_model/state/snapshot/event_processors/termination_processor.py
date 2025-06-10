# cost_model/state/snapshot/event_processors/termination_processor.py
"""
Event processor for handling termination events in snapshot updates.
"""
from typing import List

import pandas as pd

from cost_model.state.schema import (
    EMP_ACTIVE,
    EMP_EXITED,
    EMP_ID,
    EMP_TERM_DATE,
    EVT_NEW_HIRE_TERM,
    EVT_TERM,
)

from .base import BaseEventProcessor, EventProcessorResult


class TerminationEventProcessor(BaseEventProcessor):
    """Processes termination events to mark employees as terminated."""

    def get_event_types(self) -> List[str]:
        """Return event types handled by this processor."""
        return [EVT_TERM, EVT_NEW_HIRE_TERM]

    def get_required_columns(self) -> List[str]:
        """Return required columns for termination events."""
        return [EMP_ID, EMP_TERM_DATE]

    def process_events(
        self, snapshot: pd.DataFrame, events: pd.DataFrame, snapshot_year: int
    ) -> EventProcessorResult:
        """
        Process termination events to mark employees as terminated.

        Args:
            snapshot: Current snapshot DataFrame
            events: Termination events DataFrame
            snapshot_year: Current simulation year

        Returns:
            EventProcessorResult with updated snapshot
        """
        result = EventProcessorResult()

        # Filter to termination events
        term_events = self.filter_events(events)

        if term_events.empty:
            result.updated_snapshot = snapshot.copy()
            return result

        self.log_processing_start(len(term_events), len(snapshot))

        # Validate events
        validation_errors = self.validate_events(term_events)
        if validation_errors:
            for error in validation_errors:
                result.add_error(error)
            result.updated_snapshot = snapshot.copy()
            return result

        try:
            # Start with copy of current snapshot
            updated_snapshot = snapshot.copy()

            # Process each termination event
            for _, term_event in term_events.iterrows():
                emp_id = term_event[EMP_ID]
                term_date = term_event[EMP_TERM_DATE]

                # Find employee in snapshot
                if EMP_ID in updated_snapshot.columns:
                    employee_mask = updated_snapshot[EMP_ID] == emp_id

                    if employee_mask.any():
                        # Update termination status
                        updated_snapshot.loc[employee_mask, EMP_TERM_DATE] = term_date
                        updated_snapshot.loc[employee_mask, EMP_ACTIVE] = False
                        updated_snapshot.loc[employee_mask, EMP_EXITED] = True

                        result.add_affected_employee(emp_id)
                        self.logger.debug(f"Terminated employee {emp_id}")
                    else:
                        result.add_warning(f"Employee {emp_id} not found for termination")
                else:
                    result.add_error("No employee ID column in snapshot")
                    break

            result.updated_snapshot = updated_snapshot
            result.metadata["employees_terminated"] = len(result.employees_affected)

            # Check integrity
            integrity_issues = self.ensure_snapshot_integrity(
                snapshot, updated_snapshot, term_events
            )
            for issue in integrity_issues:
                result.add_warning(f"Integrity issue: {issue}")

        except Exception as e:
            result.add_error(f"Error processing termination events: {str(e)}")
            result.updated_snapshot = snapshot.copy()
            self.logger.error(f"Error in termination processing: {e}", exc_info=True)

        self.log_processing_end(result)
        return result

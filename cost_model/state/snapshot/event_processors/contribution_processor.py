# cost_model/state/snapshot/event_processors/contribution_processor.py
"""
Event processor for handling contribution events in snapshot updates.
"""
from typing import List

import pandas as pd

from cost_model.state.schema import EMP_CONTR, EMP_ID, EMPLOYER_CORE, EMPLOYER_MATCH, EVT_CONTRIB

from .base import BaseEventProcessor, EventProcessorResult


class ContributionEventProcessor(BaseEventProcessor):
    """Processes contribution events to update employee contribution data."""

    def get_event_types(self) -> List[str]:
        """Return event types handled by this processor."""
        return [EVT_CONTRIB]

    def process_events(
        self, snapshot: pd.DataFrame, events: pd.DataFrame, snapshot_year: int
    ) -> EventProcessorResult:
        """
        Process contribution events to update employee contribution data.

        Args:
            snapshot: Current snapshot DataFrame
            events: Contribution events DataFrame
            snapshot_year: Current simulation year

        Returns:
            EventProcessorResult with updated snapshot
        """
        result = EventProcessorResult()

        # Filter to contribution events
        contrib_events = self.filter_events(events)

        if contrib_events.empty:
            result.updated_snapshot = snapshot.copy()
            return result

        self.log_processing_start(len(contrib_events), len(snapshot))

        try:
            # Start with copy of current snapshot
            updated_snapshot = snapshot.copy()

            # Process each contribution event
            for _, contrib_event in contrib_events.iterrows():
                emp_id = contrib_event[EMP_ID]

                # Extract contribution values from event
                contribution_data = self._extract_contribution_values(contrib_event)

                if not contribution_data:
                    result.add_warning(
                        f"Could not extract contribution values for employee {emp_id}"
                    )
                    continue

                # Find employee in snapshot
                if EMP_ID in updated_snapshot.columns:
                    employee_mask = updated_snapshot[EMP_ID] == emp_id

                    if employee_mask.any():
                        # Update contribution columns
                        for col, value in contribution_data.items():
                            if col in updated_snapshot.columns:
                                updated_snapshot.loc[employee_mask, col] = value

                        result.add_affected_employee(emp_id)
                        self.logger.debug(f"Updated contributions for employee {emp_id}")
                    else:
                        result.add_warning(f"Employee {emp_id} not found for contribution update")
                else:
                    result.add_error("No employee ID column in snapshot")
                    break

            result.updated_snapshot = updated_snapshot
            result.metadata["contribution_updates"] = len(result.employees_affected)

        except Exception as e:
            result.add_error(f"Error processing contribution events: {str(e)}")
            result.updated_snapshot = snapshot.copy()
            self.logger.error(f"Error in contribution processing: {e}", exc_info=True)

        self.log_processing_end(result)
        return result

    def _extract_contribution_values(self, event: pd.Series) -> dict:
        """
        Extract contribution values from event.

        Args:
            event: Event Series

        Returns:
            Dictionary of contribution column names to values
        """
        contribution_data = {}

        # Direct column mapping
        column_mappings = {
            "employee_contribution": EMP_CONTR,
            "employer_core": EMPLOYER_CORE,
            "employer_match": EMPLOYER_MATCH,
            EMP_CONTR: EMP_CONTR,
            EMPLOYER_CORE: EMPLOYER_CORE,
            EMPLOYER_MATCH: EMPLOYER_MATCH,
        }

        for event_col, snapshot_col in column_mappings.items():
            if event_col in event and pd.notna(event[event_col]):
                try:
                    contribution_data[snapshot_col] = float(event[event_col])
                except (ValueError, TypeError):
                    continue

        # Try to parse from value_json if present
        if "value_json" in event and pd.notna(event["value_json"]):
            try:
                import json

                value_data = json.loads(event["value_json"])

                for json_key, snapshot_col in column_mappings.items():
                    if json_key in value_data:
                        try:
                            contribution_data[snapshot_col] = float(value_data[json_key])
                        except (ValueError, TypeError):
                            continue

            except (json.JSONDecodeError, KeyError):
                pass

        return contribution_data

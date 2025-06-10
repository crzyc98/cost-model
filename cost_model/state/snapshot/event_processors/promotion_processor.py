# cost_model/state/snapshot/event_processors/promotion_processor.py
"""
Event processor for handling promotion events in snapshot updates.
"""
from typing import List

import pandas as pd

from cost_model.state.schema import EMP_ID, EMP_LEVEL, EMP_LEVEL_SOURCE, EVT_PROMOTION

from .base import BaseEventProcessor, EventProcessorResult


class PromotionEventProcessor(BaseEventProcessor):
    """Processes promotion events to update employee levels."""

    def get_event_types(self) -> List[str]:
        """Return event types handled by this processor."""
        return [EVT_PROMOTION]

    def process_events(
        self, snapshot: pd.DataFrame, events: pd.DataFrame, snapshot_year: int
    ) -> EventProcessorResult:
        """
        Process promotion events to update employee levels.

        Args:
            snapshot: Current snapshot DataFrame
            events: Promotion events DataFrame
            snapshot_year: Current simulation year

        Returns:
            EventProcessorResult with updated snapshot
        """
        result = EventProcessorResult()

        # Filter to promotion events
        promo_events = self.filter_events(events)

        if promo_events.empty:
            result.updated_snapshot = snapshot.copy()
            return result

        self.log_processing_start(len(promo_events), len(snapshot))

        try:
            # Start with copy of current snapshot
            updated_snapshot = snapshot.copy()

            # Process each promotion event
            for _, promo_event in promo_events.iterrows():
                emp_id = promo_event[EMP_ID]

                # Extract new level from event
                new_level = self._extract_level_value(promo_event)

                if new_level is None:
                    result.add_warning(f"Could not extract level value for employee {emp_id}")
                    continue

                # Find employee in snapshot
                if EMP_ID in updated_snapshot.columns:
                    employee_mask = updated_snapshot[EMP_ID] == emp_id

                    if employee_mask.any():
                        # Update level
                        if EMP_LEVEL in updated_snapshot.columns:
                            updated_snapshot.loc[employee_mask, EMP_LEVEL] = new_level

                        # Update level source if available
                        if EMP_LEVEL_SOURCE in updated_snapshot.columns:
                            updated_snapshot.loc[employee_mask, EMP_LEVEL_SOURCE] = (
                                "promotion_event"
                            )

                        result.add_affected_employee(emp_id)
                        self.logger.debug(f"Promoted employee {emp_id} to level {new_level}")
                    else:
                        result.add_warning(f"Employee {emp_id} not found for promotion")
                else:
                    result.add_error("No employee ID column in snapshot")
                    break

            result.updated_snapshot = updated_snapshot
            result.metadata["promotions"] = len(result.employees_affected)

        except Exception as e:
            result.add_error(f"Error processing promotion events: {str(e)}")
            result.updated_snapshot = snapshot.copy()
            self.logger.error(f"Error in promotion processing: {e}", exc_info=True)

        self.log_processing_end(result)
        return result

    def _extract_level_value(self, event: pd.Series):
        """
        Extract level value from event.

        Args:
            event: Event Series

        Returns:
            Level value or None if not found
        """
        # Try different possible column names for level
        level_columns = ["new_level", "level", EMP_LEVEL, "job_level"]

        for col in level_columns:
            if col in event and pd.notna(event[col]):
                try:
                    return int(event[col])
                except (ValueError, TypeError):
                    continue

        # Try to parse from value_json if present
        if "value_json" in event and pd.notna(event["value_json"]):
            try:
                import json

                value_data = json.loads(event["value_json"])
                if "new_level" in value_data:
                    return int(value_data["new_level"])
                elif "level" in value_data:
                    return int(value_data["level"])
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass

        return None

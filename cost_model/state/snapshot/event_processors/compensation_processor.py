# cost_model/state/snapshot/event_processors/compensation_processor.py
"""
Event processor for handling compensation events in snapshot updates.
"""
import pandas as pd
from typing import List

from .base import BaseEventProcessor, EventProcessorResult
from cost_model.state.schema import EVT_COMP, EVT_COLA, EVT_RAISE, EMP_ID, EMP_GROSS_COMP


class CompensationEventProcessor(BaseEventProcessor):
    """Processes compensation-related events to update employee compensation."""
    
    def get_event_types(self) -> List[str]:
        """Return event types handled by this processor."""
        return [EVT_COMP, EVT_COLA, EVT_RAISE]
    
    def get_required_columns(self) -> List[str]:
        """Return required columns for compensation events."""
        return [EMP_ID]  # Compensation amount might be in different columns
    
    def process_events(
        self,
        snapshot: pd.DataFrame,
        events: pd.DataFrame,
        snapshot_year: int
    ) -> EventProcessorResult:
        """
        Process compensation events to update employee compensation.
        
        Args:
            snapshot: Current snapshot DataFrame
            events: Compensation events DataFrame
            snapshot_year: Current simulation year
        
        Returns:
            EventProcessorResult with updated snapshot
        """
        result = EventProcessorResult()
        
        # Filter to compensation events
        comp_events = self.filter_events(events)
        
        if comp_events.empty:
            result.updated_snapshot = snapshot.copy()
            return result
        
        self.log_processing_start(len(comp_events), len(snapshot))
        
        try:
            # Start with copy of current snapshot
            updated_snapshot = snapshot.copy()
            
            # Process each compensation event
            for _, comp_event in comp_events.iterrows():
                emp_id = comp_event[EMP_ID]
                
                # Extract compensation value from event
                new_compensation = self._extract_compensation_value(comp_event)
                
                if new_compensation is None:
                    result.add_warning(f"Could not extract compensation value for employee {emp_id}")
                    continue
                
                # Find employee in snapshot
                if EMP_ID in updated_snapshot.columns:
                    employee_mask = updated_snapshot[EMP_ID] == emp_id
                    
                    if employee_mask.any():
                        # Update compensation
                        updated_snapshot.loc[employee_mask, EMP_GROSS_COMP] = new_compensation
                        
                        result.add_affected_employee(emp_id)
                        self.logger.debug(f"Updated compensation for employee {emp_id}: {new_compensation}")
                    else:
                        result.add_warning(f"Employee {emp_id} not found for compensation update")
                else:
                    result.add_error("No employee ID column in snapshot")
                    break
            
            result.updated_snapshot = updated_snapshot
            result.metadata['compensation_updates'] = len(result.employees_affected)
            
        except Exception as e:
            result.add_error(f"Error processing compensation events: {str(e)}")
            result.updated_snapshot = snapshot.copy()
            self.logger.error(f"Error in compensation processing: {e}", exc_info=True)
        
        self.log_processing_end(result)
        return result
    
    def _extract_compensation_value(self, event: pd.Series) -> float:
        """
        Extract compensation value from event.
        
        Args:
            event: Event Series
        
        Returns:
            Compensation value or None if not found
        """
        # Try different possible column names for compensation
        comp_columns = ['new_compensation', 'compensation', EMP_GROSS_COMP, 'amount', 'value']
        
        for col in comp_columns:
            if col in event and pd.notna(event[col]):
                try:
                    return float(event[col])
                except (ValueError, TypeError):
                    continue
        
        # Try to parse from value_json if present
        if 'value_json' in event and pd.notna(event['value_json']):
            try:
                import json
                value_data = json.loads(event['value_json'])
                if 'new_compensation' in value_data:
                    return float(value_data['new_compensation'])
                elif 'compensation' in value_data:
                    return float(value_data['compensation'])
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass
        
        return None
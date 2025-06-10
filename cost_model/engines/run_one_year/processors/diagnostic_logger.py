# cost_model/engines/run_one_year/processors/diagnostic_logger.py
"""
Diagnostic logging processor for detailed simulation monitoring.
"""
import pandas as pd
import logging
from typing import Any, Dict, List, Optional

from .base import BaseProcessor, ProcessorResult
from cost_model.state.schema import EMP_ID, SIMULATION_YEAR, EMP_ACTIVE


class DiagnosticLogger(BaseProcessor):
    """Handles detailed diagnostic logging and monitoring during simulation."""
    
    def __init__(self, logger_name: Optional[str] = None):
        super().__init__(logger_name)
        self.diagnostic_data = {}
    
    def log_step_diagnostics(
        self,
        step_name: str,
        snapshot_before: pd.DataFrame,
        snapshot_after: pd.DataFrame,
        events: pd.DataFrame,
        year: int,
        **additional_context
    ) -> ProcessorResult:
        """
        Log detailed diagnostics for a simulation step.
        
        Args:
            step_name: Name of the simulation step
            snapshot_before: Snapshot before the step
            snapshot_after: Snapshot after the step
            events: Events generated during the step
            year: Simulation year
            additional_context: Additional context information
            
        Returns:
            ProcessorResult with diagnostic information
        """
        result = ProcessorResult()
        
        try:
            self.logger.info(f"[DIAGNOSTIC] {step_name} - Year {year}")
            
            # Basic counts
            before_count = len(snapshot_before) if not snapshot_before.empty else 0
            after_count = len(snapshot_after) if not snapshot_after.empty else 0
            events_count = len(events) if not events.empty else 0
            
            self.logger.info(f"  Snapshot before: {before_count} employees")
            self.logger.info(f"  Snapshot after: {after_count} employees")
            self.logger.info(f"  Events generated: {events_count}")
            
            # Active employee counts
            if EMP_ACTIVE in snapshot_before.columns:
                active_before = snapshot_before[EMP_ACTIVE].sum()
                self.logger.info(f"  Active employees before: {active_before}")
            
            if EMP_ACTIVE in snapshot_after.columns:
                active_after = snapshot_after[EMP_ACTIVE].sum()
                self.logger.info(f"  Active employees after: {active_after}")
            
            # Event type breakdown
            if not events.empty and 'event_type' in events.columns:
                event_type_counts = events['event_type'].value_counts()
                self.logger.info(f"  Event breakdown: {event_type_counts.to_dict()}")
            
            # Employee ID consistency checks
            self._check_employee_consistency(snapshot_before, snapshot_after, events, step_name)
            
            # Store diagnostic data
            self.diagnostic_data[f"{step_name}_{year}"] = {
                'step_name': step_name,
                'year': year,
                'employees_before': before_count,
                'employees_after': after_count,
                'events_count': events_count,
                'active_before': snapshot_before[EMP_ACTIVE].sum() if EMP_ACTIVE in snapshot_before.columns else None,
                'active_after': snapshot_after[EMP_ACTIVE].sum() if EMP_ACTIVE in snapshot_after.columns else None,
                'additional_context': additional_context
            }
            
            result.add_metadata('diagnostic_key', f"{step_name}_{year}")
            result.add_metadata('diagnostic_summary', self.diagnostic_data[f"{step_name}_{year}"])
            
        except Exception as e:
            self.logger.error(f"Error in diagnostic logging for {step_name}: {e}", exc_info=True)
            result.add_error(f"Diagnostic logging failed: {str(e)}")
        
        return result
    
    def _check_employee_consistency(
        self,
        snapshot_before: pd.DataFrame,
        snapshot_after: pd.DataFrame,
        events: pd.DataFrame,
        step_name: str
    ):
        """Check for employee ID consistency across snapshots and events."""
        try:
            if snapshot_before.empty or snapshot_after.empty:
                return
            
            if EMP_ID not in snapshot_before.columns or EMP_ID not in snapshot_after.columns:
                return
            
            # Check for employees that disappeared without events
            employees_before = set(snapshot_before[EMP_ID].unique())
            employees_after = set(snapshot_after[EMP_ID].unique())
            
            disappeared = employees_before - employees_after
            appeared = employees_after - employees_before
            
            if disappeared:
                # Check if they have corresponding termination events
                if not events.empty and EMP_ID in events.columns:
                    event_employees = set(events[EMP_ID].unique())
                    unaccounted_disappearances = disappeared - event_employees
                    if unaccounted_disappearances:
                        self.logger.warning(f"  {step_name}: {len(unaccounted_disappearances)} employees disappeared without events")
                else:
                    self.logger.warning(f"  {step_name}: {len(disappeared)} employees disappeared")
            
            if appeared:
                # Check if they have corresponding hire events
                if not events.empty and EMP_ID in events.columns:
                    event_employees = set(events[EMP_ID].unique())
                    unaccounted_appearances = appeared - event_employees
                    if unaccounted_appearances:
                        self.logger.warning(f"  {step_name}: {len(unaccounted_appearances)} employees appeared without events")
                else:
                    self.logger.warning(f"  {step_name}: {len(appeared)} employees appeared")
        
        except Exception as e:
            self.logger.warning(f"Could not check employee consistency: {e}")
    
    def log_year_summary(self, year: int, final_snapshot: pd.DataFrame, 
                        total_events: pd.DataFrame) -> ProcessorResult:
        """
        Log summary statistics for the entire simulation year.
        
        Args:
            year: Simulation year
            final_snapshot: Final snapshot after all processing
            total_events: All events generated during the year
            
        Returns:
            ProcessorResult with year summary
        """
        result = ProcessorResult()
        
        try:
            self.logger.info(f"[YEAR SUMMARY] Simulation Year {year}")
            
            # Final headcount
            final_headcount = len(final_snapshot) if not final_snapshot.empty else 0
            self.logger.info(f"  Final headcount: {final_headcount}")
            
            if EMP_ACTIVE in final_snapshot.columns:
                active_employees = final_snapshot[EMP_ACTIVE].sum()
                inactive_employees = final_headcount - active_employees
                self.logger.info(f"  Active employees: {active_employees}")
                self.logger.info(f"  Inactive employees: {inactive_employees}")
            
            # Event summary
            total_events_count = len(total_events) if not total_events.empty else 0
            self.logger.info(f"  Total events: {total_events_count}")
            
            if not total_events.empty and 'event_type' in total_events.columns:
                event_summary = total_events['event_type'].value_counts()
                self.logger.info(f"  Event breakdown: {event_summary.to_dict()}")
            
            # Store year summary
            summary_key = f"year_summary_{year}"
            self.diagnostic_data[summary_key] = {
                'year': year,
                'final_headcount': final_headcount,
                'active_employees': final_snapshot[EMP_ACTIVE].sum() if EMP_ACTIVE in final_snapshot.columns else None,
                'total_events': total_events_count,
                'event_breakdown': total_events['event_type'].value_counts().to_dict() if not total_events.empty and 'event_type' in total_events.columns else {}
            }
            
            result.add_metadata('summary_key', summary_key)
            result.add_metadata('year_summary', self.diagnostic_data[summary_key])
            
        except Exception as e:
            self.logger.error(f"Error in year summary logging: {e}", exc_info=True)
            result.add_error(f"Year summary logging failed: {str(e)}")
        
        return result
    
    def get_diagnostic_data(self) -> Dict[str, Any]:
        """Get all collected diagnostic data."""
        return self.diagnostic_data.copy()
    
    def clear_diagnostic_data(self):
        """Clear all collected diagnostic data."""
        self.diagnostic_data.clear()
# cost_model/engines/run_one_year/processors/termination_processor.py
"""
Processor for handling employee terminations during yearly simulation.
"""
import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple, Optional

from .base import BaseProcessor, ProcessorResult
from cost_model.engines import term
from cost_model.engines.nh_termination import run_new_hires
from cost_model.state.schema import EMP_ID, SIMULATION_YEAR, EMP_ACTIVE


class TerminationProcessor(BaseProcessor):
    """Handles terminations for both experienced employees and new hires."""
    
    def process_experienced_terminations(
        self,
        snapshot: pd.DataFrame,
        year: int,
        global_params: Any,
        hazard_table: pd.DataFrame,
        rng: Any,
        rng_seed_offset: int = 0,
        deterministic_term: bool = False,
        **kwargs
    ) -> ProcessorResult:
        """
        Process terminations for experienced employees.
        
        Args:
            snapshot: Current employee snapshot
            year: Simulation year
            global_params: Global simulation parameters
            hazard_table: Hazard rate table
            rng: Random number generator
            rng_seed_offset: Seed offset for RNG
            deterministic_term: Whether to use deterministic termination logic
            
        Returns:
            ProcessorResult with termination events
        """
        self.log_step_start(
            "Hazard-based terminations (experienced only)",
            snapshot_size=len(snapshot),
            year=year,
            deterministic=deterministic_term
        )
        
        result = ProcessorResult()
        
        try:
            # Validate inputs
            if not self.validate_termination_inputs(snapshot, year, global_params, hazard_table):
                result.add_error("Invalid inputs for experienced termination processing")
                return result
            
            # Filter to experienced employees only
            experienced_employees = snapshot[
                snapshot[SIMULATION_YEAR] < year
            ].copy() if SIMULATION_YEAR in snapshot.columns else snapshot.copy()
            
            if experienced_employees.empty:
                self.logger.info("No experienced employees found for terminations")
                return result
            
            self.logger.info(f"Processing terminations for {len(experienced_employees)} experienced employees")
            
            # Apply termination logic
            termination_events = term.run(
                snapshot=experienced_employees,
                simulation_year=year,
                global_params=global_params,
                hazard_table=hazard_table,
                rng=rng,
                rng_seed_offset=rng_seed_offset,
                deterministic=deterministic_term
            )
            
            # Log termination results
            if not termination_events.empty:
                self.logger.info(f"Generated {len(termination_events)} experienced termination events")
                
                # Log additional statistics
                if EMP_ID in termination_events.columns:
                    unique_employees = termination_events[EMP_ID].nunique()
                    self.logger.info(f"  Affecting {unique_employees} unique employees")
            else:
                self.logger.info("No experienced termination events generated")
            
            result.events = termination_events
            result.add_metadata('experienced_employees_count', len(experienced_employees))
            result.add_metadata('termination_events_count', len(termination_events))
            
        except Exception as e:
            self.logger.error(f"Error during experienced termination processing: {e}", exc_info=True)
            result.add_error(f"Experienced termination processing failed: {str(e)}")
        
        self.log_step_end(
            "Hazard-based terminations (experienced only)",
            events_generated=len(result.events),
            success=result.success
        )
        
        return result
    
    def process_new_hire_terminations(
        self,
        snapshot: pd.DataFrame,
        year: int,
        global_params: Any,
        rng: Any,
        rng_seed_offset: int = 0,
        **kwargs
    ) -> ProcessorResult:
        """
        Process deterministic terminations for new hires.
        
        Args:
            snapshot: Current employee snapshot (after hires)
            year: Simulation year
            global_params: Global simulation parameters
            rng: Random number generator
            rng_seed_offset: Seed offset for RNG
            
        Returns:
            ProcessorResult with new hire termination events
        """
        self.log_step_start(
            "Deterministic new-hire terminations",
            snapshot_size=len(snapshot),
            year=year
        )
        
        result = ProcessorResult()
        
        try:
            # Validate inputs
            if not self.validate_new_hire_inputs(snapshot, year, global_params):
                result.add_error("Invalid inputs for new hire termination processing")
                return result
            
            # Filter to new hires from this year
            new_hires = snapshot[
                snapshot[SIMULATION_YEAR] == year
            ].copy() if SIMULATION_YEAR in snapshot.columns else pd.DataFrame()
            
            if new_hires.empty:
                self.logger.info("No new hires found for termination processing")
                return result
            
            self.logger.info(f"Processing terminations for {len(new_hires)} new hires")
            
            # Apply new hire termination logic
            nh_termination_events = run_new_hires(
                snapshot=new_hires,
                simulation_year=year,
                global_params=global_params,
                rng=rng,
                rng_seed_offset=rng_seed_offset
            )
            
            # Log results
            if not nh_termination_events.empty:
                self.logger.info(f"Generated {len(nh_termination_events)} new hire termination events")
                
                if EMP_ID in nh_termination_events.columns:
                    unique_employees = nh_termination_events[EMP_ID].nunique()
                    self.logger.info(f"  Affecting {unique_employees} unique new hires")
            else:
                self.logger.info("No new hire termination events generated")
            
            result.events = nh_termination_events
            result.add_metadata('new_hires_count', len(new_hires))
            result.add_metadata('nh_termination_events_count', len(nh_termination_events))
            
        except Exception as e:
            self.logger.error(f"Error during new hire termination processing: {e}", exc_info=True)
            result.add_error(f"New hire termination processing failed: {str(e)}")
        
        self.log_step_end(
            "Deterministic new-hire terminations",
            events_generated=len(result.events),
            success=result.success
        )
        
        return result
    
    def validate_termination_inputs(self, snapshot: pd.DataFrame, year: int,
                                  global_params: Any, hazard_table: pd.DataFrame) -> bool:
        """Validate inputs for experienced termination processing."""
        if snapshot.empty:
            self.logger.warning("Empty snapshot provided for termination processing")
            return True  # Empty is valid
        
        required_columns = [EMP_ID]
        missing_columns = [col for col in required_columns if col not in snapshot.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns in snapshot: {missing_columns}")
            return False
        
        if hazard_table.empty:
            self.logger.warning("Empty hazard table provided for terminations")
        
        return True
    
    def validate_new_hire_inputs(self, snapshot: pd.DataFrame, year: int, 
                               global_params: Any) -> bool:
        """Validate inputs for new hire termination processing."""
        if snapshot.empty:
            self.logger.warning("Empty snapshot provided for new hire termination processing")
            return True
        
        required_columns = [EMP_ID]
        missing_columns = [col for col in required_columns if col not in snapshot.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns in snapshot: {missing_columns}")
            return False
        
        return True
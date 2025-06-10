# cost_model/engines/run_one_year/processors/hiring_processor.py
"""
Processor for handling employee hiring during yearly simulation.
"""
import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple, Optional

from .base import BaseProcessor, ProcessorResult
from cost_model.engines import hire
from cost_model.state.schema import EMP_ID, SIMULATION_YEAR, EMP_ACTIVE
from ..utils import compute_headcount_targets


class HiringProcessor(BaseProcessor):
    """Handles employee hiring to meet headcount targets."""
    
    def process(
        self,
        snapshot: pd.DataFrame,
        year: int,
        global_params: Any,
        rng: Any,
        census_template_path: Optional[str] = None,
        rng_seed_offset: int = 0,
        **kwargs
    ) -> ProcessorResult:
        """
        Process employee hiring for the simulation year.
        
        Args:
            snapshot: Current employee snapshot (post-terminations)
            year: Simulation year
            global_params: Global simulation parameters
            rng: Random number generator
            census_template_path: Path to census template file
            rng_seed_offset: Seed offset for RNG
            
        Returns:
            ProcessorResult with hiring events and updated snapshot
        """
        self.log_step_start(
            "Generate/apply hires",
            snapshot_size=len(snapshot),
            year=year
        )
        
        result = ProcessorResult()
        
        try:
            # Validate inputs
            if not self.validate_inputs(snapshot, year, global_params):
                result.add_error("Invalid inputs for hiring processing")
                return result
            
            # Compute headcount targets
            headcount_info = self._compute_headcount_targets(snapshot, year, global_params)
            
            if headcount_info.get('hires_needed', 0) <= 0:
                self.logger.info("No hires needed based on headcount targets")
                result.data = snapshot.copy()
                result.add_metadata('hires_needed', 0)
                return result
            
            hires_needed = headcount_info['hires_needed']
            self.logger.info(f"Need to hire {hires_needed} employees")
            
            # Generate hire events
            hire_events = hire.run(
                prev_snapshot=snapshot,
                simulation_year=year,
                global_params=global_params,
                rng=rng,
                census_template_path=census_template_path,
                rng_seed_offset=rng_seed_offset,
                target_hires=hires_needed
            )
            
            # Log hiring results
            if not hire_events.empty:
                self.logger.info(f"Generated {len(hire_events)} hire events")
                
                # Log additional statistics
                if EMP_ID in hire_events.columns:
                    unique_employees = hire_events[EMP_ID].nunique()
                    self.logger.info(f"  Hiring {unique_employees} unique new employees")
            else:
                self.logger.warning("No hire events generated despite needing hires")
            
            result.events = hire_events
            result.data = snapshot.copy()  # Will be updated with hire events later
            result.add_metadata('hires_needed', hires_needed)
            result.add_metadata('hire_events_count', len(hire_events))
            result.add_metadata('headcount_info', headcount_info)
            
        except Exception as e:
            self.logger.error(f"Error during hiring processing: {e}", exc_info=True)
            result.add_error(f"Hiring processing failed: {str(e)}")
            result.data = snapshot.copy()
        
        self.log_step_end(
            "Generate/apply hires",
            events_generated=len(result.events),
            success=result.success
        )
        
        return result
    
    def _compute_headcount_targets(self, snapshot: pd.DataFrame, year: int, 
                                 global_params: Any) -> Dict[str, Any]:
        """
        Compute headcount targets and hiring needs.
        
        Args:
            snapshot: Current employee snapshot
            year: Simulation year
            global_params: Global simulation parameters
            
        Returns:
            Dictionary with headcount information
        """
        try:
            # Use existing utility function
            headcount_info = compute_headcount_targets(
                snapshot=snapshot,
                simulation_year=year,
                global_params=global_params
            )
            
            # Log headcount details
            current_headcount = len(snapshot[snapshot.get(EMP_ACTIVE, True)]) if EMP_ACTIVE in snapshot.columns else len(snapshot)
            target_headcount = headcount_info.get('target_headcount', current_headcount)
            hires_needed = max(0, target_headcount - current_headcount)
            
            self.logger.info(f"Headcount analysis:")
            self.logger.info(f"  Current active employees: {current_headcount}")
            self.logger.info(f"  Target headcount: {target_headcount}")
            self.logger.info(f"  Hires needed: {hires_needed}")
            
            # Update the result dictionary
            headcount_info.update({
                'current_headcount': current_headcount,
                'target_headcount': target_headcount,
                'hires_needed': hires_needed
            })
            
            return headcount_info
            
        except Exception as e:
            self.logger.error(f"Error computing headcount targets: {e}")
            # Fallback to simple calculation
            current_headcount = len(snapshot)
            return {
                'current_headcount': current_headcount,
                'target_headcount': current_headcount,
                'hires_needed': 0,
                'error': str(e)
            }
    
    def validate_inputs(self, snapshot: pd.DataFrame, year: int, global_params: Any) -> bool:
        """Validate inputs for hiring processing."""
        if snapshot.empty:
            self.logger.warning("Empty snapshot provided for hiring processing")
            # This might be valid if we're starting from scratch
        
        # Check if we have basic required information
        if global_params is None:
            self.logger.error("No global parameters provided for hiring")
            return False
        
        if year < 2000 or year > 2100:
            self.logger.warning(f"Unusual simulation year: {year}")
        
        return True
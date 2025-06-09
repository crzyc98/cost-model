"""
Contribution processing functionality.

Handles the application of retirement plan contribution rules and calculations
for yearly snapshots.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Any

from .exceptions import SnapshotBuildError

logger = logging.getLogger(__name__)


class ContributionsProcessor:
    """Handles retirement plan contribution calculations."""
    
    def __init__(self):
        pass
    
    def apply_contribution_calculations(
        self,
        snapshot: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply retirement plan contribution calculations to snapshot.
        
        Args:
            snapshot: Yearly snapshot DataFrame
            config: Optional configuration for plan rules
            
        Returns:
            Updated snapshot with contribution calculations
        """
        logger.debug("Applying contribution calculations to yearly snapshot")
        
        # Check if contributions are already calculated
        contribution_columns = [
            'employee_contribution',
            'employer_core_contribution', 
            'employer_match_contribution'
        ]
        
        missing_columns = [col for col in contribution_columns if col not in snapshot.columns]
        
        if not missing_columns:
            logger.debug("Contribution columns already present in snapshot")
            return snapshot
        
        logger.info(f"Calculating missing contribution columns: {missing_columns}")
        
        try:
            # Try to use existing plan rules engine
            updated_snapshot = self._apply_plan_rules(snapshot, config)
            logger.debug("Successfully applied plan rules for contributions")
            return updated_snapshot
            
        except Exception as e:
            logger.warning(f"Plan rules calculation failed, using fallback: {e}")
            return self._apply_fallback_contributions(snapshot)
    
    def _apply_plan_rules(
        self,
        snapshot: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply plan rules using the existing plan rules engine.
        
        Args:
            snapshot: Snapshot DataFrame
            config: Plan configuration
            
        Returns:
            Updated snapshot with calculated contributions
        """
        try:
            # Import plan rules functionality
            from cost_model.rules.engine import apply_plan_rules
            
            # Apply plan rules to calculate contributions
            result = apply_plan_rules(snapshot, config or {})
            
            logger.debug("Plan rules applied successfully")
            return result
            
        except ImportError:
            logger.warning("Plan rules engine not available")
            raise SnapshotBuildError("Plan rules engine not available")
        except Exception as e:
            logger.error(f"Plan rules calculation failed: {e}")
            raise SnapshotBuildError(f"Plan rules calculation failed: {e}")
    
    def _apply_fallback_contributions(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fallback contribution calculations when plan rules fail.
        
        Args:
            snapshot: Snapshot DataFrame
            
        Returns:
            Updated snapshot with basic contribution calculations
        """
        logger.info("Applying fallback contribution calculations")
        
        snapshot = snapshot.copy()
        
        # Get compensation and deferral rate columns
        comp_col = 'employee_gross_compensation'
        deferral_col = 'employee_deferral_rate'
        active_col = 'active'
        
        # Default values for missing columns
        if comp_col not in snapshot.columns:
            logger.warning(f"Missing {comp_col}, defaulting to 50000")
            snapshot[comp_col] = 50000.0
        
        if deferral_col not in snapshot.columns:
            logger.warning(f"Missing {deferral_col}, defaulting to 0.03")
            snapshot[deferral_col] = 0.03
        
        if active_col not in snapshot.columns:
            logger.warning(f"Missing {active_col}, defaulting to True")
            snapshot[active_col] = True
        
        # Calculate employee contributions
        snapshot['employee_contribution'] = np.where(
            snapshot[active_col],
            snapshot[comp_col] * snapshot[deferral_col],
            0.0
        )
        
        # Calculate employer core contribution (flat 3% of compensation)
        snapshot['employer_core_contribution'] = np.where(
            snapshot[active_col],
            snapshot[comp_col] * 0.03,
            0.0
        )
        
        # Calculate employer match (50% of employee contribution up to 6% of compensation)
        max_match_base = snapshot[comp_col] * 0.06
        employee_contrib_eligible = snapshot['employee_contribution'].clip(upper=max_match_base)
        snapshot['employer_match_contribution'] = np.where(
            snapshot[active_col],
            employee_contrib_eligible * 0.5,
            0.0
        )
        
        # Add eligibility flag (simple rule: active employees are eligible)
        snapshot['is_eligible'] = snapshot[active_col]
        
        # Log summary statistics
        total_employee_contrib = snapshot['employee_contribution'].sum()
        total_employer_contrib = (
            snapshot['employer_core_contribution'].sum() +
            snapshot['employer_match_contribution'].sum()
        )
        
        logger.info(f"Fallback contributions calculated: "
                   f"Employee=${total_employee_contrib:,.0f}, "
                   f"Employer=${total_employer_contrib:,.0f}")
        
        return snapshot
    
    def validate_contributions(self, snapshot: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate contribution calculations.
        
        Args:
            snapshot: Snapshot with contribution calculations
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        contribution_columns = [
            'employee_contribution',
            'employer_core_contribution',
            'employer_match_contribution'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in contribution_columns if col not in snapshot.columns]
        if missing_columns:
            validation_results['errors'].append(f"Missing contribution columns: {missing_columns}")
            validation_results['valid'] = False
        
        # Check for negative contributions
        for col in contribution_columns:
            if col in snapshot.columns:
                negative_count = (snapshot[col] < 0).sum()
                if negative_count > 0:
                    validation_results['warnings'].append(
                        f"Found {negative_count} negative values in {col}"
                    )
        
        # Check for unrealistic contribution amounts
        if 'employee_contribution' in snapshot.columns:
            high_contrib = (snapshot['employee_contribution'] > 100000).sum()
            if high_contrib > 0:
                validation_results['warnings'].append(
                    f"Found {high_contrib} employees with contributions > $100,000"
                )
        
        # Check contribution ratios
        if all(col in snapshot.columns for col in ['employee_contribution', 'employee_gross_compensation']):
            contrib_data = snapshot[snapshot['employee_gross_compensation'] > 0]
            if not contrib_data.empty:
                contrib_ratio = contrib_data['employee_contribution'] / contrib_data['employee_gross_compensation']
                high_ratio = (contrib_ratio > 0.5).sum()  # More than 50% of compensation
                if high_ratio > 0:
                    validation_results['warnings'].append(
                        f"Found {high_ratio} employees contributing > 50% of compensation"
                    )
        
        return validation_results
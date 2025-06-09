"""
Employee status processing functionality.

Handles determination of employee status at end of year and related
status-based calculations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class StatusProcessor:
    """Handles employee status determination and processing."""
    
    def __init__(self):
        pass
    
    def apply_employee_status_eoy(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        """
        Apply end-of-year employee status to all employees in snapshot.
        
        Args:
            snapshot: Yearly snapshot DataFrame
            
        Returns:
            Updated snapshot with employee_status_eoy column
        """
        logger.debug("Applying end-of-year employee status")
        
        snapshot = snapshot.copy()
        
        # Apply status determination to each row
        snapshot['employee_status_eoy'] = snapshot.apply(
            self._determine_employee_status_eoy, axis=1
        )
        
        # Log status distribution
        status_distribution = snapshot['employee_status_eoy'].value_counts()
        logger.info(f"Employee status distribution: {status_distribution.to_dict()}")
        
        return snapshot
    
    def _determine_employee_status_eoy(self, row: pd.Series) -> str:
        """
        Determine employee status at end of year.
        
        This logic determines the employment status based on the employee's
        active status and termination date.
        
        Args:
            row: Employee record as pandas Series
            
        Returns:
            Employee status string
        """
        # Column names using original schema
        active_col = 'active'
        term_date_col = 'employee_termination_date'
        exited_col = 'exited'
        
        # Check if employee is active
        is_active = row.get(active_col, True)
        has_term_date = pd.notna(row.get(term_date_col))
        is_exited = row.get(exited_col, False)
        
        # Determine status based on conditions
        if is_active and not has_term_date and not is_exited:
            return 'ACTIVE'
        elif not is_active or has_term_date or is_exited:
            return 'TERMINATED'
        else:
            # Default case - should rarely happen
            logger.warning(f"Unclear employee status for {row.get('employee_id', 'unknown')}, defaulting to ACTIVE")
            return 'ACTIVE'
    
    def validate_employee_status(self, snapshot: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate employee status assignments.
        
        Args:
            snapshot: Snapshot with employee status
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'status_distribution': {}
        }
        
        status_col = 'employee_status_eoy'
        
        if status_col not in snapshot.columns:
            validation_results['errors'].append(f"Missing {status_col} column")
            validation_results['valid'] = False
            return validation_results
        
        # Get status distribution
        status_counts = snapshot[status_col].value_counts()
        validation_results['status_distribution'] = status_counts.to_dict()
        
        # Check for null statuses
        null_status = snapshot[status_col].isnull().sum()
        if null_status > 0:
            validation_results['warnings'].append(
                f"Found {null_status} employees with null status"
            )
        
        # Check for unexpected status values
        expected_statuses = {'ACTIVE', 'TERMINATED'}
        actual_statuses = set(snapshot[status_col].dropna().unique())
        unexpected_statuses = actual_statuses - expected_statuses
        
        if unexpected_statuses:
            validation_results['warnings'].append(
                f"Found unexpected status values: {unexpected_statuses}"
            )
        
        # Validate status consistency with other columns
        self._validate_status_consistency(snapshot, validation_results)
        
        return validation_results
    
    def _validate_status_consistency(
        self, 
        snapshot: pd.DataFrame, 
        validation_results: Dict[str, Any]
    ) -> None:
        """
        Validate that employee status is consistent with other columns.
        
        Args:
            snapshot: Snapshot DataFrame
            validation_results: Results dictionary to update
        """
        status_col = 'employee_status_eoy'
        active_col = 'active'
        term_date_col = 'employee_termination_date'
        
        # Check TERMINATED employees have termination dates or inactive status
        terminated_employees = snapshot[snapshot[status_col] == 'TERMINATED']
        if not terminated_employees.empty:
            # Check if terminated employees have termination dates OR are inactive
            if term_date_col in snapshot.columns and active_col in snapshot.columns:
                inconsistent_terminated = terminated_employees[
                    terminated_employees[term_date_col].isnull() & 
                    (terminated_employees[active_col] == True)
                ]
                
                if not inconsistent_terminated.empty:
                    validation_results['warnings'].append(
                        f"Found {len(inconsistent_terminated)} TERMINATED employees "
                        f"without termination dates who are still active"
                    )
        
        # Check ACTIVE employees don't have termination dates
        active_employees = snapshot[snapshot[status_col] == 'ACTIVE']
        if not active_employees.empty and term_date_col in snapshot.columns:
            active_with_term_date = active_employees[
                active_employees[term_date_col].notna()
            ]
            
            if not active_with_term_date.empty:
                validation_results['warnings'].append(
                    f"Found {len(active_with_term_date)} ACTIVE employees "
                    f"with termination dates"
                )
    
    def calculate_status_metrics(self, snapshot: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate metrics based on employee status.
        
        Args:
            snapshot: Snapshot with employee status
            
        Returns:
            Dictionary with status-based metrics
        """
        metrics = {}
        
        status_col = 'employee_status_eoy'
        comp_col = 'employee_gross_compensation'
        
        if status_col not in snapshot.columns:
            return metrics
        
        # Basic counts
        status_counts = snapshot[status_col].value_counts()
        metrics['status_counts'] = status_counts.to_dict()
        metrics['total_employees'] = len(snapshot)
        
        # Calculate percentages
        if len(snapshot) > 0:
            for status, count in status_counts.items():
                metrics[f'{status.lower()}_percentage'] = (count / len(snapshot)) * 100
        
        # Compensation metrics by status
        if comp_col in snapshot.columns:
            comp_by_status = snapshot.groupby(status_col)[comp_col].agg(['count', 'mean', 'sum'])
            metrics['compensation_by_status'] = comp_by_status.to_dict()
        
        # Contribution metrics by status
        contrib_cols = ['employee_contribution', 'employer_core_contribution', 'employer_match_contribution']
        for contrib_col in contrib_cols:
            if contrib_col in snapshot.columns:
                contrib_by_status = snapshot.groupby(status_col)[contrib_col].sum()
                metrics[f'{contrib_col}_by_status'] = contrib_by_status.to_dict()
        
        logger.debug(f"Calculated status metrics for {len(snapshot)} employees")
        
        return metrics
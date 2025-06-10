# cost_model/engines/run_one_year/orchestrator/diagnostic_utils.py
"""
Diagnostic utilities for orchestrator monitoring and debugging.
Extracted from the main orchestrator init file for better organization.
"""
import logging
from typing import Optional, Set
import pandas as pd

from cost_model.state.schema import EMP_ID, EMP_ACTIVE


def n_active(df: pd.DataFrame) -> int:
    """
    Count active employees in snapshot.

    Args:
        df: Snapshot DataFrame

    Returns:
        Number of active employees (EMP_ACTIVE == True)
    """
    if df.empty or EMP_ACTIVE not in df.columns:
        return 0
    return int(df[EMP_ACTIVE].sum())


def check_duplicates(df: pd.DataFrame, stage: str, logger: logging.Logger) -> None:
    """
    Check for duplicate employee IDs in snapshot and log warnings.

    Args:
        df: Snapshot DataFrame to check
        stage: Description of current processing stage
        logger: Logger instance for warnings
    """
    if df.empty or EMP_ID not in df.columns:
        return

    duplicates = df[EMP_ID].duplicated()
    if duplicates.any():
        duplicate_count = duplicates.sum()
        duplicate_ids = df[duplicates][EMP_ID].tolist()
        logger.warning(
            f"[{stage}] Found {duplicate_count} duplicate employee IDs: {duplicate_ids[:10]}..."
        )


def log_headcount_stage(df: pd.DataFrame, stage: str, year: int, logger: logging.Logger) -> None:
    """
    Log headcount information for a specific processing stage.

    Args:
        df: Snapshot DataFrame
        stage: Description of processing stage
        year: Simulation year
        logger: Logger instance for headcount information
    """
    total_employees = len(df)
    active_employees = n_active(df)
    inactive_employees = total_employees - active_employees

    logger.info(f"[{stage}] Year {year} headcount:")
    logger.info(f"  Total employees: {total_employees}")
    logger.info(f"  Active employees: {active_employees}")
    logger.info(f"  Inactive employees: {inactive_employees}")

    # Additional diagnostics if we have employee IDs
    if not df.empty and EMP_ID in df.columns:
        unique_ids = df[EMP_ID].nunique()
        if unique_ids != total_employees:
            logger.warning(f"  Warning: {unique_ids} unique IDs vs {total_employees} total rows")


def trace_snapshot_integrity(
    df: pd.DataFrame, 
    stage: str, 
    year: int, 
    logger: logging.Logger, 
    previous_ids: Optional[Set] = None
) -> Set:
    """
    Trace snapshot integrity across processing stages.

    Args:
        df: Current snapshot DataFrame
        stage: Description of processing stage
        year: Simulation year
        logger: Logger instance for integrity information
        previous_ids: Set of employee IDs from previous stage

    Returns:
        Set of current employee IDs for next stage comparison
    """
    if df.empty or EMP_ID not in df.columns:
        logger.warning(f"[{stage}] Empty snapshot or missing employee IDs")
        return set()

    current_ids = set(df[EMP_ID].unique())
    
    logger.info(f"[{stage}] Year {year} integrity check:")
    logger.info(f"  Current employee count: {len(current_ids)}")

    if previous_ids is not None:
        # Analyze changes from previous stage
        added_ids = current_ids - previous_ids
        removed_ids = previous_ids - current_ids
        retained_ids = current_ids & previous_ids

        logger.info(f"  Retained from previous: {len(retained_ids)}")
        
        if added_ids:
            logger.info(f"  Added employees: {len(added_ids)}")
            if len(added_ids) <= 10:
                logger.debug(f"    Added IDs: {list(added_ids)}")
            else:
                logger.debug(f"    Sample added IDs: {list(added_ids)[:10]}...")

        if removed_ids:
            logger.info(f"  Removed employees: {len(removed_ids)}")
            if len(removed_ids) <= 10:
                logger.debug(f"    Removed IDs: {list(removed_ids)}")
            else:
                logger.debug(f"    Sample removed IDs: {list(removed_ids)[:10]}...")

        # Check for unexpected changes
        if stage in ["Post-Promotions", "Post-Contributions"] and (added_ids or removed_ids):
            logger.warning(f"[{stage}] Unexpected employee changes (should only modify existing)")

    # Check for data quality issues
    check_duplicates(df, stage, logger)
    
    # Check active status distribution
    if EMP_ACTIVE in df.columns:
        active_count = df[EMP_ACTIVE].sum()
        inactive_count = len(df) - active_count
        logger.info(f"  Active/Inactive split: {active_count}/{inactive_count}")

    return current_ids


class DiagnosticTracker:
    """
    Helper class to track diagnostic information across orchestrator stages.
    """
    
    def __init__(self, logger: logging.Logger, year: int):
        self.logger = logger
        self.year = year
        self.previous_ids: Optional[Set] = None
        self.stage_history = []
    
    def track_stage(self, df: pd.DataFrame, stage: str) -> None:
        """
        Track a processing stage with full diagnostic logging.
        
        Args:
            df: Current snapshot DataFrame
            stage: Description of processing stage
        """
        self.logger.info(f"=== {stage} ===")
        
        # Log headcount
        log_headcount_stage(df, stage, self.year, self.logger)
        
        # Track integrity
        current_ids = trace_snapshot_integrity(
            df, stage, self.year, self.logger, self.previous_ids
        )
        
        # Update tracking state
        self.previous_ids = current_ids
        self.stage_history.append({
            'stage': stage,
            'employee_count': len(df),
            'active_count': n_active(df),
            'unique_ids': len(current_ids)
        })
    
    def get_summary(self) -> dict:
        """Get summary of all tracked stages."""
        return {
            'year': self.year,
            'stages': self.stage_history.copy(),
            'total_stages': len(self.stage_history)
        }
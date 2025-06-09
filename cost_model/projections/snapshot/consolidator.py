"""
Snapshot consolidation functionality.

Handles the consolidation of multiple yearly snapshots into single files
and other snapshot aggregation operations.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Union, Optional

from .exceptions import SnapshotBuildError

logger = logging.getLogger(__name__)


def consolidate_snapshots_to_parquet(
    snapshot_paths: List[Union[str, Path]], 
    output_path: Union[str, Path],
    include_age_calc: bool = True
) -> None:
    """
    Combine yearly snapshots into single parquet file.
    
    This is a wrapper around the existing consolidation functionality
    for backward compatibility.
    
    Args:
        snapshot_paths: List of paths to yearly snapshot files
        output_path: Path where consolidated file should be saved
        include_age_calc: Whether to include age calculations
        
    Raises:
        SnapshotBuildError: If consolidation fails
    """
    logger.info(f"Consolidating {len(snapshot_paths)} snapshots to {output_path}")
    
    try:
        # Import and use existing functionality
        from cost_model.projections.snapshot import consolidate_snapshots_to_parquet as original_func
        original_func(snapshot_paths, output_path)
        
        logger.info(f"Successfully consolidated snapshots to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to consolidate snapshots: {str(e)}")
        raise SnapshotBuildError(f"Snapshot consolidation failed: {str(e)}") from e


def load_and_consolidate_snapshots(snapshot_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load all snapshots from a directory and consolidate them.
    
    Args:
        snapshot_dir: Directory containing snapshot files
        
    Returns:
        Consolidated DataFrame with all snapshots
        
    Raises:
        SnapshotBuildError: If loading or consolidation fails
    """
    snapshot_dir = Path(snapshot_dir)
    
    if not snapshot_dir.exists():
        raise SnapshotBuildError(f"Snapshot directory does not exist: {snapshot_dir}")
    
    # Find all snapshot files
    snapshot_files = list(snapshot_dir.glob("*snapshot*.parquet"))
    
    if not snapshot_files:
        raise SnapshotBuildError(f"No snapshot files found in {snapshot_dir}")
    
    logger.info(f"Found {len(snapshot_files)} snapshot files in {snapshot_dir}")
    
    # Load and combine snapshots
    snapshots = []
    for file_path in sorted(snapshot_files):
        try:
            df = pd.read_parquet(file_path)
            snapshots.append(df)
            logger.debug(f"Loaded snapshot {file_path.name} with {len(df)} records")
        except Exception as e:
            logger.warning(f"Failed to load snapshot {file_path}: {e}")
    
    if not snapshots:
        raise SnapshotBuildError("No snapshots could be loaded")
    
    # Consolidate into single DataFrame
    consolidated = pd.concat(snapshots, ignore_index=True)
    logger.info(f"Consolidated {len(snapshots)} snapshots into {len(consolidated)} total records")
    
    return consolidated


def calculate_consolidated_metrics(consolidated_df: pd.DataFrame) -> dict:
    """
    Calculate summary metrics from consolidated snapshots.
    
    Args:
        consolidated_df: Consolidated snapshot DataFrame
        
    Returns:
        Dictionary with summary metrics
    """
    from cost_model.state.schema import SIMULATION_YEAR, EMP_ACTIVE, EMP_GROSS_COMP
    
    metrics = {}
    
    # Basic counts
    metrics['total_records'] = len(consolidated_df)
    
    if SIMULATION_YEAR in consolidated_df.columns:
        metrics['years_covered'] = consolidated_df[SIMULATION_YEAR].nunique()
        metrics['year_range'] = (
            consolidated_df[SIMULATION_YEAR].min(), 
            consolidated_df[SIMULATION_YEAR].max()
        )
    
    # Employee counts by year
    if EMP_ACTIVE in consolidated_df.columns and SIMULATION_YEAR in consolidated_df.columns:
        active_by_year = consolidated_df.groupby(SIMULATION_YEAR)[EMP_ACTIVE].sum()
        metrics['active_employees_by_year'] = active_by_year.to_dict()
        metrics['avg_active_employees'] = active_by_year.mean()
    
    # Compensation statistics
    if EMP_GROSS_COMP in consolidated_df.columns:
        comp_data = consolidated_df[EMP_GROSS_COMP].dropna()
        if not comp_data.empty:
            metrics['compensation_stats'] = {
                'mean': comp_data.mean(),
                'median': comp_data.median(),
                'std': comp_data.std(),
                'min': comp_data.min(),
                'max': comp_data.max()
            }
    
    logger.info(f"Calculated consolidated metrics: {len(metrics)} metric categories")
    return metrics
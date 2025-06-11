"""
Snapshot consolidation functionality.

Handles the consolidation of multiple yearly snapshots into single files
and other snapshot aggregation operations.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .exceptions import SnapshotBuildError

logger = logging.getLogger(__name__)


def consolidate_snapshots_to_parquet(
    snapshots_dir: Union[str, Path] = None,
    snapshot_paths: List[Union[str, Path]] = None,
    output_path: Union[str, Path] = None,
    include_age_calc: bool = True,
) -> None:
    """
    Combine yearly snapshots into single parquet file.

    This is a wrapper around the existing consolidation functionality
    for backward compatibility.

    Args:
        snapshots_dir: Directory containing yearly snapshot files (alternative to snapshot_paths)
        snapshot_paths: List of paths to yearly snapshot files (alternative to snapshots_dir)
        output_path: Path where consolidated file should be saved
        include_age_calc: Whether to include age calculations

    Raises:
        SnapshotBuildError: If consolidation fails
    """
    # Handle both calling conventions for backward compatibility
    if snapshots_dir is not None and snapshot_paths is None:
        # Convert snapshots_dir to snapshot_paths
        snapshots_dir = Path(snapshots_dir)
        snapshot_paths = list(snapshots_dir.glob("*.parquet"))
        logger.info(f"Found {len(snapshot_paths)} snapshot files in {snapshots_dir}")
    elif snapshot_paths is not None:
        # Use provided snapshot_paths directly
        pass
    else:
        raise ValueError("Either snapshots_dir or snapshot_paths must be provided")
    
    if snapshot_paths is not None:
        logger.debug(f"snapshot_paths type: {type(snapshot_paths)}, value: {snapshot_paths}")
        if isinstance(snapshot_paths, list):
            logger.info(f"Consolidating {len(snapshot_paths)} snapshots to {output_path}")
        else:
            logger.info(f"Consolidating snapshot paths (type: {type(snapshot_paths)}) to {output_path}")
    else:
        logger.info(f"Consolidating snapshots from directory to {output_path}")

    try:
        # Use the snapshot files directly instead of importing to avoid circular dependency
        if not snapshot_paths:
            logger.warning("No snapshot files found to consolidate")
            return
            
        # Read and combine all snapshot files
        all_snapshots = []
        for snapshot_file in snapshot_paths:
            logger.debug(f"Reading snapshot file: {snapshot_file}")
            df = pd.read_parquet(snapshot_file)
            all_snapshots.append(df)
        
        if not all_snapshots:
            logger.warning("No valid snapshot data found")
            return
            
        # Combine all snapshots
        logger.info(f"Combining {len(all_snapshots)} snapshot DataFrames")
        combined_df = pd.concat(all_snapshots, ignore_index=True)
        
        # Save the consolidated file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(output_path, index=False)
        
        logger.info(f"Consolidated {len(combined_df)} total records to {output_path}")

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
    from cost_model.state.schema import EMP_ACTIVE, EMP_GROSS_COMP, SIMULATION_YEAR

    metrics = {}

    # Basic counts
    metrics["total_records"] = len(consolidated_df)

    if SIMULATION_YEAR in consolidated_df.columns:
        metrics["years_covered"] = consolidated_df[SIMULATION_YEAR].nunique()
        metrics["year_range"] = (
            consolidated_df[SIMULATION_YEAR].min(),
            consolidated_df[SIMULATION_YEAR].max(),
        )

    # Employee counts by year
    if EMP_ACTIVE in consolidated_df.columns and SIMULATION_YEAR in consolidated_df.columns:
        active_by_year = consolidated_df.groupby(SIMULATION_YEAR)[EMP_ACTIVE].sum()
        metrics["active_employees_by_year"] = active_by_year.to_dict()
        metrics["avg_active_employees"] = active_by_year.mean()

    # Compensation statistics
    if EMP_GROSS_COMP in consolidated_df.columns:
        comp_data = consolidated_df[EMP_GROSS_COMP].dropna()
        if not comp_data.empty:
            metrics["compensation_stats"] = {
                "mean": comp_data.mean(),
                "median": comp_data.median(),
                "std": comp_data.std(),
                "min": comp_data.min(),
                "max": comp_data.max(),
            }

    logger.info(f"Calculated consolidated metrics: {len(metrics)} metric categories")
    return metrics

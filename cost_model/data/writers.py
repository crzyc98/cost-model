# cost_model/data/writers.py
"""
Functions for writing simulation outputs (snapshots, summaries).
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict

# Attempt to import column constants, provide fallbacks
try:
    # Assuming column definitions might be useful here too (e.g., for ordering)
    from cost_model.utils.columns import (
        EMP_BIRTH_DATE,
        EMP_HIRE_DATE,
        EMP_TERM_DATE,  # Example relevant date columns
        # Add other column constants if needed for output formatting/selection
    )
except ImportError:
    print(
        "Warning (writers.py): Could not import column constants from utils. Using string literals."
    )
    EMP_BIRTH_DATE, EMP_HIRE_DATE, EMP_TERM_DATE = (
        "employee_birth_date",
        "employee_hire_date",
        "employee_termination_date",
    )


logger = logging.getLogger(__name__)


# Define a custom exception for data writing errors
class DataWriteError(Exception):
    """Custom exception for errors during data writing."""

    pass


def write_snapshots(
    yearly_snapshots: Dict[int, pd.DataFrame],
    output_dir: Path,
    file_prefix: str = "snapshot",
) -> None:
    """
    Writes yearly snapshot DataFrames to Parquet files.

    Args:
        yearly_snapshots: Dictionary mapping simulation year (int) to DataFrame.
        output_dir: Path object for the directory to save files.
        file_prefix: Prefix for the output Parquet filenames.

    Raises:
        DataWriteError: If writing fails.
    """
    if not yearly_snapshots:
        logger.warning("No yearly snapshots provided to write.")
        return

    logger.info(f"Writing {len(yearly_snapshots)} yearly snapshots to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    for year, df in yearly_snapshots.items():
        # Define output path
        # Use simulation year from the snapshot data if available, otherwise use dict key
        sim_year = (
            df["simulation_year"].iloc[0] if "simulation_year" in df.columns else year
        )
        out_path = output_dir / f"{file_prefix}_year{sim_year}.parquet"
        logger.debug(
            f"Writing snapshot for year {sim_year} to {out_path} ({len(df)} records)"
        )
        try:
            # Optional: Select/reorder columns before saving
            # df_to_save = df[COLUMNS_TO_SAVE]
            df.to_parquet(out_path, index=False)
            logger.info(f"Wrote snapshot: {out_path}")
        except Exception:
            logger.exception(f"Failed to write snapshot file {out_path}")
            # Decide if you want to raise error or continue writing others
            # raise DataWriteError(f"Failed to write snapshot {out_path}") from e


def write_summary_metrics(
    metrics_df: pd.DataFrame, output_dir: Path, file_prefix: str = "summary"
) -> None:
    """
    Writes the summary metrics DataFrame to a CSV file.

    Args:
        metrics_df: DataFrame containing summary metrics per year/scenario.
        output_dir: Path object for the directory to save the file.
        file_prefix: Prefix for the output CSV filename.

    Raises:
        DataWriteError: If writing fails.
    """
    if metrics_df is None or metrics_df.empty:
        logger.warning("No summary metrics DataFrame provided to write.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    summary_path = output_dir / f"{file_prefix}_metrics.csv"
    logger.info(f"Writing summary metrics to {summary_path}...")

    try:
        # Format dates if any exist (though unlikely in summary)
        # Example:
        # for col in metrics_df.select_dtypes(include=['datetime64[ns]']).columns:
        #     metrics_df[col] = metrics_df[col].dt.strftime('%Y-%m-%d')

        metrics_df.to_csv(
            summary_path, index=False, float_format="%.4f"
        )  # Control float precision
        logger.info(
            f"Wrote summary metrics: {summary_path}\n{metrics_df.head().to_string()}"
        )
        return summary_path
    except Exception as e:
        logger.exception(f"Failed to write summary metrics file {summary_path}")
        raise DataWriteError(f"Failed to write summary metrics {summary_path}") from e


# Example Usage (for testing this module directly)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s",
    )
    test_output_dir = Path("./_temp_writer_output")
    test_output_dir.mkdir(exist_ok=True)

    # Create dummy snapshot data
    snapshots = {
        2025: pd.DataFrame(
            {
                "employee_id": ["A1", "A2"],
                "simulation_year": [2025, 2025],
                "value": [10, 20],
            }
        ),
        2026: pd.DataFrame(
            {
                "employee_id": ["A1", "A3"],
                "simulation_year": [2026, 2026],
                "value": [11, 30],
            }
        ),
    }
    # Create dummy metrics data
    metrics = pd.DataFrame(
        {"year": [2025, 2026], "headcount": [2, 2], "avg_value": [15.0, 20.5]}
    )

    print("\n--- Testing Snapshot Writer ---")
    try:
        write_snapshots(snapshots, test_output_dir, file_prefix="test_snap")
    except DataWriteError as e:
        print(f"ERROR writing snapshots: {e}")

    print("\n--- Testing Metrics Writer ---")
    try:
        write_summary_metrics(metrics, test_output_dir, file_prefix="test_summary")
    except DataWriteError as e:
        print(f"ERROR writing metrics: {e}")

    print(f"\nCheck directory: {test_output_dir.resolve()}")
    # Clean up dummy files/dir (optional)
    # import shutil
    # shutil.rmtree(test_output_dir)

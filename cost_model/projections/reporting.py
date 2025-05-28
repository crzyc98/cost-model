# cost_model/projections/reporting.py
"""
Reporting module for cost model projections.
This module handles the creation of summary statistics, plots, and reports
from simulation results. It uses canonical column names from cost_model.state.schema
to ensure consistency across the codebase.

QuickStart: see docs/cost_model/projections/reporting.md
"""

# Summary and plotting logic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Optional, List

# Import canonical column names from schema
from cost_model.state import schema

# To prevent GUI errors on headless servers, and for consistency:
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

logger = logging.getLogger(__name__)

def plot_projection_results(summary_df: pd.DataFrame, output_dir: Path):
    """
    Plots key projection results from the summary DataFrame.
    Uses canonical column names defined in schema.py and handles standardization
    of variant column names for consistent plotting.
    """
    if summary_df.empty:
        logger.warning("Summary DataFrame is empty. Skipping plotting.")
        return

    logger.info(f"Plotting projection results to {output_dir}...")
    logger.debug(f"Summary DataFrame columns: {summary_df.columns.tolist()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standardize year column first, then other summary columns for plotting
    standardized_year_df = _standardize_year_column(summary_df.copy())
    plot_df = _standardize_summary_columns_for_plotting(standardized_year_df)

    # Define required columns using schema constants
    required_cols = [
        schema.SUMMARY_YEAR,
        schema.SUMMARY_ACTIVE_HEADCOUNT,
        schema.SUMMARY_TERMINATIONS
    ]
    missing_cols = [col for col in required_cols if col not in plot_df.columns]

    if missing_cols:
        logger.warning(f"Summary DataFrame missing required columns for plotting: {missing_cols}. "
                      f"Available columns: {plot_df.columns.tolist()}. Skipping plotting.")
        return

    try:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Active Headcount', color=color)
        ax1.plot(plot_df[schema.SUMMARY_YEAR], plot_df[schema.SUMMARY_ACTIVE_HEADCOUNT], color=color, marker='o', linestyle='-')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Terminations', color=color)  # we already handled the x-label with ax1
        ax2.plot(plot_df[schema.SUMMARY_YEAR], plot_df[schema.SUMMARY_TERMINATIONS], color=color, marker='x', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Projection Summary: Headcount and Terminations')
        plot_path = output_dir / "projection_summary_plot.png"
        plt.savefig(plot_path)
        logger.info(f"Saved summary plot to {plot_path}")
        plt.close(fig) # Close the figure to free memory

        # Add more plots as needed, e.g., for contributions if that data is in summary_df
        contrib_cols = ['total_contributions', 'Total EE Contribution', 'Total ER Contribution']
        contrib_col = None
        for col in contrib_cols:
            if col in plot_df.columns:
                contrib_col = col
                break

        if contrib_col:
            fig_contrib, ax_contrib = plt.subplots(figsize=(10, 6))
            color = 'tab:green'
            ax_contrib.set_xlabel('Year')
            ax_contrib.set_ylabel('Total Contributions', color=color)
            ax_contrib.bar(plot_df[schema.SUMMARY_YEAR], plot_df[contrib_col], color=color, alpha=0.7)
            ax_contrib.tick_params(axis='y', labelcolor=color)
            ax_contrib.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')) # Format as currency or number
            plt.title('Projection Summary: Total Contributions')
            contrib_plot_path = output_dir / "projection_contributions_plot.png"
            plt.savefig(contrib_plot_path)
            logger.info(f"Saved contributions plot to {contrib_plot_path}")
            plt.close(fig_contrib)

    except Exception as e:
        logger.error(f"Error during plotting: {e}", exc_info=True)


def _standardize_summary_columns_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize various summary column names to their canonical forms defined in schema.
    This ensures consistent column names across all summary statistics and plots.
    """
    # Dictionary of possible column name variants mapped to their canonical schema constants
    column_map = {
        "Active Headcount": schema.SUMMARY_ACTIVE_HEADCOUNT,
        "active_employees": schema.SUMMARY_ACTIVE_HEADCOUNT,
        "Terminations": schema.SUMMARY_TERMINATIONS,
        "Employee Contributions": schema.SUMMARY_TOTAL_EE_CONTRIBUTIONS,
        "Employer Contributions": schema.SUMMARY_TOTAL_ER_CONTRIBUTIONS,
        "Total Contributions": schema.SUMMARY_TOTAL_CONTRIBUTIONS,
        "Average Compensation": schema.SUMMARY_AVG_COMP,
        "Average Tenure": schema.SUMMARY_AVG_TENURE,
        "Total Benefits": schema.SUMMARY_TOTAL_BENEFITS,
        "New Hires": schema.SUMMARY_NEW_HIRES,
        "New Hire Terminations": schema.SUMMARY_NEW_HIRE_TERMINATIONS,
    }

    # Track which standardizations were applied for logging
    applied_standardizations = []

    # Apply renaming where columns exist
    for old_col, new_col in column_map.items():
        if old_col in df.columns and old_col != new_col:
            df = df.rename(columns={old_col: new_col})
            applied_standardizations.append(f"{old_col} -> {new_col}")

    if applied_standardizations:
        logger.info(f"Standardized column names: {', '.join(applied_standardizations)}")

    return df


def _standardize_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the year column to the canonical name defined in schema.SUMMARY_YEAR.
    Handle different possible column names for year across datasets.
    """
    canonical = schema.SUMMARY_YEAR
    variants = ["Projection Year", "Year", "simulation_year", schema.SIMULATION_YEAR]
    
    # Check if canonical already exists along with variants
    found_variants = [v for v in variants if v in df.columns]
    
    if canonical in df.columns and found_variants:
        logger.warning(
            f"Both canonical '{canonical}' and variant(s) {found_variants} found in summary. "
            f"Keeping '{canonical}' column."
        )
        df = df.drop(columns=found_variants)
    elif found_variants:
        # Use the first variant found and rename to canonical
        df = df.rename(columns={found_variants[0]: canonical})
        logger.info(
            f"Standardized year column from variant '{found_variants[0]}' to "
            f"canonical '{canonical}'."
        )
    return df


def save_detailed_results(
    output_path: Path,
    scenario_name: str,
    final_snapshot: pd.DataFrame,
    full_event_log: pd.DataFrame,
    summary_statistics: pd.DataFrame,
    employment_status_summary_df: pd.DataFrame,
    yearly_snapshots: Dict[int, pd.DataFrame] = None,
    config_to_save: SimpleNamespace = None
) -> None:
    """Save detailed results to the specified output path."""
    logger.info(f"Saving detailed results for '{scenario_name}' to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter out terminated employees from prior years
    if 'employee_termination_date' in final_snapshot.columns and 'simulation_year' in final_snapshot.columns:
        # Keep active employees and those terminated in the current simulation year
        current_year = final_snapshot['simulation_year'].max()
        filtered_snapshot = final_snapshot[
            (final_snapshot['employee_termination_date'].isna()) |
            (pd.to_datetime(final_snapshot['employee_termination_date']).dt.year == current_year)
        ]
        logger.info(f"Filtered out {len(final_snapshot) - len(filtered_snapshot)} employees terminated in prior years")
        final_snapshot = filtered_snapshot

    # Save final EOY snapshot
    final_snapshot_path = output_path / f"{scenario_name}_final_eoy_snapshot.parquet"
    final_snapshot.to_parquet(final_snapshot_path, index=False)
    logger.info(f"Final EOY snapshot saved to {final_snapshot_path}")

    # Save full event log with enhanced validation
    event_log_path = output_path / f"{scenario_name}_final_cumulative_event_log.parquet"

    # Ensure we have a DataFrame with the expected columns
    if full_event_log is None or not isinstance(full_event_log, pd.DataFrame) or full_event_log.empty:
        logger.warning("No event log data provided or empty event log. Creating an empty event log with expected schema.")
        from cost_model.state.event_log import EVENT_COLS
        full_event_log = pd.DataFrame(columns=EVENT_COLS)

    # Log some statistics about the event log
    logger.info(f"Saving event log with {len(full_event_log)} events")
    if not full_event_log.empty:
        event_counts = full_event_log['event_type'].value_counts()
        logger.info(f"Event type counts in final log:\n{event_counts.to_string()}")

        # Check for required columns
        required_columns = ['event_id', 'event_type', 'event_time', 'employee_id']
        missing_columns = [col for col in required_columns if col not in full_event_log.columns]
        if missing_columns:
            logger.warning(f"Event log is missing required columns: {missing_columns}")

    # Save the event log
    try:
        full_event_log.to_parquet(event_log_path, index=False)
        logger.info(f"Final cumulative event log saved to {event_log_path} with {len(full_event_log)} events")

        # Verify the saved file
        if event_log_path.exists():
            file_size = event_log_path.stat().st_size
            logger.info(f"Event log file size: {file_size} bytes")

            # Read back a small sample to verify
            try:
                sample = pd.read_parquet(event_log_path, engine='pyarrow')
                logger.info(f"Successfully read back event log with {len(sample)} events")
                if not sample.empty:
                    logger.info(f"Sample event types: {sample['event_type'].value_counts().to_dict()}")
            except Exception as e:
                logger.error(f"Error reading back event log: {e}")
        else:
            logger.error("Failed to save event log: file not created")
    except Exception as e:
        logger.error(f"Error saving event log: {e}", exc_info=True)
        raise

    # Save summary statistics merging core summary with full employment status counts
    summary_to_save = summary_statistics.copy()
    import cost_model.state.event_log as schema

    # Add new_hires and terminated_employees columns using event log
    if full_event_log is not None and not full_event_log.empty:
        if 'simulation_year' in full_event_log.columns:
            hires_per_year = full_event_log.query("event_type == @schema.EVT_HIRE").groupby('simulation_year').size()
            terms_per_year = full_event_log.query("event_type == @schema.EVT_TERM").groupby('simulation_year').size()

            # Check if 'year' column exists in summary_to_save
            if 'year' not in summary_to_save.columns:
                # Try to use simulation_year if available in summary_statistics
                if 'simulation_year' in summary_to_save.columns:
                    logger.info("Using 'simulation_year' as 'year' for event mapping")
                    summary_to_save['year'] = summary_to_save['simulation_year']
                # If we have Projection Year, it might have been used instead
                elif 'Projection Year' in summary_to_save.columns:
                    logger.info("Using 'Projection Year' as 'year' for event mapping")
                    summary_to_save['year'] = summary_to_save['Projection Year']
                else:
                    # Create year column from available years in event log
                    logger.warning("No year column found in summary. Creating one from event log years.")
                    event_years = sorted(full_event_log['simulation_year'].unique())
                    if len(event_years) > 0:
                        if not summary_to_save.empty and len(summary_to_save) == len(event_years):
                            summary_to_save['year'] = event_years
                        else:
                            logger.warning("Cannot create year column: summary rows don't match event years.")
                            # Create placeholder to prevent further errors
                            summary_to_save['year'] = summary_to_save.index
                    else:
                        logger.warning("No simulation years found in event log.")
                        # Create placeholder to prevent further errors
                        summary_to_save['year'] = summary_to_save.index

            # Now we can safely map the hires and terminations
            if 'year' in summary_to_save.columns:
                summary_to_save['new_hires'] = summary_to_save['year'].map(hires_per_year).fillna(0).astype(int)
                summary_to_save['terminated_employees'] = summary_to_save['year'].map(terms_per_year).fillna(0).astype(int)
            else:
                logger.error("Failed to create 'year' column in summary_to_save. Cannot map hires/terminations.")
                # Add default values to avoid further errors
                summary_to_save['new_hires'] = 0
                summary_to_save['terminated_employees'] = 0
        else:
            logger.warning("simulation_year column missing in event log; cannot compute new_hires/terminated_employees by year.")
    else:
        logger.warning("Event log missing or empty; cannot compute new_hires/terminated_employees.")

    # Standardize year column in summary_to_save before attempting merge
    summary_to_save = _standardize_year_column(summary_to_save)

    if employment_status_summary_df is not None and not employment_status_summary_df.empty:
        logger.info("Merging employment status summary with main summary statistics")

        # Create completely new DataFrames with guaranteed unique columns
        # This works around any internal pandas column representation issues

        # First, create a clean copy of summary_to_save with guaranteed unique columns
        summary_clean = pd.DataFrame()
        summary_cols_seen = set()
        for col in summary_to_save.columns:
            # Create a unique column name if this one already exists
            col_clean = col
            count = 1
            while col_clean in summary_cols_seen:
                col_clean = f"{col}_{count}"
                count += 1
            summary_cols_seen.add(col_clean)
            summary_clean[col_clean] = summary_to_save[col].copy()

        # Now create a clean copy of the employment status DataFrame
        emp_clean = pd.DataFrame()
        emp_cols_seen = set()
        for col in employment_status_summary_df.columns:
            # If this column would conflict with summary, prefix it
            if col in summary_cols_seen and col.lower() not in ['year', 'projection year', 'simulation_year']:
                col_clean = f"emp_status_{col}"
            else:
                col_clean = col

            # Make sure the name is unique
            count = 1
            while col_clean in emp_cols_seen:
                col_clean = f"{col_clean}_{count}"
                count += 1

            emp_cols_seen.add(col_clean)
            emp_clean[col_clean] = employment_status_summary_df[col].copy()

        # Now identify the year columns for merging
        year_columns = ['Year', 'Projection Year', 'year', 'simulation_year']

        # Find year column in summary_clean
        summary_year_col = None
        for col in year_columns:
            if col in summary_clean.columns:
                summary_year_col = col
                break

        # Find year column in emp_clean
        emp_year_col = None
        for col in year_columns:
            if col in emp_clean.columns:
                emp_year_col = col
                break

        # If either DataFrame is missing a year column, create one
        if summary_year_col is None:
            logger.warning("No year column found in summary. Creating index-based year column.")
            summary_clean['year'] = range(len(summary_clean))
            summary_year_col = 'year'

        if emp_year_col is None:
            logger.warning("No year column found in employment status. Creating index-based year column.")
            emp_clean['year'] = range(len(emp_clean))
            emp_year_col = 'year'

        # Standardize to a common year column name if needed
        if summary_year_col != emp_year_col:
            # Choose 'year' as the standard
            standard_year_col = 'year'

            # Rename in summary if needed
            if summary_year_col != standard_year_col:
                summary_clean[standard_year_col] = summary_clean[summary_year_col].copy()
                summary_clean = summary_clean.drop(columns=[summary_year_col])
                logger.info(f"Created standard '{standard_year_col}' column in summary from '{summary_year_col}'")

            # Rename in employment status if needed
            if emp_year_col != standard_year_col:
                emp_clean[standard_year_col] = emp_clean[emp_year_col].copy()
                emp_clean = emp_clean.drop(columns=[emp_year_col])
                logger.info(f"Created standard '{standard_year_col}' column in employment status from '{emp_year_col}'")

            # Set the merge column to our standard
            merge_col = standard_year_col
        else:
            # If they already match, use that as the merge column
            merge_col = summary_year_col

        logger.info(f"Will merge DataFrames on column: {merge_col}")

        # Now perform the merge with our clean DataFrames
        try:
            # Convert the merge column to the same type in both DataFrames to avoid merge issues
            summary_clean[merge_col] = summary_clean[merge_col].astype(str)
            emp_clean[merge_col] = emp_clean[merge_col].astype(str)

            merged_df = summary_clean.merge(
                emp_clean,
                on=merge_col,
                how='left'
            )

            logger.info(f"Successfully merged summary and employment status data with {len(merged_df)} rows")
            logger.debug(f"Merged columns ({len(merged_df.columns)}): {sorted(merged_df.columns.tolist())}")

            # Replace our working copy with the merged result
            summary_to_save = merged_df
        except Exception as e:
            logger.error(f"Error during merge: {e}")
            logger.warning("Using only summary statistics without employment status data")
            # Keep just the summary data if merge fails
            summary_to_save = summary_clean

    # Standardize year column naming to prevent conflicts
    summary_to_save = _standardize_year_column(summary_to_save)

    # Convert dictionary columns to strings to avoid parquet serialization issues
    for col in summary_to_save.columns:
        if summary_to_save[col].apply(lambda x: isinstance(x, dict)).any():
            summary_to_save[col] = summary_to_save[col].apply(
                lambda x: str({str(k): v for k, v in x.items()}) if isinstance(x, dict) else x
            )

    summary_path = output_path / f"{scenario_name}_summary_statistics.parquet"
    summary_to_save.to_parquet(summary_path, index=False)
    logger.info(f"Summary statistics saved to {summary_path}")

    # Save employment status summary
    emp_status_path = output_path / f"{scenario_name}_employment_status_summary.parquet"

    # Convert dictionary columns to strings in employment status summary
    if employment_status_summary_df is not None and not employment_status_summary_df.empty:
        emp_status_df = employment_status_summary_df.copy()
        for col in emp_status_df.columns:
            if emp_status_df[col].apply(lambda x: isinstance(x, dict)).any():
                emp_status_df[col] = emp_status_df[col].apply(
                    lambda x: str({str(k): v for k, v in x.items()}) if isinstance(x, dict) else x
                )
        emp_status_df.to_parquet(emp_status_path, index=False)
        logger.info(f"Employment status summary saved to {emp_status_path}")
    else:
        logger.warning("No employment status summary data to save")

    # Save yearly snapshots if provided
    if yearly_snapshots:
        yearly_snapshots_dir = output_path / "yearly_snapshots"
        yearly_snapshots_dir.mkdir(exist_ok=True)
        for year, snapshot_df in yearly_snapshots.items():
            year_path = yearly_snapshots_dir / f"{scenario_name}_snapshot_{year}.parquet"
            snapshot_df.to_parquet(year_path, index=False)
        logger.info(f"Yearly snapshots saved to {yearly_snapshots_dir}")

    # Save config
    if config_to_save:
        config_path = output_path / f"{scenario_name}_run_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_to_namespace_to_dict(config_to_save), f, default_flow_style=False)
        logger.info(f"Run config saved to {config_path}")

    logger.info(f"Detailed results for '{scenario_name}' saved to {output_path}")
    
    # Upstream recommendation for future standardization
    logger.debug("NOTE: For optimal schema compliance, upstream summary generation processes "
                "should use canonical column names from cost_model.state.schema "
                "(e.g., schema.SUMMARY_YEAR, schema.SUMMARY_ACTIVE_HEADCOUNT, etc.).")

def config_to_namespace_to_dict(config: SimpleNamespace) -> Dict[str, Any]:
    return config.__dict__

# cost_model/projections/reporting.py
"""
Reporting module for cost model projections.
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
from typing import Dict, Any, Optional

# To prevent GUI errors on headless servers, and for consistency:
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

logger = logging.getLogger(__name__)

def plot_projection_results(summary_df: pd.DataFrame, output_dir: Path):
    """
    Plots key projection results from the summary DataFrame.
    Example: Plot active headcount and terminations over years.
    """
    if summary_df.empty or not all(col in summary_df.columns for col in ['year', 'active_headcount_eoy', 'terminations']):
        logger.warning("Summary DataFrame is empty or missing required columns for plotting. Skipping plotting.")
        return

    logger.info(f"Plotting projection results to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Active Headcount EOY', color=color)
        ax1.plot(summary_df['year'], summary_df['active_headcount_eoy'], color=color, marker='o', linestyle='-')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Terminations During Year', color=color)  # we already handled the x-label with ax1
        ax2.plot(summary_df['year'], summary_df['terminations'], color=color, marker='x', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Projection Summary: Headcount and Terminations')
        plot_path = output_dir / "projection_summary_plot.png"
        plt.savefig(plot_path)
        logger.info(f"Saved summary plot to {plot_path}")
        plt.close(fig) # Close the figure to free memory

        # Add more plots as needed, e.g., for contributions if that data is in summary_df
        if 'total_contributions' in summary_df.columns:
            fig_contrib, ax_contrib = plt.subplots(figsize=(10, 6))
            color = 'tab:green'
            ax_contrib.set_xlabel('Year')
            ax_contrib.set_ylabel('Total Contributions', color=color)
            ax_contrib.bar(summary_df['year'], summary_df['total_contributions'], color=color, alpha=0.7)
            ax_contrib.tick_params(axis='y', labelcolor=color)
            ax_contrib.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')) # Format as currency or number
            plt.title('Projection Summary: Total Contributions')
            contrib_plot_path = output_dir / "projection_contributions_plot.png"
            plt.savefig(contrib_plot_path)
            logger.info(f"Saved contributions plot to {contrib_plot_path}")
            plt.close(fig_contrib)

    except Exception as e:
        logger.error(f"Error during plotting: {e}", exc_info=True)

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
            summary_to_save['new_hires'] = summary_to_save['year'].map(hires_per_year).fillna(0).astype(int)
            summary_to_save['terminated_employees'] = summary_to_save['year'].map(terms_per_year).fillna(0).astype(int)
        else:
            logger.warning("simulation_year column missing in event log; cannot compute new_hires/terminated_employees by year.")
    else:
        logger.warning("Event log missing or empty; cannot compute new_hires/terminated_employees.")
    
    if employment_status_summary_df is not None and not employment_status_summary_df.empty:
        emp_df = employment_status_summary_df.rename(columns={'Year': 'Projection Year'})
        summary_to_save = summary_to_save.merge(
            emp_df,
            on='Projection Year',
            how='left'
        )
    
    # Rename 'Projection Year' to 'year' if present
    if 'Projection Year' in summary_to_save.columns:
        summary_to_save = summary_to_save.rename(columns={'Projection Year': 'year'})
    
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

def config_to_namespace_to_dict(config: SimpleNamespace) -> Dict[str, Any]:
    return config.__dict__

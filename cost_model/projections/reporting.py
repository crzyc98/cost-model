# cost_model/projections/reporting.py
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
    
    # Save final EOY snapshot
    final_snapshot_path = output_path / f"{scenario_name}_final_eoy_snapshot.parquet"
    final_snapshot.to_parquet(final_snapshot_path, index=False)
    logger.info(f"Final EOY snapshot saved to {final_snapshot_path}")
    
    # Save full event log
    event_log_path = output_path / f"{scenario_name}_final_cumulative_event_log.parquet"
    full_event_log.to_parquet(event_log_path, index=False)
    logger.info(f"Final cumulative event log saved to {event_log_path}")
    
    # Save summary statistics
    summary_path = output_path / f"{scenario_name}_summary_statistics.parquet"
    summary_statistics.to_parquet(summary_path, index=False)
    logger.info(f"Summary statistics saved to {summary_path}")
    
    # Save employment status summary
    emp_status_path = output_path / f"{scenario_name}_employment_status_summary.parquet"
    employment_status_summary_df.to_parquet(emp_status_path, index=False)
    logger.info(f"Employment status summary saved to {emp_status_path}")
    
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

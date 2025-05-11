# scripts/employment_status_summary.py

import pandas as pd
import glob
import os
import numpy as np


def main():
    # Read per-year snapshots with employment_status from the projection output
    files = sorted(glob.glob("output_dev/*/yearly_snapshots/*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    summary = []
    for i, df in enumerate(dfs):
        year = int(os.path.basename(files[i]).split("_")[-1].split(".")[0][4:])
        status_counts = df.groupby("employment_status").size().to_dict()

        # Define keys for all relevant statuses
        continuous_active_key = "Continuous Active"
        new_hire_active_key = "New Hire Active"
        exp_term_key = "Experienced Terminated"  # Assuming this is the exact name
        nh_term_key = "New Hire Terminated"  # Assuming this is the exact name

        # Get counts for active statuses
        num_continuous_active = status_counts.get(continuous_active_key, 0)
        num_new_hires_active = status_counts.get(new_hire_active_key, 0)
        active_count = num_continuous_active + num_new_hires_active

        # --- ADDITION: Calculate Total Terminated ---
        num_exp_term = status_counts.get(exp_term_key, 0)
        num_nh_term = status_counts.get(nh_term_key, 0)
        total_terminated = num_exp_term + num_nh_term
        # --- END OF ADDITION ---

        # Calculate growth rate
        prev_active = summary[-1]["Active"] if i > 0 else np.nan
        growth_rate = (
            (active_count - prev_active) / prev_active
            if i > 0 and prev_active > 0
            else np.nan
        )

        # Construct the row, including the new total_terminated count
        # Keep **status_counts if you still want the individual columns too
        row = {
            "Year": year,
            **status_counts,  # Keeps individual status columns
            "Total Terminated": total_terminated,  # Add the combined column
            "Active": active_count,
            "ActiveGrowthRate": growth_rate,
        }
        summary.append(row)

    summary_df = pd.DataFrame(summary).fillna(0)  # Fill NaNs with 0 for numeric columns

    # --- Optional: Reorder columns for better readability ---
    desired_order = [
        "Year",
        continuous_active_key,
        exp_term_key,
        new_hire_active_key,
        nh_term_key,
        "Total Terminated",
        "Active",
        "ActiveGrowthRate",
    ]
    # Include only columns that actually exist in the summary_df
    final_columns = [col for col in desired_order if col in summary_df.columns]
    # Add any extra status columns that might have appeared but aren't in desired_order
    extra_cols = [col for col in summary_df.columns if col not in final_columns]
    summary_df = summary_df[final_columns + extra_cols]
    # --- End of Optional Reordering ---

    summary_df = summary_df.sort_values("Year")
    summary_df.to_csv(
        "output_dev/TinyDev/TinyDev_employment_status_summary.csv", index=False
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

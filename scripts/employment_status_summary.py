# scripts/employment_status_summary.py

import glob
import os

import numpy as np
import pandas as pd


def main():
    # Read per-year snapshots with employment_status from the projection output
    # Look for snapshot files in both old and new directory structures
    files = []

    # Try new structure: output_*/*/year=YYYY/snapshot.parquet
    new_pattern_files = sorted(glob.glob("output_*/*/year=*/snapshot.parquet"))
    if new_pattern_files:
        files.extend(new_pattern_files)

    # Try old structure: output_dev/*/yearly_snapshots/*.parquet
    old_pattern_files = sorted(glob.glob("output_dev/*/yearly_snapshots/*.parquet"))
    if old_pattern_files:
        files.extend(old_pattern_files)

    if not files:
        print("No snapshot files found. Checked patterns:")
        print("  - output_*/*/year=*/snapshot.parquet")
        print("  - output_dev/*/yearly_snapshots/*.parquet")
        return

    files = sorted(set(files))  # Remove duplicates and sort
    print(f"Found {len(files)} snapshot files:")
    for f in files:
        print(f"  {f}")

    dfs = []
    years = []

    for file_path in files:
        try:
            df = pd.read_parquet(file_path)

            # Extract year from file path
            if "/year=" in file_path:
                # New structure: extract from year=YYYY directory
                year_str = file_path.split("/year=")[1].split("/")[0]
                year = int(year_str)
            else:
                # Old structure: extract from filename
                filename = os.path.basename(file_path)
                if "_" in filename and filename.split("_")[-1].split(".")[0][4:]:
                    year = int(filename.split("_")[-1].split(".")[0][4:])
                else:
                    print(f"Warning: Could not extract year from {file_path}, skipping")
                    continue

            dfs.append(df)
            years.append(year)
            print(f"  Loaded {file_path} for year {year} with {len(df)} records")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if not dfs:
        print("No valid snapshot files could be loaded")
        return

    summary = []
    for i, (df, year) in enumerate(zip(dfs, years)):
        # Check for employment status column - try multiple possible names
        status_column = None
        if "employment_status" in df.columns:
            status_column = "employment_status"
        elif "employee_status_eoy" in df.columns:
            status_column = "employee_status_eoy"
        elif "active" in df.columns:
            # Fallback: create simple status from active column
            df = df.copy()
            df["derived_status"] = df["active"].map({True: "Active", False: "Inactive"})
            status_column = "derived_status"

        if status_column is None:
            print(
                f"Warning: No suitable status column found in {files[i] if i < len(files) else 'file'}"
            )
            print(f"Available columns: {list(df.columns)}")
            continue

        # Filter out null/NaN values before grouping
        valid_status_df = df[df[status_column].notna()]
        if len(valid_status_df) == 0:
            print(f"Warning: All status values are null in {status_column} column for year {year}")
            # Create a basic summary using active column if available
            if "active" in df.columns:
                print(f"  Falling back to 'active' column for year {year}")
                df_copy = df.copy()
                df_copy["derived_status"] = df_copy["active"].map(
                    {True: "Active", False: "Inactive"}
                )
                status_counts = df_copy.groupby("derived_status").size().to_dict()
            else:
                continue
        else:
            status_counts = valid_status_df.groupby(status_column).size().to_dict()

        # Define keys for all relevant statuses
        continuous_active_key = "Continuous Active"
        new_hire_active_key = "New Hire Active"
        exp_term_key = "Experienced Terminated"  # Assuming this is the exact name
        nh_term_key = "New Hire Terminated"  # Assuming this is the exact name

        # Calculate active count - handle both detailed and simple status formats
        if "Active" in status_counts and "Inactive" in status_counts:
            # Simple active/inactive format (fallback from active column)
            active_count = status_counts.get("Active", 0)
        else:
            # Detailed employment status format
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
            (active_count - prev_active) / prev_active if i > 0 and prev_active > 0 else np.nan
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

    if not summary:
        print("No valid data found to create summary")
        return

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

    # Create output directory if it doesn't exist
    output_dir = "output_summary"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/employment_status_summary.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"Summary saved to: {output_file}")
    print("\nEmployment Status Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

import pandas as pd
import glob
import os
import numpy as np

def main():
    files = sorted(glob.glob('output_dev/TinyDev/TinyDev_year*.parquet'))
    dfs = [pd.read_parquet(f) for f in files]
    summary = []
    for i, df in enumerate(dfs):
        year = int(os.path.basename(files[i]).split('_')[-1].split('.')[0][4:])
        status_counts = df.groupby('employment_status').size().to_dict()

        # USER ACTION: Replace these placeholder keys with your actual status names
        continuous_active_status_key = 'Continuous Active' # Updated based on output
        new_hire_active_status_key = 'New Hire Active'    # Updated based on output

        num_continuous_active = status_counts.get(continuous_active_status_key, 0)
        num_new_hires_active = status_counts.get(new_hire_active_status_key, 0)
        
        active_count = num_continuous_active + num_new_hires_active
        
        prev_active = summary[-1]['Active'] if i > 0 else np.nan
        growth_rate = (active_count - prev_active) / prev_active if i > 0 and prev_active > 0 else np.nan
        row = {'Year': year, **status_counts, 'Active': active_count, 'ActiveGrowthRate': growth_rate}
        summary.append(row)
    summary_df = pd.DataFrame(summary).fillna('').sort_values('Year')
    summary_df.to_csv('output_dev/TinyDev/TinyDev_employment_status_summary.csv', index=False)
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()

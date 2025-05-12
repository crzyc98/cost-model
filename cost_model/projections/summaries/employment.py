"""
Employment status module: builds per-year snapshots and status summaries.

## QuickStart

To generate employment status snapshots and summaries programmatically:

```python
import pandas as pd
import matplotlib.pyplot as plt
from cost_model.projections.summaries.employment import (
    build_employment_status_snapshot,
    build_employment_status_summary
)

# Load a snapshot for a specific year
snapshot_df = pd.read_parquet('data/snapshot_2025.parquet')
event_log_df = pd.read_parquet('data/events_2025.parquet')
sim_year = 2025

# Add employment status to the snapshot
status_snapshot = build_employment_status_snapshot(snapshot_df, event_log_df, sim_year)

# Examine the distribution of employment statuses
print(status_snapshot['employment_status'].value_counts())

# Get employees with specific status
experienced_terminated = status_snapshot[status_snapshot['employment_status'] == 'Experienced Terminated']
print(f"Experienced Terminated count: {len(experienced_terminated)}")

# Generate a summary dictionary with counts by status
summary = build_employment_status_summary(snapshot_df, event_log_df, sim_year)
print(summary)

# Visualize the employment status distribution
plt.figure(figsize=(10, 6))
status_counts = [
    summary['Continuous Active'],
    summary['New Hire Active'], 
    summary['Experienced Terminated'],
    summary['New Hire Terminated']
]
status_labels = [
    'Continuous Active',
    'New Hire Active',
    'Experienced Terminated',
    'New Hire Terminated'
]
plt.bar(status_labels, status_counts)
plt.title(f'Employment Status Distribution - Year {sim_year}')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'employment_status_{sim_year}.png')
plt.show()
```

This shows how to analyze employment status distributions and create visualizations from your snapshot data.
"""

import pandas as pd
from cost_model.projections.utils import assign_employment_status

def build_employment_status_snapshot(snapshot_df: pd.DataFrame, event_log_df: pd.DataFrame, sim_year: int) -> pd.DataFrame:
    """
    Adds 'employment_status' to the snapshot based on sim_year.
    """
    df = snapshot_df.copy()
    df['employment_status'] = df.apply(lambda row: assign_employment_status(row, sim_year), axis=1)
    return df


def build_employment_status_summary(snapshot_df: pd.DataFrame, event_log_df: pd.DataFrame, sim_year: int) -> dict:
    """
    Returns a summary dict with counts for each employment status.
    """
    snap = build_employment_status_snapshot(snapshot_df, event_log_df, sim_year)
    counts = snap['employment_status'].value_counts().to_dict()
    return {
        'Year': sim_year,
        'Continuous Active': counts.get('Continuous Active', 0),
        'New Hire Active': counts.get('New Hire Active', 0),
        'Experienced Terminated': counts.get('Experienced Terminated', 0),
        'New Hire Terminated': counts.get('New Hire Terminated', 0),
        'Total Terminated': counts.get('Experienced Terminated', 0) + counts.get('New Hire Terminated', 0),
        'Active': counts.get('Continuous Active', 0) + counts.get('New Hire Active', 0)
    }
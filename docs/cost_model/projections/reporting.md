## Quick Start

To use the reporting functions directly from a Python REPL or notebook:

```python
# Import the reporting functions
from pathlib import Path
import pandas as pd
from cost_model.projections.reporting import save_detailed_results, plot_projection_results

# Load your projection results
yearly_snapshots = {...}  # Dict mapping years to DataFrames
final_snapshot = pd.read_parquet('path/to/final_snapshot.parquet')
full_event_log = pd.read_parquet('path/to/event_log.parquet')
summary_statistics = pd.read_parquet('path/to/summary.parquet')
employment_status_df = pd.read_parquet('path/to/employment_status.parquet')

# Define output location
output_dir = Path('output/my_analysis')

# Save detailed results
save_detailed_results(
    output_path=output_dir,
    scenario_name='my_scenario',
    final_snapshot=final_snapshot,
    full_event_log=full_event_log,
    summary_statistics=summary_statistics,
    employment_status_summary_df=employment_status_df,
    yearly_snapshots=yearly_snapshots
)

# Generate plots
plot_projection_results(summary_statistics, output_dir)
```

This will save all detailed results to the specified output directory and generate
plots visualizing key metrics from your projection results.
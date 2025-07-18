"""
Core summary module: wraps the core summary building.

## QuickStart

To build and work with core summaries programmatically:

```python
import pandas as pd
from pathlib import Path
from cost_model.projections.summaries.core import build_core_summary
from cost_model.projections.summaries.employment import build_employment_summary
from ..utils.columns import EMP_ID, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_TERM_DATE, ACTIVE

# Create sample snapshot data for different years
snapshots = {}

# Year 1 snapshot
snapshots[2025] = pd.DataFrame({
    EMP_ID: ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
    EMP_ROLE: ['Engineer', 'Manager', 'Engineer', 'Analyst'],
    EMP_GROSS_COMP: [75000.0, 85000.0, 65000.0, 55000.0],
    ACTIVE: [True, True, True, True],
    EMP_HIRE_DATE: [
        pd.Timestamp('2025-01-01'),
        pd.Timestamp('2025-02-01'),
        pd.Timestamp('2025-03-01'),
        pd.Timestamp('2025-04-01')
    ],
    EMP_TERM_DATE: [None, None, None, None]
}).set_index(EMP_ID)

# Year 2 snapshot (with some changes)
snapshots[2026] = pd.DataFrame({
    EMP_ID: ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
    EMP_ROLE: ['Engineer', 'Manager', 'Engineer', 'Analyst', 'Engineer'],
    EMP_GROSS_COMP: [77250.0, 87550.0, 66950.0, 56650.0, 70000.0],
    ACTIVE: [True, True, False, True, True],
    EMP_HIRE_DATE: [
        pd.Timestamp('2025-01-01'),
        pd.Timestamp('2025-02-01'),
        pd.Timestamp('2025-03-01'),
        pd.Timestamp('2025-04-01'),
        pd.Timestamp('2025-05-01')
    ],
    EMP_TERM_DATE: [None, None, pd.Timestamp('2026-07-15'), None, None]
}).set_index(EMP_ID)

# Build individual summaries
employment_summary = build_employment_summary(snapshots)

# Create a comprehensive summary dictionary
summary_dict = {
    'snapshots': snapshots,
    'employment': employment_summary,
    'years': list(snapshots.keys()),
    'metadata': {
        'start_year': min(snapshots.keys()),
        'end_year': max(snapshots.keys()),
        'run_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'version': '1.0.0'
    }
}

# Build the core summary
core_summary = build_core_summary(summary_dict)

# Analyze the summary
print(f"Projection period: {core_summary['metadata']['start_year']} to {core_summary['metadata']['end_year']}")

# Access employment summary data
for year in core_summary['years']:
    year_data = core_summary['employment'][year]
    active_count = year_data.get('active_count', 0)
    terminated_count = year_data.get('terminated_count', 0)
    new_hire_count = year_data.get('new_hire_count', 0)

    print(f"\nYear {year} Summary:")
    print(f"Active employees: {active_count}")
    print(f"Terminated employees: {terminated_count}")
    print(f"New hires: {new_hire_count}")

# Save the summary to a file
output_dir = Path('output/summaries')
output_dir.mkdir(parents=True, exist_ok=True)

# Save as JSON
import json
with open(output_dir / 'core_summary.json', 'w') as f:
    # Convert non-serializable objects to strings
    serializable_summary = {}
    for key, value in core_summary.items():
        if key == 'snapshots':
            # Skip snapshots as they're large DataFrames
            continue
        elif key == 'employment':
            # Convert employment data to serializable format
            serializable_summary[key] = {}
            for year, year_data in value.items():
                serializable_summary[key][str(year)] = year_data
        else:
            serializable_summary[key] = value

    json.dump(serializable_summary, f, indent=2)

print(f"\nSummary saved to {output_dir / 'core_summary.json'}")
```

This demonstrates how to build a comprehensive core summary from snapshot data and analyze the results.
"""


def build_core_summary(summary_dict: dict) -> dict:
    """
    Takes a core summary dict matching runner inputs and returns it.
    """
    return summary_dict

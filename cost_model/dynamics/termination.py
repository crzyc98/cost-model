# cost_model/dynamics/termination.py
"""
Functions related to simulating employee terminations.

## QuickStart

To simulate employee terminations programmatically:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from cost_model.dynamics.termination import apply_turnover

# Create a sample workforce snapshot
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
    'employee_role': ['Engineer', 'Manager', 'Engineer', 'Analyst', 'Engineer'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0, 55000.0, 72000.0],
    'active': [True, True, True, True, True],
    'employee_hire_date': [
        pd.Timestamp('2023-06-01'),  # 2 years of service
        pd.Timestamp('2022-03-15'),  # 3 years of service
        pd.Timestamp('2024-01-10'),  # 1.5 years of service
        pd.Timestamp('2024-11-05'),  # 0.5 years of service
        pd.Timestamp('2025-01-15')   # 0.25 years of service (new hire)
    ],
    'employee_termination_date': [None, None, None, None, None],
    'tenure_band': ['1-3', '3-5', '1-3', '0-1', '0-1']
}).set_index('employee_id')

# Set simulation period
start_date = pd.Timestamp('2025-01-01')
end_date = pd.Timestamp('2025-12-31')

# Method 1: Apply a flat termination rate to all employees
flat_rate = 0.15  # 15% annual termination rate

updated_snapshot_flat, term_events_flat = apply_turnover(
    df=snapshot.copy(),
    hire_col='employee_hire_date',
    probs_or_rate=flat_rate,
    start_date=start_date,
    end_date=end_date,
    term_date_col='employee_termination_date',
    active_col='active',
    new_hire_term_rate=0.25  # Higher termination rate for new hires
)

print(f"Flat rate termination results:")
print(f"  Initial employees: {len(snapshot)}")
print(f"  Remaining active: {updated_snapshot_flat['active'].sum()}")
print(f"  Terminated: {len(updated_snapshot_flat) - updated_snapshot_flat['active'].sum()}")
print(f"  Generated {len(term_events_flat)} termination events")

# Method 2: Apply different termination rates based on tenure
# Define termination probabilities per employee
tenure_based_probs = pd.Series([
    0.10,  # EMP001: 10% probability (1-3 years tenure)
    0.05,  # EMP002: 5% probability (3-5 years tenure)
    0.10,  # EMP003: 10% probability (1-3 years tenure)
    0.20,  # EMP004: 20% probability (0-1 years tenure)
    0.25   # EMP005: 25% probability (0-1 years tenure, new hire)
], index=snapshot.index)

updated_snapshot_tenure, term_events_tenure = apply_turnover(
    df=snapshot.copy(),
    hire_col='employee_hire_date',
    probs_or_rate=tenure_based_probs,
    start_date=start_date,
    end_date=end_date,
    term_date_col='employee_termination_date',
    active_col='active'
)

print(f"\nTenure-based termination results:")
print(f"  Initial employees: {len(snapshot)}")
print(f"  Remaining active: {updated_snapshot_tenure['active'].sum()}")
print(f"  Terminated: {len(updated_snapshot_tenure) - updated_snapshot_tenure['active'].sum()}")
print(f"  Generated {len(term_events_tenure)} termination events")

# Method 3: Apply role-based termination rates
# First create a mapping of roles to termination rates
role_term_rates = {
    'Engineer': 0.12,
    'Manager': 0.08,
    'Analyst': 0.15
}

# Then map these rates to each employee based on their role
role_based_probs = pd.Series(
    [role_term_rates[role] for role in snapshot['employee_role']],
    index=snapshot.index
)

updated_snapshot_role, term_events_role = apply_turnover(
    df=snapshot.copy(),
    hire_col='employee_hire_date',
    probs_or_rate=role_based_probs,
    start_date=start_date,
    end_date=end_date,
    term_date_col='employee_termination_date',
    active_col='active'
)

print(f"\nRole-based termination results:")
print(f"  Initial employees: {len(snapshot)}")
print(f"  Remaining active: {updated_snapshot_role['active'].sum()}")
print(f"  Terminated: {len(updated_snapshot_role) - updated_snapshot_role['active'].sum()}")
print(f"  Generated {len(term_events_role)} termination events")

# Examine termination dates
if len(term_events_role) > 0:
    print("\nTermination events:")
    for _, event in term_events_role.iterrows():
        emp_id = event['employee_id']
        term_date = updated_snapshot_role.loc[emp_id, 'employee_termination_date']
        print(f"  {emp_id} terminated on {term_date.strftime('%Y-%m-%d')}")

# Save the results
output_dir = Path('output/terminations')
output_dir.mkdir(parents=True, exist_ok=True)
updated_snapshot_role.to_parquet(output_dir / 'snapshot_with_terminations.parquet')
term_events_role.to_parquet(output_dir / 'termination_events.parquet')
```

This demonstrates how to simulate employee terminations using flat rates, tenure-based rates, and role-based rates.
"""

import logging
import pandas as pd
import numpy as np
from typing import Union, Optional

# Use relative imports for other modules within the dynamics package if needed
try:
    from .sampling.terminations import sample_terminations  # Assumes this exists
    from .sampling.salary import (
        SalarySampler,
        DefaultSalarySampler,
    )  # If needed for salary sampling
except ImportError:
    print("Warning (termination.py): Could not import sampling helpers.")

    # Define dummy functions/classes if needed
    def sample_terminations(df, *args, **kwargs):
        return df  # Passthrough

    class DefaultSalarySampler:
        def sample_terminations(self, *args, **kwargs):
            return None  # Return None if sampling fails


# Use absolute imports for modules outside dynamics
try:
    from cost_model.utils.columns import EMP_TERM_DATE, EMP_HIRE_DATE, EMP_GROSS_COMP
except ImportError:
    print("Warning (termination.py): Could not import column constants.")
    EMP_TERM_DATE, EMP_HIRE_DATE, EMP_GROSS_COMP = (
        "employee_termination_date",
        "employee_hire_date",
        "employee_gross_compensation",
    )


logger = logging.getLogger(__name__)


# Renamed from _apply_turnover
def apply_turnover(
    df: pd.DataFrame,
    hire_col: str,
    probs_or_rate: Union[float, pd.Series, np.ndarray],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rng: np.random.Generator,
    # Optional: sampler and prev_term_salaries if sampling term comp
    sampler: Optional[SalarySampler] = None,
    prev_term_salaries: Optional[pd.Series] = None,
    log: Optional[logging.Logger] = None,  # Accept logger instance
) -> pd.DataFrame:
    """
    Applies terminations to a DataFrame based on probabilities or a flat rate.
    Updates the EMP_TERM_DATE column. Optionally samples termination compensation.

    Args:
        df: DataFrame with employee data.
        hire_col: Name of the hire date column.
        probs_or_rate: A float rate, or a Series/ndarray of probabilities per employee.
        start_date: The start date of the period during which terminations occur.
        end_date: The end date of the period during which terminations occur.
        rng: NumPy random number generator.
        sampler: Optional SalarySampler for termination compensation.
        prev_term_salaries: Optional Series of previous termination salaries for sampling base.
        log: Optional logger instance.

    Returns:
        DataFrame with termination dates potentially updated.
    """
    local_log = log or logger  # Use passed logger or module logger

    # Call the underlying sampling function (which should handle masking etc.)
    # sample_terminations should return df with EMP_TERM_DATE updated
    local_log.debug(
        f"Applying turnover for period {start_date.date()} to {end_date.date()}..."
    )
    df_with_terms = sample_terminations(df, hire_col, probs_or_rate, end_date, rng)

    # --- Optional: Sample Termination Compensation ---
    # Check if sampler and previous salaries are provided
    if sampler and prev_term_salaries is not None and not prev_term_salaries.empty:
        term_mask = df_with_terms[EMP_TERM_DATE].between(
            start_date, end_date, inclusive="both"
        )
        # Also check if the term date was newly assigned in this step (optional, depends on sample_terminations logic)
        # Example: term_mask = term_mask & df[EMP_TERM_DATE].isna() # Only sample for newly termed

        n_term_in_period = term_mask.sum()
        if n_term_in_period > 0:
            local_log.debug(
                f"Sampling termination compensation for {n_term_in_period} employees."
            )
            try:
                draws = sampler.sample_terminations(
                    prev_term_salaries, size=n_term_in_period, rng=rng
                )
                # Assign sampled compensation
                # Ensure index alignment if draws is Series
                df_with_terms.loc[term_mask, EMP_GROSS_COMP] = (
                    draws if isinstance(draws, np.ndarray) else draws.values
                )
            except Exception:
                local_log.exception("Error during termination compensation sampling.")
        else:
            local_log.debug(
                "No terminations within the specified period to sample compensation for."
            )
    elif sampler and (prev_term_salaries is None or prev_term_salaries.empty):
        local_log.warning(
            "Termination sampler provided, but prev_term_salaries is missing or empty. Cannot sample termination compensation."
        )

    return df_with_terms

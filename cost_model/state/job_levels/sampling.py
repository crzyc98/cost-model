from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from .models import JobLevel
from .init import get_level_by_id
from .transitions import PROMOTION_MATRIX
from cost_model.state.schema import EMP_LEVEL


def sample_new_hire_compensation(
    level: JobLevel,
    age: int,
    random_state: Optional[np.random.RandomState] = None
) -> float:
    """Sample compensation for a new hire at a given level."""
    rng = random_state or np.random
    base_comp = level.comp_base_salary * (1 + level.comp_age_factor * age)
    variation = rng.normal(0, level.comp_stochastic_std_dev)
    compensation = max(
        level.min_compensation,
        min(level.max_compensation, base_comp * (1 + variation))
    )
    return compensation


def sample_new_hires_vectorized(
    level: JobLevel,
    ages: np.ndarray,
    random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Vectorized sampling of compensation for multiple new hires at the same level."""
    rng = random_state or np.random
    base_comp = level.comp_base_salary * (1 + level.comp_age_factor * ages)
    variations = rng.normal(0, level.comp_stochastic_std_dev, size=len(ages))
    raw_comp = base_comp * (1 + variations)
    return np.clip(raw_comp, level.min_compensation, level.max_compensation)


def sample_mixed_new_hires(
    level_counts: Dict[int, int],
    age_range: Tuple[int, int] = (25, 55),
    random_state: Optional[np.random.RandomState] = None
) -> pd.DataFrame:
    """Sample compensation for new hires across multiple levels."""
    rng = random_state or np.random.RandomState()
    results = []
    for level_id, count in level_counts.items():
        level = get_level_by_id(level_id)
        if level is None:
            raise ValueError(f"Invalid level_id: {level_id}")
        ages = rng.uniform(age_range[0], age_range[1], size=count)
        compensation = sample_new_hires_vectorized(level, ages, random_state=rng)
        df = pd.DataFrame({
            'level_id': level_id,
            'age': ages,
            'compensation': compensation
        })
        results.append(df)
    if not results:
        return pd.DataFrame(columns=['level_id', 'age', 'compensation'])
    return pd.concat(results, ignore_index=True)


# --- New: Markov-chain promotion engine ---
import numpy as np

def apply_promotion_markov(
    df: pd.DataFrame,
    level_col: str = EMP_LEVEL,
    matrix: pd.DataFrame = PROMOTION_MATRIX,
    rng: Optional[np.random.RandomState] = None,
    term_date_col: str = 'employee_termination_date',
    simulation_year: Optional[int] = None
) -> pd.DataFrame:
    """
    For each employee, sample next-level (or exit) according to the Markov matrix.
    Returns a new DataFrame where:
      - level_col is updated to the new level (0-4), or NaN for exit
      - a new column 'exited' == True if they left
      - term_date_col is set to a date within the simulation year if exited
    """
    rng = rng or np.random
    # Create a copy of the input DataFrame to avoid modifying the original
    df = df.copy()
    
    # Store the original index to ensure alignment
    original_index = df.index
    
    # Reset index to ensure we have a clean 0-based index for array operations
    df = df.reset_index(drop=True)
    
    states = list(matrix.columns)
    prob_matrix = matrix.values
    idx_map = {lvl: i for i, lvl in enumerate(matrix.index)}
    
    # Initialize arrays with the same length as the dataframe
    new_levels = [np.nan] * len(df)
    exited = [False] * len(df)
    
    # Generate a random day within the simulation year for each employee
    if simulation_year is None:
        # Try to get the year from the DataFrame if not provided
        if 'simulation_year' in df.columns:
            simulation_year = df['simulation_year'].iloc[0]
        else:
            # Default to current year if we can't determine it
            simulation_year = pd.Timestamp.now().year
    
    # Generate random termination dates for all employees first (we'll only use them for those who exit)
    start_date = pd.Timestamp(f"{simulation_year}-01-01")
    end_date = pd.Timestamp(f"{simulation_year}-12-31")
    days_in_year = (end_date - start_date).days + 1
    random_days = rng.integers(0, days_in_year, size=len(df))
    term_dates = [start_date + pd.Timedelta(days=int(days)) for days in random_days]
    
    for idx, (_, row) in enumerate(df.iterrows()):
        curr = row[level_col]
        if pd.isna(curr):
            new_levels[idx] = np.nan
            exited[idx] = True
            continue
            
        row_i = idx_map[curr]
        probs = prob_matrix[row_i]
        choice = rng.choice(states, p=probs)
        
        if choice == 'exit':
            new_levels[idx] = np.nan
            exited[idx] = True
            # Set the termination date for this employee
            df.at[idx, term_date_col] = term_dates[idx]
        else:
            new_levels[idx] = int(choice)
            exited[idx] = False
    
    # Update the DataFrame with new levels and exited status
    df[level_col] = new_levels
    df['exited'] = exited
    
    # Restore the original index before returning
    df.index = original_index
    return df

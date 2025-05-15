from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from .models import JobLevel
from .init import get_level_by_id
from .transitions import PROMOTION_MATRIX
from cost_model.utils.columns import EMP_LEVEL


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
    rng: Optional[np.random.RandomState] = None
) -> pd.DataFrame:
    """
    For each employee, sample next-level (or exit) according to the Markov matrix.
    Returns a new DataFrame where:
      - level_col is updated to the new level (0-4), or NaN for exit
      - a new column 'exited' == True if they left
    """
    rng = rng or np.random
    df = df.copy()
    states = list(matrix.columns)
    prob_matrix = matrix.values
    idx_map = {lvl: i for i, lvl in enumerate(matrix.index)}
    new_levels = []
    exited = []
    for curr in df[level_col]:
        if pd.isna(curr):
            new_levels.append(np.nan)
            exited.append(True)
            continue
        row_i = idx_map[curr]
        probs = prob_matrix[row_i]
        choice = rng.choice(states, p=probs)
        if choice == 'exit':
            new_levels.append(np.nan)
            exited.append(True)
        else:
            new_levels.append(int(choice))
            exited.append(False)
    df[level_col] = new_levels
    df['exited'] = exited
    return df

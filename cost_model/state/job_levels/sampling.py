import logging
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import yaml
import os

# Centralized warnings_errors logger for all error reporting
logger = logging.getLogger("warnings_errors")

from .models import JobLevel
from .init import get_level_by_id
from .transitions import PROMOTION_MATRIX
from cost_model.state.schema import EMP_LEVEL, EMP_ID, EMP_HIRE_DATE, EMP_EXITED, EMP_LEVEL_SOURCE


def validate_promotion_matrix(matrix: pd.DataFrame) -> None:
    """
    Validates that the promotion matrix is properly formatted.
    
    Args:
        matrix: DataFrame representing the Markov transition matrix
        
    Raises:
        ValueError: If the matrix is None, empty, or invalid
    """
    if matrix is None:
        raise ValueError("Promotion transition matrix cannot be None")
    
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError(f"Expected promotion matrix to be a pandas DataFrame, got {type(matrix).__name__}")
    
    if matrix.empty:
        raise ValueError("Promotion transition matrix is empty")
    
    # Check that all values are between 0 and 1
    if (matrix < 0).any().any() or (matrix > 1).any().any():
        # Find the first invalid value to provide a helpful error message
        for i, row in matrix.iterrows():
            for col in matrix.columns:
                val = row[col]
                if not (0 <= val <= 1):
                    raise ValueError(
                        f"Invalid transition probability {val:.4f} at level={i}, next_state={col}. "
                        "All probabilities must be between 0 and 1."
                    )
    
    # Check that rows sum to approximately 1.0
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            f"Invalid transition matrix: Rows must sum to 1.0. "
            f"Row sums: {row_sums.to_dict()}"
        )


def load_markov_matrix(path: Optional[str] = None, allow_default: bool = False) -> pd.DataFrame:
    """
    Load a Markov transition matrix from YAML (path) or use the canonical PROMOTION_MATRIX if allowed.

    The YAML file should have the following structure:
    ```yaml
    states: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 'exit']
    transitions:
      0: [0.85, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      # ... other levels ...
      exit: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ```

    Args:
      path: Path to the YAML file. If None and allow_default=True, returns PROMOTION_MATRIX.
      allow_default: Whether to fall back to PROMOTION_MATRIX when path is None.

    Returns:
      A square DataFrame of transition probabilities (rows sum to 1).

    Raises:
      FileNotFoundError: If path is provided but file is missing.
      ValueError: If path is None and allow_default=False, or if the loaded matrix is invalid.
    """
    # 1) Handle default case if no path provided
    if path is None:
        if allow_default:
            logger.info("No promotion_matrix_path provided; using full default PROMOTION_MATRIX")
            df = PROMOTION_MATRIX.copy()
            validate_promotion_matrix(df)
            return df.astype(float)
        raise ValueError(
            "Promotion transition matrix path must be specified or enable dev_mode to use default."
        )

    # 2) Load from YAML file
    if not os.path.exists(path):
        raise FileNotFoundError(f"Promotion matrix file not found: {path}")

    logger.info(f"Loading promotion matrix from: {path}")
    
    try:
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f)
            
        # Expect YAML with 'states' (list) and 'transitions' (dict)
        if 'states' not in yaml_data or 'transitions' not in yaml_data:
            raise ValueError(
                "YAML must contain 'states' (list) and 'transitions' (dict) keys. "
                "See function docstring for expected format."
            )
            
        states = yaml_data['states']
        transitions = yaml_data['transitions']
        
        # Ensure transitions keys are all strings for robust lookup
        transitions_str_keys = {str(k): v for k, v in transitions.items()}
        
        # Build matrix data as a list of lists, ordered by states
        matrix_data = []
        for state in states:
            key = str(state)
            if key not in transitions_str_keys:
                raise ValueError(f"Missing transition row for state '{state}' in promotion matrix YAML.")
                
            row = transitions_str_keys[key]
            if len(row) != len(states):
                raise ValueError(
                    f"Transition row for state '{state}' has length {len(row)} but expected {len(states)}. "
                    f"Each row must have exactly one probability per state."
                )
            matrix_data.append(row)
            
        # Construct DataFrame with explicit row and column order
        df = pd.DataFrame(matrix_data, index=states, columns=states, dtype=float)
        
        # Log successful load
        logger.info(f"Successfully loaded promotion matrix with states: {states}")
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {path}: {str(e)}")
        raise ValueError(f"Invalid YAML in promotion matrix file: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading promotion matrix from {path}: {str(e)}")
        raise ValueError(f"Invalid promotion matrix: {str(e)}")
    
    # Validate the matrix structure and probabilities
    validate_promotion_matrix(df)
    
    return df


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


# --- Markov-chain promotion engine ---


def apply_promotion_markov(
    df: pd.DataFrame,
    level_col: str = EMP_LEVEL,
    matrix: pd.DataFrame = PROMOTION_MATRIX,
    rng: Optional[np.random.RandomState] = None,
    term_date_col: str = 'employee_termination_date',
    simulation_year: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    For each employee, sample next-level (or exit) according to the Markov matrix.
    
    Args:
        df: Input DataFrame containing employee data
        level_col: Column name containing the current job level
        matrix: Markov transition matrix as a DataFrame
        rng: Random number generator instance
        term_date_col: Column name for termination date (if employee exits)
        simulation_year: Current simulation year
        logger: Optional logger instance for debug/info messages
        
    Returns:
        DataFrame with updated levels and exit status
        
    Raises:
        ValueError: If the promotion matrix is invalid or missing required columns
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Validate the promotion matrix before proceeding
    try:
        validate_promotion_matrix(matrix)
    except ValueError as e:
        logger.error("Invalid promotion matrix: %s", str(e))
        raise
    
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
    
    # Log matrix info for debugging
    logger.debug("Using promotion matrix with states: %s", states)
    logger.debug("Matrix shape: %s, Index: %s", prob_matrix.shape, matrix.index.tolist())
    logger.debug("Matrix columns: %s", matrix.columns.tolist())
    
    # Validate that all current levels in the data exist in the matrix
    unique_levels = df[level_col].dropna().unique()
    missing_levels = set(unique_levels) - set(matrix.index)
    if missing_levels:
        error_msg = f"Levels {missing_levels} exist in data but not in the promotion matrix"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # CHANGE 1: Initialize new_levels with CURRENT levels, not np.nan
    # This ensures we have a fallback value for every employee
    new_levels = df[level_col].tolist()  # Start with current levels as default
    exited = [False] * len(df)
    
    # Log the number of employees being processed
    logger.debug("Processing %d employees for promotion/termination", len(df))
    
    if simulation_year is not None:
        # Generate a random day in the simulation year for terminations
        start_date = pd.Timestamp(f"{simulation_year}-01-01")
        end_date = pd.Timestamp(f"{simulation_year}-12-31")
        days_in_year = (end_date - start_date).days + 1
        # numpy.random.Generator uses .integers(), older RandomState uses .randint()
        if hasattr(rng, "integers"):
            random_days = rng.integers(0, days_in_year, size=len(df))
        else:
            random_days = rng.randint(0, days_in_year, size=len(df))
        term_dates = [start_date + pd.Timedelta(days=int(d)) for d in random_days]
    
    # For each employee, sample their next state
    for i, (_, row) in enumerate(df.iterrows()):
        current_level = row[level_col]
        
        # CHANGE 2: Handle missing current levels explicitly
        if pd.isna(current_level):
            # CHANGE 3: Log this as a warning, not just debug, and provide a default level
            emp_id = row.get(EMP_ID, f"at index {i}")
            logger.warning(
                f"Employee {emp_id} has no current level in apply_promotion_markov. "
                f"This shouldn't happen for active employees. "
                f"Assigning default level 1."
            )
            # CHANGE 4: Assign a default level rather than skipping
            new_levels[i] = 1  # Default level for employees with missing levels
            continue
            
        # Get the row index in the transition matrix
        row_idx = idx_map.get(current_level)
        if row_idx is None:
            # If level not in matrix, keep them at the same level
            logger.warning(
                "Employee at index %d has level %s which is not in the promotion matrix. "
                "Keeping at current level.", i, current_level
            )
            new_levels[i] = current_level
            continue
            
        # Get transition probabilities for the current level
        probs = prob_matrix[row_idx]
        
        # Log transition probabilities for debugging
        logger.debug("Employee %d (level %s) transition probs: %s", 
                    i, current_level, 
                    dict(zip(states, [f"{p:.2f}" for p in probs])))
        
        try:
            # Sample the next state based on transition probabilities
            next_state_idx = rng.choice(len(states), p=probs)
            next_state = states[next_state_idx]
            
            logger.debug("Employee %d (level %s) transitioning to: %s", 
                        i, current_level, next_state)
            
            if next_state == 'exit':
                # Employee is terminated
                exited[i] = True
                if simulation_year is not None:
                    # Set termination date to a random day in the simulation year
                    term_date = start_date + pd.Timedelta(days=int(random_days[i]))
                    df.at[i, term_date_col] = term_date
                    logger.debug("Employee %d terminated on %s", i, term_date)
            else:
                # Employee stays or gets promoted
                new_levels[i] = next_state
                if next_state > current_level:
                    logger.debug("Employee %d promoted from level %s to %s", 
                                i, current_level, next_state)
                elif next_state < current_level:
                    logger.warning("Employee %d demoted from level %s to %s", 
                                 i, current_level, next_state)
        except ValueError as e:
            logger.error(
                "Error sampling next state for employee %d (level %s). "
                "Probabilities: %s. Error: %s", 
                i, current_level, probs, str(e)
            )
            # In case of error, keep employee at current level
            new_levels[i] = current_level
    
    # Update the DataFrame
    df[level_col] = new_levels
    df['exited'] = exited
    
    # CHANGE 5: Add additional diagnostic check after assignment
    if pd.isna(df[level_col]).any():
        na_count = pd.isna(df[level_col]).sum()
        na_indices = df[pd.isna(df[level_col])].index.tolist()
        logger.warning(
            f"CRITICAL: After processing all employees, {na_count} still have NaN levels. "
            f"Problem indices: {na_indices[:5]}{'...' if len(na_indices) > 5 else ''}. "
            f"Filling with default level 1."
        )
        # Final safety net
        df[level_col] = df[level_col].fillna(1)
    
    # Log summary statistics
    num_exits = sum(exited)
    num_promotions = sum(1 for new, old in zip(new_levels, df[level_col]) 
                        if not pd.isna(new) and not pd.isna(old) and new > old)
    num_demotions = sum(1 for new, old in zip(new_levels, df[level_col]) 
                       if not pd.isna(new) and not pd.isna(old) and new < old)
    
    logger.info(
        "Promotion/termination summary - Total: %d, Promotions: %d, "
        "Demotions: %d, Exits: %d (%.1f%%)",
        len(df), num_promotions, num_demotions, num_exits, 
        (num_exits / len(df) * 100) if len(df) > 0 else 0
    )
    
    # Restore the original index
    df.index = original_index
    
    # CRITICAL DIAGNOSIS: Check if any NaNs are present in level_col IMMEDIATELY before return
    if pd.isna(df[level_col]).any():
        na_count = pd.isna(df[level_col]).sum()
        na_indices = df.index[pd.isna(df[level_col])].tolist()
        
        # Get employee IDs for better debugging if available
        if EMP_ID in df.columns:
            emp_ids = df.loc[pd.isna(df[level_col]), EMP_ID].tolist()
            logger.warning(
                f"CRITICAL DIAGNOSIS: {na_count} NaN values in {level_col} detected at return from apply_promotion_markov. "
                f"Employee IDs: {emp_ids[:5]}{'...' if len(emp_ids) > 5 else ''}. "
                f"These employees lost their level values during processing."
            )
            
            # Get more information about employees with NaN levels
            for emp_id in emp_ids[:5]:  # Limit to first 5 for brevity
                emp_row = df[df[EMP_ID] == emp_id]
                if not emp_row.empty:
                    # Log information about employees with NaN levels
                    logger.warning(
                        f"Employee {emp_id} details: "
                        f"exited={emp_row['exited'].values[0]}, "
                        f"hire_date={emp_row[EMP_HIRE_DATE].dt.strftime('%Y-%m-%d').values[0] if EMP_HIRE_DATE in emp_row.columns else 'N/A'}"
                    )
        else:
            # Just log indices if no employee IDs available
            logger.warning(
                f"CRITICAL DIAGNOSIS: {na_count} NaN values in {level_col} detected at return from apply_promotion_markov. "
                f"Indices: {na_indices[:5]}{'...' if len(na_indices) > 5 else ''}."
            )
        
        # Let's fix this immediately rather than letting it propagate
        logger.warning(f"Applying emergency fix: filling NaN values in {level_col} with default level 1")
        df[level_col] = df[level_col].fillna(1)
    
    return df

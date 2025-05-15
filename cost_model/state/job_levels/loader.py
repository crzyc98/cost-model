from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml
import logging

from .models import JobLevel, ConfigError
from .intervals import check_for_overlapping_bands as _check_for_overlapping_bands

logger = logging.getLogger(__name__)

def load_job_levels_from_config(config: Dict[str, Any], strict_validation: bool = True) -> Dict[int, JobLevel]:
    """Load job levels from a configuration dictionary."""
    levels: Dict[int, JobLevel] = {}
    for data in config.get("job_levels", []):
        level_id = data.get("level_id")
        if level_id is None:
            msg = "Job level missing 'level_id'"
            if strict_validation:
                raise ConfigError(msg)
            else:
                logger.warning(msg)
                continue
        try:
            level = JobLevel(
                level_id=int(level_id),
                name=data.get("name", f"Level {level_id}"),
                description=data.get("description", ""),
                min_compensation=float(data.get("min_compensation", 0)),
                max_compensation=float(data.get("max_compensation", 0)),
                comp_base_salary=float(data.get("comp_base_salary", 0)),
                comp_age_factor=float(data.get("comp_age_factor", 0)),
                comp_stochastic_std_dev=float(data.get("comp_stochastic_std_dev", 0.1)),
                mid_compensation=data.get("mid_compensation"),
                avg_annual_merit_increase=float(data.get("avg_annual_merit_increase", 0.03)),
                promotion_probability=float(data.get("promotion_probability", 0.1)),
                target_bonus_percent=float(data.get("target_bonus_percent", 0.0)),
                source=data.get("source", "band"),
                job_families=data.get("job_families", [])
            )
            levels[level.level_id] = level
        except (TypeError, ValueError) as e:
            msg = f"Invalid data for level {level_id}: {e}"
            if strict_validation:
                raise ConfigError(msg)
            else:
                logger.warning(msg)
    if not levels:
        raise ConfigError("No job levels found in configuration")
    # Validate overlaps
    _check_for_overlapping_bands(levels, strict=strict_validation)
    return levels

def load_from_yaml(path: Union[str, Path], strict_validation: bool = True) -> Dict[int, JobLevel]:
    """Load job levels from a YAML file."""
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return load_job_levels_from_config(config, strict_validation=strict_validation)
    except (yaml.YAMLError, OSError) as e:
        msg = f"Error loading job levels from {path}: {e}"
        raise ConfigError(msg)

# ---------------------- DataFrame ingestion with imputation ----------------------

def ingest_with_imputation(
    raw_df: 'pd.DataFrame', comp_col: str = 'employee_gross_compensation'
) -> 'pd.DataFrame':
    """
    Assign static bands, then impute missing levels by compensation percentile.
    Returns DataFrame with column 'job_level' fully populated.
    """
    import pandas as pd
    from .assign import assign_levels_to_dataframe
    from .engine import infer_job_level_by_percentile

    # assign using static bands
    df = assign_levels_to_dataframe(raw_df, comp_col)
    # percentile-based imputation
    df = infer_job_level_by_percentile(df, comp_col)
    # fill into job_level column
    df['job_level'] = df.get('level_id').fillna(df['imputed_level']).astype('Int64')
    return df.drop(columns=['imputed_level'])

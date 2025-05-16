"""
Initializes the projection engine with validated parameters and configurations.
"""

from typing import Dict, Any, List, Tuple
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from cost_model.utils.columns import (
    EMP_ID, 
    SIMULATION_YEAR,
    EVENT_TYPE,
    EVT_HIRE,
    EVT_TERM,
    EVT_CONTRIB,
)
from cost_model.state.schema import SNAPSHOT_DTYPES
from cost_model.config.loaders import load_yaml_config
from cost_model.config.params import parse_config

logger = logging.getLogger(__name__)


def initialize(config_ns: Any, 
                initial_snapshot: pd.DataFrame, 
                initial_log: pd.DataFrame) -> Tuple[Dict, Dict, np.random.Generator, List[int], str, List[str]]:
    """
    Initializes the projection engine with validated parameters and configurations.
    
    Args:
        config_ns: Parsed configuration namespace (can be dict or SimpleNamespace)
        initial_snapshot: Initial employee snapshot DataFrame
        initial_log: Initial event log DataFrame
        
    Returns:
        Tuple containing:
        - global_params: Parsed global parameters
        - plan_rules: Plan rules configuration
        - rng: Random number generator
        - years: List of simulation years
        - census_path: Path to census template
        - ee_contrib_event_types: List of employee contribution event types
    """
    # 1. Parse config into global_params and plan_rules
    if isinstance(config_ns, dict):
        config = load_yaml_config(config_ns["config"])
    else:
        config = load_yaml_config(getattr(config_ns, "config", "config/dev_tiny.yaml"))
    global_params, plan_rules = parse_config(config)
    
    # 2. Initialize random number generator
    rng = np.random.default_rng(global_params.seed)
    
    # 3. Validate and normalize initial_snapshot
    if not isinstance(initial_snapshot, pd.DataFrame):
        raise ValueError("initial_snapshot must be a pandas DataFrame")
    
    # Ensure snapshot has required columns and types
    missing_cols = set(SNAPSHOT_DTYPES.keys()) - set(initial_snapshot.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in snapshot: {missing_cols}")
    
    # Ensure snapshot has unique EMP_IDs
    if not initial_snapshot[EMP_ID].is_unique:
        raise ValueError("initial_snapshot must have unique EMP_IDs")
    
    # 4. Build list of projection years
    start_year = getattr(config_ns, "start_year", datetime.now().year)
    num_years = getattr(config_ns, "num_years", 5)
    years = list(range(start_year, start_year + num_years))
    
    # 5. Derive census template path
    census_path = getattr(config_ns, "census", "data/census_preprocessed.parquet")
    
    # 6. Get employee contribution event types
    ee_contrib_event_types = [EVT_CONTRIB]
    
    logger.info(f"Initialized projection engine for years: {years}")
    logger.debug(f"Global parameters: {global_params}")
    logger.debug(f"Plan rules: {plan_rules}")
    
    return global_params, plan_rules, rng, years, census_path, ee_contrib_event_types

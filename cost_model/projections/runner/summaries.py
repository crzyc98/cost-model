"""
Handles computation of various metrics and summaries for the projection.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
from datetime import datetime


def make_yearly_summaries(snapshot: pd.DataFrame, 
                         year_events: pd.DataFrame, 
                         year: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute yearly summaries for core metrics and employment status.
    
    Args:
        snapshot: Current year's snapshot
        year_events: Events that occurred during the year
        year: Current year
        
    Returns:
        Tuple containing:
        - core_summary: Dictionary of core metrics
        - employment_summary: Dictionary of employment status metrics
    """
    # Core metrics
    core_summary = {
        "year": year,
        "headcount": len(snapshot),
        "active_headcount": len(snapshot[snapshot["active"]]),
        "total_contributions": snapshot[EMP_CONTR].sum(),
        "employer_contributions": snapshot["employer_core_contribution"].sum() + 
                                snapshot["employer_match_contribution"].sum(),
        "avg_deferral_rate": snapshot[EMP_DEFERRAL_RATE].mean(),
        "participation_rate": (len(snapshot[snapshot[EMP_DEFERRAL_RATE] > 0]) / 
                            len(snapshot[snapshot["active"]])) * 100
    }
    
    # Employment status metrics
    employment_summary = {
        "year": year,
        "hires": len(year_events[year_events[EVENT_TYPE] == EVT_HIRE]),
        "terminations": len(year_events[year_events[EVENT_TYPE] == EVT_TERM]),
        "new_hires": len(year_events[year_events[EVENT_TYPE] == EVT_CONTRIB]),
        "by_level": snapshot[EMP_LEVEL].value_counts().to_dict(),
        "by_tenure_band": snapshot["tenure_band"].value_counts().to_dict()
    }
    
    return core_summary, employment_summary

"""
Utility functions for standardizing and managing employee tenure bands.
"""

import pandas as pd
from typing import Optional, Union, List, Dict

def standardize_tenure_band(tenure_band: Optional[Union[str, float, int]]) -> Optional[str]:
    """
    Standardize tenure band formats to ensure consistent representation across the codebase.
    
    Args:
        tenure_band: Tenure band string in various formats, or numeric tenure value
        
    Returns:
        Standardized tenure band string in format: '0-1', '1-3', '3-5', '5+'
        Returns pd.NA if input is NA/None
        
    Examples:
        >>> standardize_tenure_band('<1')
        '0-1'
        >>> standardize_tenure_band('0-1yr')
        '0-1'
        >>> standardize_tenure_band('1-3')  # Already standardized
        '1-3'
        >>> standardize_tenure_band(0.5)    # Numeric input
        '0-1'
    """
    if pd.isna(tenure_band):
        return pd.NA
    
    # Handle numeric inputs by converting to standardized string format
    if isinstance(tenure_band, (int, float)):
        if tenure_band < 1:
            return "0-1"
        elif tenure_band < 3:
            return "1-3"
        elif tenure_band < 5:
            return "3-5"
        else:
            return "5+"
    
    # Handle string formats
    if not isinstance(tenure_band, str):
        return pd.NA
    
    # Map variations to standard format
    mapping = {
        # For tenure bands with '<1' or similar notation
        '<1': '0-1',
        '0-1yr': '0-1',
        '0-1 yr': '0-1',
        '0-1 yrs': '0-1',
        '0-1years': '0-1',
        '0-1 years': '0-1',
        
        # For tenure bands with '1-3' or similar notation
        '1-3yr': '1-3',
        '1-3 yr': '1-3',
        '1-3 yrs': '1-3',
        '1-3years': '1-3',
        '1-3 years': '1-3',
        
        # For tenure bands with '3-5' or similar notation
        '3-5yr': '3-5',
        '3-5 yr': '3-5',
        '3-5 yrs': '3-5',
        '3-5years': '3-5',
        '3-5 years': '3-5',
        
        # For tenure bands with '5+' or similar notation
        '5+ yr': '5+',
        '5+ yrs': '5+',
        '5+yr': '5+',
        '5+years': '5+',
        '5+ years': '5+',
        '>5': '5+'
    }
    
    # Return the standardized version if found in mapping, otherwise return the original
    # if it's already in standard format ('0-1', '1-3', '3-5', '5+')
    if tenure_band in mapping:
        return mapping[tenure_band]
    elif tenure_band in ('0-1', '1-3', '3-5', '5+'):
        return tenure_band
    else:
        # Log a warning about unexpected format here if desired
        # For now, try to determine the correct format based on patterns
        if tenure_band.startswith('0') or tenure_band.startswith('<'):
            return '0-1'
        elif tenure_band.startswith('1'):
            return '1-3'
        elif tenure_band.startswith('3'):
            return '3-5'
        elif tenure_band.startswith('5') or tenure_band.startswith('>'):
            return '5+'
        else:
            # If we can't determine the format, return as is (could log a warning)
            return tenure_band

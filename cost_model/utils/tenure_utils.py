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
        Standardized tenure band string in format: '<1', '1-3', '3-5', '5-10', '10-15', '15+'
        Returns pd.NA if input is NA/None

    Examples:
        >>> standardize_tenure_band('0-1')
        '<1'
        >>> standardize_tenure_band('0-1yr')
        '<1'
        >>> standardize_tenure_band('1-3')  # Already standardized
        '1-3'
        >>> standardize_tenure_band('5+')   # Legacy format
        '15+'
        >>> standardize_tenure_band(0.5)    # Numeric input
        '<1'
        >>> standardize_tenure_band(12.0)   # Numeric input
        '10-15'
        >>> standardize_tenure_band(18.0)   # Numeric input
        '15+'
    """
    if pd.isna(tenure_band):
        return pd.NA

    # Handle numeric inputs by converting to standardized string format
    if isinstance(tenure_band, (int, float)):
        if tenure_band < 1:
            return "<1"
        elif tenure_band < 3:
            return "1-3"
        elif tenure_band < 5:
            return "3-5"
        elif tenure_band < 10:
            return "5-10"
        elif tenure_band < 15:
            return "10-15"
        else:
            return "15+"

    # Handle string formats
    if not isinstance(tenure_band, str):
        return pd.NA

    # Normalize spacing and case for matching
    tb_norm = str(tenure_band).strip().casefold()

    # Map variations to standard format
    mapping = {
        # Legacy '0-1' format maps to '<1'
        '0-1': '<1',
        '0-1yr': '<1',
        '0-1 yr': '<1',
        '0-1 yrs': '<1',
        '0-1years': '<1',
        '0-1 years': '<1',

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

        # For tenure bands with '5-10' or similar notation
        '5-10yr': '5-10',
        '5-10 yr': '5-10',
        '5-10 yrs': '5-10',
        '5-10years': '5-10',
        '5-10 years': '5-10',

        # For tenure bands with '10-15' or similar notation
        '10-15yr': '10-15',
        '10-15 yr': '10-15',
        '10-15 yrs': '10-15',
        '10-15years': '10-15',
        '10-15 years': '10-15',

        # For tenure bands with '15+' or similar notation
        '15+ yr': '15+',
        '15+ yrs': '15+',
        '15+yr': '15+',
        '15+years': '15+',
        '15+ years': '15+',
        '>15': '15+',

        # Legacy formats map to new format
        '5+': '15+',  # Legacy '5+' format maps to '15+'
        '5+ yr': '15+',
        '5+ yrs': '15+',
        '5+yr': '15+',
        '5+years': '15+',
        '5+ years': '15+',
        '>5': '15+',
        '10+': '15+',  # Legacy '10+' format maps to '15+'
        '10+ yr': '15+',
        '10+ yrs': '15+',
        '10+yr': '15+',
        '10+years': '15+',
        '10+ years': '15+',
        '>10': '15+'
    }

    # Return the standardized version if found in mapping, otherwise return the original
    # if it's already in standard format ('<1', '1-3', '3-5', '5-10', '10-15', '15+')
    if tb_norm in mapping:
        return mapping[tb_norm]
    elif tb_norm in ('<1', '1-3', '3-5', '5-10', '10-15', '15+'):
        return tb_norm
    else:
        # Log a warning about unexpected format here if desired
        # For now, try to determine the correct format based on patterns
        if tb_norm.startswith('0') or tb_norm.startswith('<'):
            return '<1'
        elif tb_norm.startswith('1'):
            return '1-3'
        elif tb_norm.startswith('3'):
            return '3-5'
        elif tb_norm.startswith('5'):
            return '5-10'
        elif tb_norm.startswith('10'):
            return '10-15'
        elif tb_norm.startswith('15') or tb_norm.startswith('>'):
            return '15+'
        else:
            # If we can't determine the format, return as is (could log a warning)
            return tb_norm

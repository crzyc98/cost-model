"""
Helper functions for parsing employer match formulas.
"""
import re
from typing import Tuple, List, Dict, Any


def parse_match_formula(formula_str: Any) -> Tuple[float, float]:
    """
    Parses an employer match formula string (e.g., "50% up to 6%") into components.

    Args:
        formula_str (str): The match formula string. Assumes format like
                           "X% up to Y%" or "X%". Returns (0, 0) if empty or invalid.

    Returns:
        tuple: A tuple containing (match_rate, cap_deferral_percentage).
               Returns (0, 0) if parsing fails or formula is empty.
               Rates are returned as decimals (e.g., 0.50 for 50%).
    """
    if not formula_str or not isinstance(formula_str, str):
        return 0.0, 0.0

    formula_str = formula_str.strip()
    # Simple case: Flat match rate (e.g., "5%")
    match_simple = re.match(r"\s*(\d+\.?\d*)\s*%", formula_str)
    if match_simple and "up to" not in formula_str.lower():
        match_rate = float(match_simple.group(1)) / 100.0
        return match_rate, 1.0

    # Standard case: "X% up to Y%"
    match_standard = re.match(r"(\d+\.?\d*)\s*%\s+up\s+to\s+(\d+\.?\d*)\s*%", formula_str, re.IGNORECASE)
    if match_standard:
        match_rate = float(match_standard.group(1)) / 100.0
        cap_deferral_perc = float(match_standard.group(2)) / 100.0
        return match_rate, cap_deferral_perc

    # Could not parse
    print(f"  Warning: Could not parse match formula: '{formula_str}'. Returning (0, 0).")
    return 0.0, 0.0


def parse_match_tiers(formula_str: Any) -> List[Dict[str, float]]:
    """
    Parses a tiered employer match formula string (e.g., "100% up to 3%, 50% up to 5%").

    Returns:
        list of dict: Each dict has 'match_pct' and 'deferral_cap_pct'.
    """
    if not formula_str or not isinstance(formula_str, str):
        return []

    tiers = []
    try:
        parts = formula_str.split(',')
        for part in parts:
            match_re = re.match(r"\s*(\d+\.?\d*)\s*%\s+up\s+to\s+(\d+\.?\d*)\s*%", part.strip(), re.IGNORECASE)
            if match_re:
                match_percent = float(match_re.group(1)) / 100.0
                deferral_cap_percent = float(match_re.group(2)) / 100.0
                tiers.append({'match_pct': match_percent, 'deferral_cap_pct': deferral_cap_percent})
        tiers.sort(key=lambda x: x['deferral_cap_pct'])
    except Exception as e:
        print(f"  Warning: Could not parse tiered match formula: '{formula_str}'. Error: {e}.")
        return []

    return tiers

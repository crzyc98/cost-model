# utils/rules/formula_parsers.py
import re
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

_SIMPLE_RX = re.compile(r"^\s*(\d+(\.\d+)?)\s*%\s*$")
_STANDARD_RX = re.compile(
    r"^\s*(\d+(\.\d+)?)\s*%\s+up\s+to\s+(\d+(\.\d+)?)\s*%$", re.IGNORECASE
)
_TIER_RX = re.compile(
    r"^\s*(\d+(\.\d+)?)\s*%\s+up\s+to\s+(\d+(\.\d+)?)\s*%$", re.IGNORECASE
)


def _pct_to_dec(val: str) -> float:
    return float(val) / 100.0


def parse_match_formula(formula_str: Any) -> Tuple[float, float]:
    if not formula_str or not isinstance(formula_str, str):
        return 0.0, 0.0

    s = formula_str.strip()
    m = _STANDARD_RX.match(s)
    if m:
        return _pct_to_dec(m.group(1)), _pct_to_dec(m.group(3))

    m = _SIMPLE_RX.match(s)
    if m:
        rate = _pct_to_dec(m.group(1))
        return rate, 1.0  # or (rate, rate) if you prefer

    logger.warning(
        "Could not parse match formula '%s'; returning (0.0, 0.0)", formula_str
    )
    return 0.0, 0.0


def parse_match_tiers(formula_str: Any) -> List[Dict[str, float]]:
    if not formula_str or not isinstance(formula_str, str):
        return []

    tiers = []
    for part in formula_str.split(","):
        part = part.strip()
        m = _TIER_RX.match(part)
        if m:
            tiers.append(
                {
                    "match_pct": _pct_to_dec(m.group(1)),
                    "deferral_cap_pct": _pct_to_dec(m.group(3)),
                }
            )

    if not tiers:
        logger.warning("Could not parse tiered match formula '%s'", formula_str)
        return []

    return sorted(tiers, key=lambda t: t["deferral_cap_pct"])

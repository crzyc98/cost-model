import logging
from typing import Dict, Optional
import pandas as pd
from .models import JobLevel

logger = logging.getLogger(__name__)

# Global state
LEVEL_TAXONOMY: Dict[int, JobLevel] = {}
_COMP_INTERVALS: Optional[pd.IntervalIndex] = None
_WARNING_COUNTS: Dict[str, int] = {"snap_below": 0, "snap_above": 0}
MAX_WARNINGS = 5

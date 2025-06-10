import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _generate_sequential_ids(existing_ids: Optional[Sequence[str]], num_new: int) -> List[str]:
    """Generates unique sequential placeholder IDs prefixed with 'NEW_'.

    Handles potential numeric suffixes in existing IDs to continue sequence,
    with fallbacks for non-numeric or complex existing IDs.
    """
    new_ids = []
    temp_new_ids = set()  # Use set for efficient checking of newly generated IDs

    current_max = 0
    existing_ids_set = set(existing_ids) if existing_ids else set()

    if existing_ids_set:
        try:
            # Extract numeric parts only if they are strings and match pattern 'prefix_number'
            numeric_parts = pd.to_numeric(
                pd.Series([s for s in existing_ids_set if isinstance(s, str)]).str.extract(
                    r"_(\d+)$", expand=False
                ),  # Look for underscore prefix
                errors="coerce",
            )
            if numeric_parts.notna().any():
                current_max = int(numeric_parts.max())
            else:
                # Fallback if no numeric parts found or all NaN
                current_max = 900000000 + len(existing_ids_set)
        except Exception as e:
            logger.warning(
                f"Error extracting numeric part of existing IDs: {e}. Using fallback sequence start."
            )
            current_max = 900000000 + len(existing_ids_set)
    else:
        current_max = 900000000  # Starting point if no existing IDs

    loop_count = 0
    max_loops = num_new * 100 + 100  # Safety break limit

    while len(new_ids) < num_new and loop_count < max_loops:
        current_max += 1
        potential_id = f"NEW_{current_max}"
        if potential_id not in existing_ids_set and potential_id not in temp_new_ids:
            new_ids.append(potential_id)
            temp_new_ids.add(potential_id)
        loop_count += 1

    if len(new_ids) < num_new:
        logger.warning(
            "Potential issue generating unique sequential IDs or reached loop limit; using fallback for remaining."
        )
        needed = num_new - len(new_ids)
        base_ts = pd.Timestamp.now().value
        rng_fallback = np.random.default_rng()  # Local RNG for fallback
        for i in range(needed):
            while True:
                fallback_id = f"FALLBACK_{base_ts}_{rng_fallback.integers(10000,99999)}_{i}"
                if fallback_id not in existing_ids_set and fallback_id not in temp_new_ids:
                    new_ids.append(fallback_id)
                    temp_new_ids.add(fallback_id)
                    break
    return new_ids

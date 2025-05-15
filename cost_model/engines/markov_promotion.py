import numpy as np
import pandas as pd
from typing import Tuple, Optional
from cost_model.state.job_levels.sampling import apply_promotion_markov
from cost_model.state.event_log import EVENT_COLS, EVT_PROMOTION, create_event
from cost_model.utils.columns import EMP_ID, EMP_LEVEL, EMP_ROLE, EMP_EXITED, EMP_LEVEL_SOURCE


def apply_markov_promotions(
    snapshot: pd.DataFrame,
    promo_time: pd.Timestamp,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Markov-chain based promotions to the workforce.
    
    Args:
        snapshot: Current workforce snapshot DataFrame
        promo_time: Timestamp for when promotions occur
        rng: Optional random number generator for reproducibility
    
    Returns:
        Tuple of (promotions_df, exits_df) where:
        - promotions_df: DataFrame of promotion events
        - exits_df: DataFrame of exit events
    """
    # Apply Markov promotions
    out = apply_promotion_markov(snapshot, rng=rng)
    
    # Create promotion events for level changes
    promoted_mask = (out[EMP_LEVEL] != snapshot[EMP_LEVEL]) & ~out[EMP_EXITED]
    promoted = out[promoted_mask].copy()
    
    if promoted.empty:
        return (
            pd.DataFrame(columns=EVENT_COLS),
            out[out[EMP_EXITED]].copy()
        )
    
    # Create promotion events
    promotions = pd.DataFrame({
        "event_time": promo_time,
        EMP_ID: promoted[EMP_ID],
        "event_type": EVT_PROMOTION,
        "value_json": promoted.apply(
            lambda r: json.dumps({
                "from_level": int(snapshot.loc[r.name, EMP_LEVEL]),
                "to_level": int(r[EMP_LEVEL]),
                "from_role": snapshot.loc[r.name, EMP_ROLE],
                "to_role": r[EMP_ROLE]
            }),
            axis=1
        ),
        "meta": "Markov-chain based promotion"
    })
    
    # Update job_level_source for promoted employees
    promoted[EMP_LEVEL_SOURCE] = 'markov-promo'
    
    # Return promotions and exits
    return promotions, out[out[EMP_EXITED]].copy()

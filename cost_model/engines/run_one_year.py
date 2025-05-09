import pandas as pd
import numpy as np
from typing import Tuple, List
from math import ceil

from . import comp, term, hire
from cost_model.state.event_log import EVT_COMP, EVT_TERM, EVT_HIRE, EVENT_COLS
from cost_model.state import snapshot


def run_one_year(year: int, prev_snapshot: pd.DataFrame, 
                 event_log: pd.DataFrame,
                 hazard_table: pd.DataFrame,
                 rng: np.random.Generator,
                 deterministic_term: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. extract Jan 1 snapshot (prev_snapshot)
    2. hazard_slice = hazard_table[hazard_table.year == year]
    3. comp_events  = comp.bump(prev_snapshot, hazard_slice, as_of=Jan1)
    4. term_events  = term.run(prev_snapshot, hazard_slice, rng, deterministic_term)
    5. temporary_snapshot = snapshot.update(prev_snapshot, comp_events+term_events)
    6. survivors = count active in temporary_snapshot
    7. target = ceil(active * (1+growth_rate_from_cfg))
    8. hire_events, hire_comp_events = hire.run(temporary_snapshot, target, hazard_slice, rng)
    9. all_new_events = comp_events + term_events + hire_events + hire_comp_events
    10. updated_snapshot = snapshot.update(temporary_snapshot, all_new_events)
    11. return updated_snapshot, event_log.append(all_new_events)
    """
    # 1. prev_snapshot is already Jan 1
    # 2. hazard_slice for this year
    hazard_slice = hazard_table[hazard_table['year'] == year]
    as_of = pd.Timestamp(f"{year}-01-01")

    # 3. Compensation bumps
    comp_events = comp.bump(prev_snapshot, hazard_slice, as_of)
    # 4. Terminations
    term_events = term.run(prev_snapshot, hazard_slice, rng, deterministic_term)
    # 5. Update snapshot with comp and term events
    temp_events = comp_events + term_events
    # Flatten and concat to single DataFrame
    temp_events_df = pd.concat([df for df in temp_events if isinstance(df, pd.DataFrame) and not df.empty], ignore_index=True) if temp_events else pd.DataFrame([], columns=EVENT_COLS)
    temporary_snapshot = snapshot.update(prev_snapshot, temp_events_df)
    # 6. Count survivors
    survivors = temporary_snapshot[temporary_snapshot['term_date'].isna() | (temporary_snapshot['term_date'] > as_of)]
    n_survivors = survivors.shape[0]
    # 7. Get growth rate from hazard_slice (assume all same for now)
    growth_rate = hazard_slice['growth_rate'].iloc[0] if 'growth_rate' in hazard_slice else 0.0
    target = int(ceil(n_survivors * (1 + growth_rate)))
    # 8. Hires
    hire_out = hire.run(temporary_snapshot, target, hazard_slice, rng)
    if isinstance(hire_out, list) and len(hire_out) == 2:
        hire_events, hire_comp_events = hire_out
    else:
        hire_events, hire_comp_events = hire_out, pd.DataFrame([], columns=EVENT_COLS)
    # 9. All new events
    all_new_events = comp_events + term_events + [hire_events, hire_comp_events]
    # Flatten and filter empty
    all_new_events = [df for df in all_new_events if isinstance(df, pd.DataFrame) and not df.empty]
    if all_new_events:
        all_new_events_df = pd.concat(all_new_events, ignore_index=True)
    else:
        all_new_events_df = pd.DataFrame([], columns=EVENT_COLS)
    # 10. Update snapshot with all new events
    # Flatten and concat hire events for update
    hire_events_df = pd.concat([df for df in [hire_events, hire_comp_events] if isinstance(df, pd.DataFrame) and not df.empty], ignore_index=True) if hire_events is not None else pd.DataFrame([], columns=EVENT_COLS)
    updated_snapshot = snapshot.update(temporary_snapshot, hire_events_df)
    # 11. Return updated snapshot and appended event log
    new_event_log = pd.concat([event_log, all_new_events_df], ignore_index=True)
    return updated_snapshot, new_event_log

# cost_model/plan_rules/enrollment.py
"""
Enrollment module for generating auto-enrollment and opt-out events based on plan rules.

## QuickStart

To generate enrollment events programmatically:

```python
import pandas as pd
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from cost_model.plan_rules.enrollment import run, EVT_ELIGIBLE, EVT_AUTO_ENROLL, EVT_ENROLL, EVT_OPT_OUT
from cost_model.config.plan_rules import EnrollmentConfig

# Create a sample snapshot with employee data
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0, 55000.0],
    'active': [True, True, True, True],
    'eligible': [True, True, True, True],
    'enrolled': [False, False, False, False]
}).set_index('employee_id')

# Create a sample events DataFrame with eligibility events
as_of = pd.Timestamp('2025-06-15')
eligibility_date = as_of - timedelta(days=30)  # Eligible 30 days ago

events = pd.DataFrame([
    {
        'event_id': str(uuid.uuid4()),
        'event_time': eligibility_date,
        'employee_id': emp_id,
        'event_type': EVT_ELIGIBLE,
        'value_num': None,
        'value_json': None,
        'meta': None
    } for emp_id in snapshot.index
])

# Configure enrollment rules
enrollment_config = EnrollmentConfig(
    enabled=True,
    auto_enroll=True,                # Enable auto-enrollment
    auto_enroll_delay_days=30,       # Wait 30 days after eligibility
    auto_enroll_default_pct=3.0,     # Default contribution rate of 3%
    opt_out_pct=25.0,                # 25% of employees opt out
    opt_out_delay_days=15            # Opt out within 15 days of auto-enrollment
)

# Generate enrollment events
enrollment_events = run(snapshot, events, as_of, enrollment_config)

# Examine the generated events
if enrollment_events:
    # Combine all event DataFrames
    all_events = pd.concat(enrollment_events)
    print(f"Generated {len(all_events)} enrollment events")
    
    # Count events by type
    event_counts = all_events['event_type'].value_counts()
    print("\nEvent counts by type:")
    for event_type, count in event_counts.items():
        print(f"{event_type}: {count}")
    
    # Show auto-enrollment details
    auto_enroll_events = all_events[all_events['event_type'] == EVT_AUTO_ENROLL]
    if not auto_enroll_events.empty:
        print("\nAuto-enrollment details:")
        print(auto_enroll_events[['employee_id', 'event_time', 'value_num']].head())
    
    # Show opt-out details
    opt_out_events = all_events[all_events['event_type'] == EVT_OPT_OUT]
    if not opt_out_events.empty:
        print("\nOpt-out details:")
        print(opt_out_events[['employee_id', 'event_time']].head())
    
    # Save the events
    output_dir = Path('output/enrollment')
    output_dir.mkdir(parents=True, exist_ok=True)
    all_events.to_parquet(output_dir / 'enrollment_events_2025.parquet')
else:
    print("No enrollment events generated")
```

This demonstrates how to generate auto-enrollment and opt-out events based on eligibility and plan rules.
"""

from datetime import timedelta
from typing import List

import uuid

import numpy as np
import pandas as pd

from cost_model.config.plan_rules import EnrollmentConfig
from cost_model.utils.columns import EMP_ID
import logging

logger = logging.getLogger(__name__)

EVT_ELIGIBLE = "EVT_ELIGIBLE"
EVT_AUTO_ENROLL = "EVT_AUTO_ENROLL"
EVT_ENROLL = "EVT_ENROLL"
EVT_OPT_OUT = "EVT_OPT_OUT"

EVENT_COLS = [
    "event_id",
    "event_time",
    EMP_ID,
    "event_type",
    "value_num",
    "value_json",
    "meta",
]
EVENT_DTYPES = {
    "event_id": "string",
    "event_time": "datetime64[ns]",
    EMP_ID: "string",
    "event_type": "string",
    "value_num": "float64",
    "value_json": "string",
    "meta": "string",
}


def run(
    snapshot: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: EnrollmentConfig,
) -> List[pd.DataFrame]:
    # 0) Check if config is present and enabled
    if not cfg or not getattr(cfg, 'enabled', True): # Default to enabled if 'enabled' attr is missing, but cfg must exist
        logger.debug("[Enrollment] rule skipped, config missing or disabled.")
        return []

    # 1) Ensure employee_id is index (optional)
    if snapshot.index.name != EMP_ID and EMP_ID in snapshot.columns:
        snapshot = snapshot.set_index(EMP_ID, drop=False)

    # 2) Sanity check on events
    if not {EMP_ID, "event_type", "event_time"}.issubset(events.columns):
        return []

    # 3) Who’s eligible?
    # Convert timestamps to date objects for more robust comparison
    event_dates = events.event_time.dt.date
    as_of_date = as_of.date()
    
    eligible_ids = (
        events.loc[
            (events.event_type == EVT_ELIGIBLE) & (event_dates <= as_of_date),
            EMP_ID,
        ]
        .astype(str)
        .unique()
    )
    logger.debug(f"[Enrollment] eligible_ids   {eligible_ids.tolist()}")

    # 4) Who’s already opted out or enrolled?
    blocked_ids = (
        events.loc[
            events.event_type.isin({EVT_OPT_OUT, EVT_ENROLL, EVT_AUTO_ENROLL}),
            EMP_ID,
        ]
        .astype(str)
        .unique()
    )
    logger.debug(f"[Enrollment] blocked_ids   {blocked_ids.tolist()}")

    candidates = set(eligible_ids) - set(blocked_ids)
    logger.debug(f"[Enrollment] candidates after exclusion   {candidates}")
    if not candidates:
        return []

    out: List[pd.DataFrame] = []
    rng = np.random.default_rng(getattr(cfg, "random_seed", 42))

    # Auto-enroll
    if getattr(cfg, "auto_enrollment", None) and cfg.auto_enrollment.enabled:
        rows = []
        for emp in list(candidates):
            rows.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "event_time": as_of,
                    EMP_ID: emp,
                    "event_type": EVT_AUTO_ENROLL,
                    "value_num": cfg.auto_enrollment.default_rate,
                    "value_json": None,
                    "meta": None,
                }
            )
        out.append(pd.DataFrame(rows, columns=EVENT_COLS).astype(EVENT_DTYPES))
        candidates.clear()

    # Voluntary
    vol_ids = sorted(list(candidates))
    if not vol_ids:
        return out

    if getattr(cfg, "voluntary_match_multiplier", None) is not None:
        rate = min(1.0, cfg.voluntary_enrollment_rate * cfg.voluntary_match_multiplier)
        logger.debug(
            f"[Enrollment] using match_multiplier; effective rate = {rate:.2%}"
        )
        sample = True
    elif getattr(cfg, "auto_enrollment", None) and cfg.auto_enrollment.enabled:
        rate = cfg.voluntary_enrollment_rate
        logger.debug(
            f"[Enrollment] auto_enrollment enabled; voluntary rate = {rate:.2%}"
        )
        sample = True
    else:
        rate = 1.0
        logger.debug(f"[Enrollment] no sampling; enrolling all  rate = {rate:.2%}")
        sample = False

    if sample:
        flips = rng.random(len(vol_ids))
        logger.debug(f"[Enrollment] flips = {flips}")
        chosen = [emp for emp, f in zip(vol_ids, flips) if f < rate]
        logger.debug(f"[Enrollment] chosen for voluntary   {chosen}")
    else:
        chosen = vol_ids
        logger.debug(f"[Enrollment] chosen (no sampling)   {chosen}")

    window = (
        getattr(cfg.auto_enrollment, "window_days", 0)
        if getattr(cfg, "auto_enrollment", None)
        else 0
    )
    rows = []
    for emp in chosen:
        # --- DEBUG: show what the engine sees for this employee ---
        subset = events.loc[
            events.event_type.eq(EVT_ELIGIBLE), [EMP_ID, "event_time"]
        ]
        logger.debug(f"[Enrollment] all eligibility rows:\n{subset}")

        # Try matching with a direct .eq() on employee_id
        eligibility_rows = subset.loc[subset[EMP_ID].eq(emp)]
        logger.debug(f"[Enrollment] emp {emp} eligible rows:\n{eligibility_rows}")
        if eligibility_rows.empty:
            logger.warning(f"[Enrollment] emp {emp} has no eligibility event? Skipping.")
            continue
        first_eligible_dt = eligibility_rows.event_time.min()
        effective_dt = first_eligible_dt + timedelta(days=window)
        logger.debug(f"[Enrollment]   emp {emp} eligible {first_eligible_dt}, effective {effective_dt}")
        if effective_dt.date() > as_of.date():
            logger.debug(f"[Enrollment]   emp {emp} not yet enrolled, window not passed")
            continue

        rows.append(
            {
                "event_id": str(uuid.uuid4()),
                "event_time": as_of,
                EMP_ID: emp,
                "event_type": EVT_ENROLL,
                "value_num": cfg.voluntary_default_rate,
                "value_json": None,
                "meta": None,
            }
        )

    if rows:
        out.append(pd.DataFrame(rows, columns=EVENT_COLS).astype(EVENT_DTYPES))

    return out

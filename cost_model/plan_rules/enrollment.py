# cost_model/plan_rules/enrollment.py

from datetime import timedelta
from typing import List

import json
import uuid
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from cost_model.config.plan_rules import EnrollmentConfig
import logging
logger = logging.getLogger(__name__)

EVT_ELIGIBLE     = "EVT_ELIGIBLE"
EVT_AUTO_ENROLL  = "EVT_AUTO_ENROLL"
EVT_ENROLL       = "EVT_ENROLL"
EVT_OPT_OUT      = "EVT_OPT_OUT"

EVENT_COLS = [
    "event_id", "event_time", "employee_id",
    "event_type", "value_num", "value_json", "meta"
]
EVENT_DTYPES = {
    "event_id": "string",
    "event_time": "datetime64[ns]",
    "employee_id": "string",
    "event_type": "string",
    "value_num": "float64",
    "value_json": "string",
    "meta": "string",
}


def run(
    snapshot: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: EnrollmentConfig
) -> List[pd.DataFrame]:
    # 1) Ensure employee_id is index (optional)
    if snapshot.index.name != "employee_id" and "employee_id" in snapshot.columns:
        snapshot = snapshot.set_index("employee_id", drop=False)

    # 2) Sanity check on events
    if not {"employee_id", "event_type", "event_time"}.issubset(events.columns):
        return []

    # 3) Who’s eligible?
    eligible_ids = (
        events.loc[
            (events.event_type == EVT_ELIGIBLE) &
            (events.event_time <= as_of),
            "employee_id"
        ]
        .astype(str)
        .unique()
    )
    logger.debug(f"[Enrollment] eligible_ids  {eligible_ids.tolist()}")

    # 4) Who’s already opted out or enrolled?
    blocked_ids = (
        events.loc[
            events.event_type.isin({EVT_OPT_OUT, EVT_ENROLL, EVT_AUTO_ENROLL}),
            "employee_id"
        ]
        .astype(str)
        .unique()
    )
    logger.debug(f"[Enrollment] blocked_ids  {blocked_ids.tolist()}")

    candidates = set(eligible_ids) - set(blocked_ids)
    logger.debug(f"[Enrollment] candidates after exclusion  {candidates}")
    if not candidates:
        return []

    out: List[pd.DataFrame] = []
    rng = np.random.default_rng(getattr(cfg, "random_seed", 42))

    # Auto-enroll
    if getattr(cfg, "auto_enrollment", None) and cfg.auto_enrollment.enabled:
        rows = []
        for emp in list(candidates):
            rows.append({
                "event_id": str(uuid.uuid4()),
                "event_time": as_of,
                "employee_id": emp,
                "event_type": EVT_AUTO_ENROLL,
                "value_num": cfg.auto_enrollment.default_rate,
                "value_json": None,
                "meta": None,
            })
        out.append(pd.DataFrame(rows, columns=EVENT_COLS).astype(EVENT_DTYPES))
        candidates.clear()

    # Voluntary
    vol_ids = sorted(list(candidates))
    if not vol_ids:
        return out

    if getattr(cfg, "voluntary_match_multiplier", None) is not None:
        rate = min(1.0, cfg.voluntary_enrollment_rate * cfg.voluntary_match_multiplier)
        logger.debug(f"[Enrollment] using match_multiplier; effective rate = {rate:.2%}")
        sample = True
    elif getattr(cfg, "auto_enrollment", None) and cfg.auto_enrollment.enabled:
        rate = cfg.voluntary_enrollment_rate
        logger.debug(f"[Enrollment] auto_enrollment enabled; voluntary rate = {rate:.2%}")
        sample = True
    else:
        rate = 1.0
        logger.debug(f"[Enrollment] no sampling; enrolling all  rate = {rate:.2%}")
        sample = False

    if sample:
        flips = rng.random(len(vol_ids))
        logger.debug(f"[Enrollment] flips = {flips}")
        chosen = [emp for emp, f in zip(vol_ids, flips) if f < rate]
        logger.debug(f"[Enrollment] chosen for voluntary  {chosen}")
    else:
        chosen = vol_ids
        logger.debug(f"[Enrollment] chosen (no sampling)  {chosen}")

    window = getattr(cfg.auto_enrollment, "window_days", 0) if getattr(cfg, "auto_enrollment", None) else 0
    rows = []
    for emp in chosen:
        # --- DEBUG: show what the engine sees for this employee ---
        subset = events.loc[events.event_type.eq(EVT_ELIGIBLE), ['employee_id','event_time']]
        logger.debug(f"[Enrollment] all eligibility rows:\n{subset}")

        # Try matching with a direct .eq() on employee_id
        mask = events.employee_id.eq(emp) & events.event_type.eq(EVT_ELIGIBLE)
        elig_dates = events.loc[mask, "event_time"].dropna()
        logger.debug(f"[Enrollment] elig_dates for '{emp}': {elig_dates.tolist()}")

        if elig_dates.empty:
            logger.debug(f"[Enrollment] skipping {emp}: no matching elig_dates")
            continue

        enroll_date = elig_dates.min() + timedelta(days=window)
        logger.debug(f"[Enrollment] computed enroll_date for {emp}: {enroll_date}")

        if enroll_date <= as_of:
            logger.debug(f"[Enrollment] appending enrollment event for {emp}")
            rows.append({
                "event_id": str(uuid.uuid4()),
                "event_time": enroll_date,
                "employee_id": emp,
                "event_type": EVT_ENROLL,
                "value_num": cfg.default_rate,
                "value_json": None,
                "meta": None,
            })
    if rows:
        out.append(pd.DataFrame(rows, columns=EVENT_COLS).astype(EVENT_DTYPES))

    return out
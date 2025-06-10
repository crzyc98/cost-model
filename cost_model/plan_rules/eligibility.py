# cost_model/plan_rules/eligibility.py
"""
Eligibility module for determining employee eligibility for retirement plans.
QuickStart: see docs/cost_model/plan_rules/eligibility.md
"""

import logging
import uuid
from datetime import date
from typing import List

import pandas as pd

from cost_model.config.plan_rules import EligibilityConfig
from cost_model.plan_rules.enrollment import EVENT_COLS, EVENT_DTYPES, EVT_ELIGIBLE
from cost_model.utils.columns import EMP_BIRTH_DATE, EMP_HIRE_DATE, EMP_ID

logger = logging.getLogger(__name__)


def run(snapshot: pd.DataFrame, as_of: date, cfg: EligibilityConfig) -> List[pd.DataFrame]:
    """
    Determine eligibility events for each employee in the snapshot.
    Returns a list of DataFrames (event records) in canonical event log schema.
    """
    rows = []
    for idx, row in snapshot.iterrows():
        birth = row.get(EMP_BIRTH_DATE)
        hire = row.get(EMP_HIRE_DATE)
        if birth is None or hire is None:
            logger.debug(f"Skipping {row.get(EMP_ID, idx)}: missing birth or hire date")
            continue
        # Calculate age
        age = (as_of.year - birth.year) - ((as_of.month, as_of.day) < (birth.month, birth.day))
        # Calculate service in months
        service_months = (as_of.year - hire.year) * 12 + (as_of.month - hire.month)
        logger.debug(
            f"Eligibility check for '{row.get(EMP_ID, idx)}': "
            f"birth={birth}, hire={hire}, age={age}, service_months={service_months}"
        )
        print(
            f"[ELIGIBILITY DEBUG] emp={row.get(EMP_ID, idx)}, birth={birth}, hire={hire}, age={age}, service_months={service_months}"
        )
        if age >= cfg.min_age and service_months >= cfg.min_service_months:
            print(f"[ELIGIBILITY DEBUG] --> ELIGIBLE")
            rows.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "event_time": as_of,
                    EMP_ID: row[EMP_ID],
                    "event_type": EVT_ELIGIBLE,
                    "value_num": None,
                    "value_json": None,
                    "meta": None,
                }
            )
        else:
            print(f"[ELIGIBILITY DEBUG] --> NOT eligible")
            logger.debug(
                f"Not eligible: {row.get(EMP_ID, idx)} (age={age}, service_months={service_months})"
            )
    if rows:
        return [pd.DataFrame(rows, columns=EVENT_COLS).astype(EVENT_DTYPES)]
    else:
        return []

"""
Functions for extracting employee details from events.
"""

import json
import logging
import pandas as pd
from typing import Dict, Any

from .constants import EMP_ID, EMP_ROLE, EMP_BIRTH_DATE

logger = logging.getLogger(__name__)

def extract_hire_details(hire_events: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts static details (role, birth_date) from hire event records.
    Assumes these details are stored in the 'value_json' field of hire events.

    Args:
        hire_events: DataFrame containing only hire events per employee.

    Returns:
        A DataFrame indexed by EMP_ID with columns EMP_ROLE, EMP_BIRTH_DATE.
    """
    details = []
    if hire_events.empty:
        return pd.DataFrame(
            columns=[EMP_ROLE, EMP_BIRTH_DATE], index=pd.Index([], name=EMP_ID)
        )

    for _, event in hire_events.iterrows():
        employee_id = event[EMP_ID]
        role = None
        birth_date = pd.NaT  # Default to Not-a-Time

        # Try extracting from value_json
        if pd.notna(event["value_json"]):
            try:
                # Load JSON string into a Python dict
                data = json.loads(event["value_json"])
                role = data.get("role")  # Safely get role
                # Safely get and parse birth_date
                birth_date_str = data.get("birth_date")
                if birth_date_str:
                    birth_date = pd.to_datetime(birth_date_str, errors="coerce")

            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"Could not parse value_json for hire event {event.get('event_id', 'N/A')} for emp {employee_id}. Error: {e}"
                )
        # else: # Optionally check 'meta' as a fallback if needed
        #     pass

        if role is None:
            logger.debug(
                f"Could not extract 'role' for emp {employee_id} from hire event."
            )
        if pd.isna(birth_date):
            logger.debug(
                f"Could not extract valid 'birth_date' for emp {employee_id} from hire event."
            )

        details.append({EMP_ID: employee_id, EMP_ROLE: role, EMP_BIRTH_DATE: birth_date})

    if not details:  # Should not happen if hire_events was not empty, but safeguard
        return pd.DataFrame(
            columns=[EMP_ROLE, EMP_BIRTH_DATE], index=pd.Index([], name=EMP_ID)
        )

    details_df = pd.DataFrame(details).set_index(EMP_ID)

    # Ensure correct dtypes before returning
    details_df[EMP_ROLE] = details_df[EMP_ROLE].astype(pd.StringDtype())
    details_df[EMP_BIRTH_DATE] = pd.to_datetime(
        details_df[EMP_BIRTH_DATE]
    )  # Already datetime, but ensures consistency
    return details_df

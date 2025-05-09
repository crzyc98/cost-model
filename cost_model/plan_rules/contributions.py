# cost_model/plan_rules/contributions.py

import pandas as pd
import numpy as np
import json
import uuid
from datetime import date
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field # Keep Pydantic if models used

# Import the actual config models you need
# Adjust path based on your structure
try:
    from ..config.models import EmployerMatchRules, MatchTier
    # Determine if you pass PlanRules or just EmployerMatchRules
    ConfigType = EmployerMatchRules # Or PlanRules
except ImportError:
    # Fallback placeholder if models aren't available (not recommended for production)
    class MatchTier(BaseModel): match_rate: float; cap_deferral_pct: float
    class EmployerMatchRules(BaseModel): tiers: List[MatchTier] = []
    ConfigType = EmployerMatchRules

# Import constants and schema info from event_log
try:
    from ..state.event_log import EVT_CONTRIB, EVT_ENROLL, EVENT_COLS, EVENT_PANDAS_DTYPES, EMP_ID
except ImportError:
    EVT_CONTRIB = 'EVT_CONTRIB'
    EVT_ENROLL = 'EVT_ENROLL'
    EMP_ID = 'employee_id'
    EVENT_COLS = ["event_id", "event_time", EMP_ID, "event_type", "value_num", "value_json", "meta"]
    EVENT_PANDAS_DTYPES = { # Example dtypes
        "event_id": pd.StringDtype(), "event_time": "datetime64[ns]", EMP_ID: pd.StringDtype(),
        "event_type": pd.StringDtype(), "value_num": pd.Float64Dtype(),
        "value_json": pd.StringDtype(), "meta": pd.StringDtype()
    }


import logging
logger = logging.getLogger(__name__)

# --- Corrected Function ---
def run(
    snapshot: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: ConfigType # Use the imported config type (e.g., EmployerMatchRules or PlanRules)
) -> pd.DataFrame:
    """
    Generate contribution events for enrolled employees based on snapshot data.
    """
    logger.debug(f"Running contribution calculation for {as_of.date()}")

    # 1) Get match tiers safely from the config object
    tiers: List[MatchTier] = [] # Default to empty list
    if cfg:
        # If cfg is PlanRules, access via cfg.employer_match.tiers
        if hasattr(cfg, 'employer_match') and cfg.employer_match:
            tiers = cfg.employer_match.tiers
        # If cfg is EmployerMatchRules directly, access via cfg.tiers
        elif hasattr(cfg, 'tiers'):
            tiers = cfg.tiers

    if not tiers:
        logger.info("No employer match tiers found in config. Only EE contributions calculated (if any).")
        # Decide if you still want to create events with ER=0 or return early.
        # We'll continue and calculate match as 0 if no tiers.

    out_events: List[Dict] = [] # Collect event dictionaries

    # 2) Ensure snapshot is indexed by employee_id
    if snapshot.index.name != EMP_ID:
        if EMP_ID in snapshot.columns:
            logger.warning(f"Snapshot not indexed by '{EMP_ID}'. Setting index.")
            try:
                snapshot = snapshot.set_index(EMP_ID, drop=False)
            except Exception as e:
                 logger.error(f"Failed to set snapshot index to '{EMP_ID}': {e}")
                 return pd.DataFrame(columns=EVENT_COLS).astype(EVENT_PANDAS_DTYPES)
        else:
            logger.error(f"Snapshot missing required index/column '{EMP_ID}'.")
            return pd.DataFrame(columns=EVENT_COLS).astype(EVENT_PANDAS_DTYPES)

    # 3) Check events DataFrame
    required_event_cols = {EMP_ID, 'event_type', 'event_time'}
    if events.empty or not required_event_cols.issubset(events.columns):
        logger.warning("Input events DataFrame unusable for enrollment check.")
        return pd.DataFrame(columns=EVENT_COLS).astype(EVENT_PANDAS_DTYPES)

    # 4) Find enrolled employees
    try:
        enrolled = events.loc[
            (events.event_type == EVT_ENROLL) &
            (events.event_time <= as_of),
            "employee_id"
        ].unique()
        if not enrolled.size:
            logger.info("No employees found enrolled by the specified 'as_of' date.")
            return pd.DataFrame(columns=EVENT_COLS).astype(EVENT_PANDAS_DTYPES)
        logger.debug(f"Found {len(enrolled)} enrolled employees.")
        enrolled_ids = set(enrolled.astype(str))
    except KeyError as e:
        logger.error(f"Missing column in events DataFrame for enrollment check: {e}")
        return pd.DataFrame(columns=EVENT_COLS).astype(EVENT_PANDAS_DTYPES)

    # 5) Find who already contributed today
    try:
        already_contributed_mask = (events['event_type'] == EVT_CONTRIB) & (events['event_time'] == as_of)
        already_contributed_ids = set(events.loc[already_contributed_mask, EMP_ID].astype(str).unique())
        if already_contributed_ids:
             logger.debug(f"Found {len(already_contributed_ids)} employees with existing contribution event on {as_of.date()}.")
    except KeyError as e:
        logger.error(f"Missing column in events DataFrame for contribution check: {e}")
        return pd.DataFrame(columns=EVENT_COLS).astype(EVENT_PANDAS_DTYPES)

    # 6) Loop through candidates
    candidate_ids = enrolled_ids - already_contributed_ids
    logger.debug(f"Processing contributions for {len(candidate_ids)} candidates.")

    required_snapshot_cols = ['employee_gross_compensation', 'employee_deferral_rate']
    if not all(col in snapshot.columns for col in required_snapshot_cols):
         logger.error(f"Snapshot missing required columns for contribution calc: {required_snapshot_cols}")
         return pd.DataFrame(columns=EVENT_COLS).astype(EVENT_PANDAS_DTYPES)

    processed_count = 0
    for emp_id in candidate_ids:
        if emp_id not in snapshot.index:
            logger.warning(f"Enrolled employee '{emp_id}' not found in snapshot index. Skipping.")
            continue

        try:
            comp = snapshot.loc[emp_id, 'employee_gross_compensation']
            rate = snapshot.loc[emp_id, 'employee_deferral_rate']
            if pd.isna(comp) or pd.isna(rate): continue # Skip if data missing
            comp = float(comp)
            rate = float(rate)
        except (KeyError, ValueError, TypeError):
            logger.warning(f"Data lookup/type error for comp/rate for employee '{emp_id}'. Skipping.")
            continue

        # Calculate Employee Deferral
        ee = comp * rate
        if ee <= 0:
            continue # Skip if not deferring

        # Calculate Employer Match based on *correct* tiered logic
        er = 0.0
        processed_deferral_pct = 0.0 # Track the top of the last tier processed

        # Sort tiers by cap_deferral_pct to process correctly
        sorted_tiers = sorted(tiers, key=lambda t: t.cap_deferral_pct)

        for tier in sorted_tiers:
            tier_cap_pct = tier.cap_deferral_pct
            tier_match_rate = tier.match_rate

            # Deferral percentage relevant for *this specific tier*
            # (i.e., deferral between previous tier's cap and this tier's cap)
            effective_deferral_this_tier = max(0, min(rate, tier_cap_pct) - processed_deferral_pct)

            if effective_deferral_this_tier <= 0:
                continue # No deferral % falls into this tier

            # Calculate match base for this tier's contribution
            match_base_this_tier = effective_deferral_this_tier * comp
            er += match_base_this_tier * tier_match_rate

            # Update the processed ceiling
            processed_deferral_pct = tier_cap_pct

        # Format payload for value_json
        contrib_payload = {'employee_deferral': round(ee, 2), 'employer_match': round(er, 2)}

        # Create event dictionary matching EVENT_COLS structure
        out_events.append({
            "event_id": str(uuid.uuid4()),
            "event_time": as_of,
            EMP_ID: emp_id,
            "event_type": EVT_CONTRIB,
            "value_num": None,
            "value_json": json.dumps(contrib_payload), # Serialize dict to JSON string
            "meta": None
        })
        processed_count += 1

    logger.info(f"Generated {processed_count} contribution events for {as_of.date()}.")

    # 7. Create and return final DataFrame
    if out_events:
        out_df = pd.DataFrame(out_events)
        # Ensure schema and dtypes
        for col in EVENT_COLS: # Use the imported list of columns
            if col not in out_df.columns: out_df[col] = None
        out_df = out_df[EVENT_COLS].astype(EVENT_PANDAS_DTYPES) # Use imported dtypes
        return out_df
    else:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=EVENT_COLS).astype(EVENT_PANDAS_DTYPES)
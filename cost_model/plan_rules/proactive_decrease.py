# cost_model/plan_rules/proactive_decrease.py

"""
Proactive Decrease Engine:

Scans recent contribution history to detect if an employee's deferral rate has fallen by more than
 a configured threshold relative to their peak contribution rate in the lookback window.  If so,
 emits a "proactive decrease" event for plan adjustment.
"""
import json
import uuid
from datetime import timedelta # Not strictly used in this version, but often useful for date logic
from typing import List

import pandas as pd
import numpy as np # Import numpy for np.nan

# Assuming ProactiveDecreaseConfig is defined in your config models
try:
    from ..config.models import ProactiveDecreaseConfig
except ImportError:
    # Fallback for standalone testing or if path is different
    from pydantic import BaseModel, Field # type: ignore
    class ProactiveDecreaseConfig(BaseModel): # type: ignore
        lookback_months: int = 12
        threshold_pct: float = 0.05
        event_type: str = "EVT_PROACTIVE_DECREASE"

# Assuming EVENT_COLS and EVENT_DTYPES are defined in event_log.py
# and EMP_ID in columns.py
try:
    from ..state.event_log import EVENT_COLS, EVENT_PANDAS_DTYPES
    from ..utils.columns import EMP_ID, EMP_DEFERRAL_RATE
except ImportError:
    # Fallbacks for standalone testing
    EVENT_COLS = [
        "event_id", "event_time", EMP_ID,
        "event_type", "value_num", "value_json", "meta"
    ]
    EVENT_PANDAS_DTYPES = {
        "event_id": "string",
        "event_time": "datetime64[ns]",
        EMP_ID: "string",
        "event_type": "string",
        "value_num": "float64",
        "value_json": "string",
        "meta": "string",
    }

import logging
# logger = logging.getLogger(__name__) # Using print for direct visibility in pytest -s

def run(
    snapshot: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: ProactiveDecreaseConfig
) -> List[pd.DataFrame]:
    """
    For each employee in `snapshot`, look back `cfg.lookback_months` from `as_of` to find their
    historical peak contribution rate (from EVT_CONTRIB or EVT_CONTRIB_INCR in value_num).
    If the difference between that peak and the current deferral rate in `snapshot['employee_deferral_rate']`
    is >= cfg.threshold_pct, emit one decrease event per employee.

    Returns an empty list if no events are emitted, otherwise a list containing a single DataFrame of events.
    """
    # 0) Check if config is present and enabled
    if not cfg or not getattr(cfg, 'enabled', True): # Default to enabled if 'enabled' attr is missing
        print("[PD_RUN_DEBUG] Proactive Decrease rule skipped, config missing or disabled.") # Using print to match existing style
        return []

    # Ensure event_time is datetime64[ns] for filtering to work
    if not np.issubdtype(events['event_time'].dtype, np.datetime64):
        print("[PD_RUN_DEBUG] Converting event_time to datetime64[ns]")
        events['event_time'] = pd.to_datetime(events['event_time'])
    print(f"[PD_RUN_DEBUG] Running proactive decrease for as_of: {as_of.date()}") # Changed to print
    # Define lookback window
    window_start = (as_of - pd.Timedelta(days=cfg.lookback_months * 30.4375)).normalize()
    print("[PD_RUN_DEBUG] event_time dtype:", events['event_time'].dtype)
    print("[PD_RUN_DEBUG] event_time min/max:", events['event_time'].min(), events['event_time'].max())
    print("[PD_RUN_DEBUG] window_start:", window_start, "as_of:", as_of)
    print("[PD_RUN_DEBUG] All event_time values:\n", events[[EMP_ID,'event_time','event_type']]) # Changed 'employee_id' to EMP_ID
    # Ensure snapshot indexed by employee_id
    if snapshot.index.name != EMP_ID:
        if EMP_ID in snapshot.columns:
            print(f"[PD_RUN_DEBUG] Setting snapshot index to '{EMP_ID}'.") # Changed to print
            snapshot = snapshot.set_index(EMP_ID, drop=False)
        else:
            print(f"[PD_RUN_ERROR] Snapshot missing '{EMP_ID}' column for index.") # Changed to print
            return []

    print(f"[PD_RUN_DEBUG] Lookback window: {window_start.date()} to {as_of.date()}") # Changed to print

    valid_contrib_event_types = {'EVT_CONTRIB', 'EVT_CONTRIB_INCR', 'EVT_ENROLL', 'EVT_AUTO_ENROLL'}
    required_event_cols = {'event_type', 'event_time', EMP_ID, 'value_num'}
    if not required_event_cols.issubset(events.columns):
        print(f"[PD_RUN_ERROR] Events DataFrame missing one of required columns: {required_event_cols}") # Changed to print
        return []

    # Convert timestamps to date objects for comparison to avoid issues with hidden milliseconds
    print("[PD_RUN_DEBUG] Raw window_start:", repr(window_start))
    print("[PD_RUN_DEBUG] Raw event_time[0]:", repr(events['event_time'].iloc[0]) if len(events) > 0 else "No events")
    
    # Use date objects for comparison instead of timestamps
    event_dates = events['event_time'].dt.date
    window_start_date = window_start.date()
    as_of_date = as_of.date()
    
    hist_mask = (
        events['event_type'].isin(valid_contrib_event_types) &
        (event_dates >= window_start_date) &
        (event_dates < as_of_date)
    )
    hist = events.loc[hist_mask].copy()
    
    # Debug the mask components
    print("[PD_RUN_DEBUG] Type check:", events['event_type'].isin(valid_contrib_event_types).sum(), "events match valid types")
    print("[PD_RUN_DEBUG] Start date check:", (event_dates >= window_start_date).sum(), "events >= window_start_date")
    print("[PD_RUN_DEBUG] End date check:", (event_dates < as_of_date).sum(), "events < as_of_date")
    print(f"[PD_RUN_DEBUG] Found {len(hist)} historical contribution-related events in window.") # Changed to print
    if not hist.empty:
        print(f"[PD_RUN_DEBUG] Sample of hist (historical events for peak calc):\n{hist.head().to_string()}")

    if 'value_num' in hist.columns:
        hist['value_num'] = pd.to_numeric(hist['value_num'], errors='coerce')
    else:
        print("[PD_RUN_WARNING] 'value_num' column not found in hist. Creating with NaN.") # Changed to print
        hist['value_num'] = np.nan

    if not hist.empty and 'value_num' in hist.columns:
        peaks = hist.groupby(EMP_ID)['value_num'].max()
    else:
        peaks = pd.Series(dtype='float64', name='value_num').rename_axis(EMP_ID)
    print(f"[PD_RUN_DEBUG] Calculated peaks (sample):\n{peaks.head().to_string()}") # Changed to print
    if 'C' in peaks.index:
        print(f"[PD_RUN_DEBUG] Peak for C from hist: {peaks.get('C')}")


    if EMP_DEFERRAL_RATE not in snapshot.columns:
        print(f"[PD_RUN_ERROR] Snapshot missing '{EMP_DEFERRAL_RATE}' column.") # Changed
        return []
    current_rates = snapshot[EMP_DEFERRAL_RATE]

    output_event_rows = []
    for emp_id_val in current_rates.index:
        current_rate_val = current_rates.loc[emp_id_val]
        if pd.isna(current_rate_val):
            print(f"[PD_RUN_DEBUG] emp={emp_id_val}, current_rate is NA. Skipping.") # Changed to print
            continue
        current_rate = float(current_rate_val)
        
        peak_from_series = peaks.get(emp_id_val) # Get raw value from peaks
        
        peak_rate_val = peaks.get(emp_id_val, current_rate)
        if pd.isna(peak_rate_val):
            peak_rate = current_rate
        else:
            peak_rate = float(peak_rate_val)

        print(f"[PD_RUN_DEBUG] emp={emp_id_val}, current_rate={current_rate:.4f}, raw_peak_from_series={peak_from_series}, effective_peak_rate={peak_rate:.4f}, threshold={cfg.threshold_pct:.4f}") # Changed to print

        if (peak_rate - current_rate) >= cfg.threshold_pct:
            delta = current_rate - peak_rate
            print(f"[PD_RUN_INFO] FIRED for emp={emp_id_val}. Peak: {peak_rate:.2%}, Current: {current_rate:.2%}, Delta: {delta:.2%}") # Changed to print
            output_event_rows.append({
                "event_id": str(uuid.uuid4()), "event_time": as_of, EMP_ID: emp_id_val,
                "event_type": cfg.event_type, "value_num": None,
                "value_json": json.dumps({"old_rate": peak_rate, "new_rate": current_rate, "delta": delta}),
                "meta": None,
            })
        else:
            print(f"[PD_RUN_DEBUG] --> NOT fired for emp={emp_id_val}") # Changed to print

    if not output_event_rows:
        print("[PD_RUN_DEBUG] No proactive decrease events generated.") # Changed to print
        return []

    df_out = pd.DataFrame(output_event_rows)
    for col in EVENT_COLS:
        if col not in df_out.columns: df_out[col] = None
    # Skip applying dtypes since EVENT_DTYPES is not defined
    print(f"[PD_RUN_INFO] Generated {len(df_out)} proactive decrease events.") # Changed to print
    return [df_out]

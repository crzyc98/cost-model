# cost_model/state/snapshot_utils.py
"""
Utility helpers for snapshot build/update flows.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from cost_model.state.schema import (
    EMP_BIRTH_DATE,
    EMP_ID,
    EMP_LEVEL,
    SNAPSHOT_COLS,
    SNAPSHOT_DTYPES,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Event filtering helpers
# -----------------------------------------------------------------------------


def get_first_event(
    events: pd.DataFrame, event_type: Union[str, List[str]]
) -> pd.DataFrame:  # noqa: N802
    """Return first occurrence of *event_type* per employee."""
    if isinstance(event_type, str):
        events_of_type = events[events["event_type"] == event_type]
    else:  # it's a list
        events_of_type = events[events["event_type"].isin(event_type)]
    return events_of_type.drop_duplicates(subset=EMP_ID, keep="first")


def get_last_event(
    events: pd.DataFrame, event_type: Union[str, List[str]]
) -> pd.DataFrame:  # noqa: N802
    """Return last occurrence of *event_type* per employee."""
    if isinstance(event_type, str):
        events_of_type = events[events["event_type"] == event_type]
    else:  # it's a list
        events_of_type = events[events["event_type"].isin(event_type)]
    return events_of_type.drop_duplicates(subset=EMP_ID, keep="last")


# -----------------------------------------------------------------------------
# JSON parsing helper for hire events
# -----------------------------------------------------------------------------


def parse_hire_payload(value_json: str | None) -> Dict[str, Any]:  # noqa: D401
    """Return dict parsed from *value_json* or empty dict on error."""
    if value_json is None or pd.isna(value_json):
        return {}
    try:
        return json.loads(value_json)
    except (TypeError, json.JSONDecodeError):
        return {}


# -----------------------------------------------------------------------------
# Hire details extraction
# -----------------------------------------------------------------------------


def extract_hire_details(hire_events: pd.DataFrame) -> pd.DataFrame:  # noqa: N802
    """Return DF indexed by EMP_ID with EMP_LEVEL & EMP_BIRTH_DATE columns."""
    if hire_events.empty:
        return pd.DataFrame(columns=[EMP_ID, EMP_LEVEL, EMP_BIRTH_DATE]).set_index(EMP_ID)

    records = []
    for _, evt in hire_events.iterrows():
        emp_id = evt[EMP_ID]
        payload = parse_hire_payload(evt.get("value_json"))

        # Get level from payload or default to 1
        level = payload.get("level", 1)

        # Parse birth date
        birth_raw = payload.get("birth_date")
        birth_dt = pd.to_datetime(birth_raw, errors="coerce") if birth_raw else pd.NaT

        records.append({EMP_ID: emp_id, EMP_LEVEL: level, EMP_BIRTH_DATE: birth_dt})

    # Create DataFrame with proper types
    df = pd.DataFrame(records)
    df[EMP_LEVEL] = df[EMP_LEVEL].fillna(1).astype("int64")  # Ensure level is integer
    df = df.set_index(EMP_ID)

    return df


# -----------------------------------------------------------------------------
# Column/type enforcement
# -----------------------------------------------------------------------------


def ensure_columns_and_types(df: pd.DataFrame) -> pd.DataFrame:  # noqa: N802
    """Ensure *df* conforms to SNAPSHOT_COLS/DTYPES order & dtypes.

    Critical fix: do **not** overwrite existing non-null ``EMP_ID`` values. In prior logic, pandas
    conversion to ``pd.StringDtype`` occasionally wiped out valid IDs, leaving the column full of
    ``<NA>``. We now:
      1. Add truly *missing* columns only.
      2. Apply dtype coercion *except* for the ``EMP_ID`` column, which we coerce separately to
         a python string dtype while preserving data.
      3. Assert the result still contains non-null & unique IDs.
    """
    # Work on a copy to avoid mutating caller's DF
    # ------------------------------------------------------------------
    # 0. Ensure EMP_ID is a column, not an index (some operations may
    #    accidentally set it as the index which hides it from df.columns)
    # ------------------------------------------------------------------
    if EMP_ID not in df.columns and EMP_ID in df.index.names:
        df = df.reset_index()
        logger.warning("[ENSURE] EMP_ID found as index; reset to column")

    result = df.copy()

    # Preserve original EMP_ID values to avoid accidental coercion side-effects
    emp_id_series = result[EMP_ID].copy() if EMP_ID in result.columns else None

    if EMP_ID in result.columns:
        pre_null = result[EMP_ID].isna().sum()
        pre_dup = result[EMP_ID].duplicated().sum()
        logger.info(
            f"[ENSURE] Pre-process EMP_ID integrity: {pre_dup} dup, {pre_null} null (rows={len(result)})"
        )

    # ------------------------------------------------------------------
    # 1. Ensure all required columns exist (do NOT touch existing ones)
    # ------------------------------------------------------------------
    for col, dtype in SNAPSHOT_DTYPES.items():
        if col not in result.columns:
            if pd.api.types.is_numeric_dtype(dtype):
                result[col] = np.nan
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                result[col] = pd.NaT
            elif dtype == pd.BooleanDtype():
                result[col] = pd.NA
            else:
                result[col] = pd.NA

    # ------------------------------------------------------------------
    # 2. Re-order columns per schema (preserves existing data)
    # ------------------------------------------------------------------
    result = result[SNAPSHOT_COLS].copy()

    if EMP_ID in result.columns:
        post_reorder_null = result[EMP_ID].isna().sum()
        post_reorder_dup = result[EMP_ID].duplicated().sum()
        logger.info(
            f"[ENSURE] After reorder EMP_ID integrity: {post_reorder_dup} dup, {post_reorder_null} null"
        )

    # Replace pd.NA with np.nan in numeric columns before astype
    for col, dtype in SNAPSHOT_DTYPES.items():
        if pd.api.types.is_numeric_dtype(dtype) and col in result.columns:
            result.loc[:, col] = result[col].replace({pd.NA: np.nan})

    # ------------------------------------------------------------------
    # 3. Apply dtype coercion safely
    #    – apply to all columns *except* EMP_ID to avoid clobbering values
    # ------------------------------------------------------------------
    dtype_map = {k: v for k, v in SNAPSHOT_DTYPES.items() if k != EMP_ID}
    result = result.astype(dtype_map)

    if EMP_ID in result.columns:
        post_dtype_null = result[EMP_ID].isna().sum()
        post_dtype_dup = result[EMP_ID].duplicated().sum()
        logger.info(
            f"[ENSURE] After astype EMP_ID integrity: {post_dtype_dup} dup, {post_dtype_null} null"
        )

    # Restore EMP_ID column (unchanged) and set dtype explicitly
    if emp_id_series is not None:
        result[EMP_ID] = emp_id_series

    # Explicitly coerce EMP_ID to python string dtype without introducing <NA>
    if EMP_ID in result.columns:
        result[EMP_ID] = result[EMP_ID].astype("string[python]")
        post_final_null = result[EMP_ID].isna().sum()
        post_final_dup = result[EMP_ID].duplicated().sum()
        logger.info(
            f"[ENSURE] After final coercion EMP_ID integrity: {post_final_dup} dup, {post_final_null} null"
        )

    # ------------------------------------------------------------------
    # 4. Integrity assertions – must keep non-null, unique EMP_IDs
    # ------------------------------------------------------------------
    if EMP_ID in result.columns:
        null_ids = result[EMP_ID].isna().sum()
        dup_ids = result[EMP_ID].duplicated().sum()
        if null_ids or dup_ids:
            msg = (
                f"ensure_columns_and_types integrity failure: {null_ids} null and "
                f"{dup_ids} duplicate {EMP_ID}s after dtype coercion"
            )
            logger.error(msg)
            raise ValueError(msg)

    return result

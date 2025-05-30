# cost_model/state/snapshot.py
"""
Compatibility layer for the workforce snapshot module.
This file maintains backward compatibility while gradually transitioning to the new modular structure.
"""

import warnings
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import pandas as pd
import numpy as np

from .snapshot_build import build_full
from .snapshot_update import update
from .schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_LEVEL, EMP_GROSS_COMP,
    EMP_TERM_DATE, EMP_ACTIVE, EMP_TENURE, EMP_TENURE_BAND,
    EMP_DEFERRAL_RATE, SNAPSHOT_COLS, SNAPSHOT_DTYPES,
    TERM_RATE, COMP_RAISE_PCT, NEW_HIRE_TERM_RATE, COLA_PCT, CFG
)

# Add additional constants separately
NEW_HIRE_TERMINATION_RATE = 'new_hire_termination_rate'
COLA_PCT = 'cola_pct'
CFG = 'cfg'
TERM_RATE = 'term_rate'
COMP_RAISE_PCT = 'comp_raise_pct'

# Legacy imports for compatibility
try:
    from .event_log import EVT_HIRE, EVT_TERM, EVT_COMP, EVT_NEW_HIRE_TERM, EVENT_COLS
except ImportError:
    # Define fallbacks if run standalone or structure changes
    EVT_HIRE, EVT_TERM, EVT_COMP = "EVT_HIRE", "EVT_TERM", "EVT_COMP"
    EVENT_COLS = [
        "event_id",
        "event_time",
        "employee_id",
        "event_type",
        "value_num",
        "value_json",
        "meta",
    ]

logger = logging.getLogger(__name__)

# --- Legacy API ---
SNAPSHOT_COLS = [
    EMP_ID,          # Ensure EMP_ID is part of the columns
    EMP_HIRE_DATE,
    EMP_BIRTH_DATE,
    EMP_LEVEL,
    EMP_GROSS_COMP,  # Was "current_comp", maps to gross compensation
    EMP_TERM_DATE,   # Was "term_date"
    EMP_ACTIVE,        # pandas BooleanDtype (True/False/NA) - snapshot specific
    EMP_DEFERRAL_RATE, # Was "employee_deferral_rate"
    EMP_TENURE_BAND,   # Snapshot specific for grouping/logic
    EMP_TENURE,      # Standardized tenure column
    TERM_RATE,
    COMP_RAISE_PCT,
    NEW_HIRE_TERM_RATE,
    COLA_PCT,
    CFG,
    NEW_HIRE_TERMINATION_RATE,
    COLA_PCT,
    CFG,
    TERM_RATE,
    COMP_RAISE_PCT
]

# Corresponding Pandas dtypes using nullable types where appropriate
SNAPSHOT_DTYPES = {
    EMP_ID: pd.StringDtype(),           # Add dtype for EMP_ID
    EMP_HIRE_DATE: "datetime64[ns]",
    EMP_BIRTH_DATE: "datetime64[ns]",
    EMP_LEVEL: pd.Int64Dtype(),          # Nullable string
    EMP_GROSS_COMP: pd.Float64Dtype(),  # Nullable float, was "current_comp"
    EMP_TERM_DATE: "datetime64[ns]",     # Stays datetime, NaT represents null
    "active": pd.BooleanDtype(),         # Nullable boolean - snapshot specific
    EMP_DEFERRAL_RATE: pd.Float64Dtype(),# Was "employee_deferral_rate"
    EMP_TENURE_BAND: pd.StringDtype(),     # Snapshot specific
    EMP_TENURE: 'float64',               # Standardized tenure dtype
    TERM_RATE: pd.Float64Dtype(),
    COMP_RAISE_PCT: pd.Float64Dtype(),
    NEW_HIRE_TERM_RATE: pd.Float64Dtype(),
    COLA_PCT: pd.Float64Dtype(),
    CFG: pd.StringDtype(),
    NEW_HIRE_TERMINATION_RATE: pd.Float64Dtype(),
    COLA_PCT: pd.Float64Dtype(),
    CFG: pd.StringDtype(),
    TERM_RATE: pd.Float64Dtype(),
    COMP_RAISE_PCT: pd.Float64Dtype()
}

# --- Helper Functions ---

def _get_first_event(events: pd.DataFrame, event_type: Union[str, List[str]]) -> pd.DataFrame:
    """
    Returns the first occurrence of the given event_type for each employee.
    """
    if isinstance(event_type, str):
        events_of_type = events[events["event_type"] == event_type]
    else: # it's a list
        events_of_type = events[events["event_type"].isin(event_type)]
    return events_of_type.drop_duplicates(subset=EMP_ID, keep="first")

def _get_last_event(events: pd.DataFrame, event_type: Union[str, List[str]]) -> pd.DataFrame:
    """
    Returns the last occurrence of the given event_type for each employee.
    """
    if isinstance(event_type, str):
        events_of_type = events[events["event_type"] == event_type]
    else: # it's a list
        events_of_type = events[events["event_type"].isin(event_type)]
    return events_of_type.drop_duplicates(subset=EMP_ID, keep="last")

def _assign_tenure_band(tenure: float) -> str:
    """
    Assigns a tenure band label given a tenure in years.
    """
    if pd.isna(tenure):
        return pd.NA
    if tenure < 1:
        return '<1'  # Standardized to match hazard table format
    elif tenure < 3:
        return '1-3'
    elif tenure < 5:
        return '3-5'
    elif tenure < 10:
        return '5-10'
    elif tenure < 15:
        return '10-15'
    else:
        return '15+'

def _ensure_columns_and_types(df: pd.DataFrame, columns: list, dtypes: dict) -> pd.DataFrame:
    """
    Ensures DataFrame has all specified columns with correct dtypes, adding NA columns if missing.
    For numeric columns, uses np.nan for missing values to avoid astype errors.
    """
    for col, dtype in dtypes.items():
        if col not in df.columns:
            if pd.api.types.is_datetime64_any_dtype(dtype):
                df[col] = pd.NaT
            elif dtype == pd.BooleanDtype():
                df[col] = pd.NA
            elif pd.api.types.is_numeric_dtype(dtype):
                # For configuration columns, use default values from config
                if col == TERM_RATE:
                    df[col] = 0.15  # Default term rate
                elif col == COMP_RAISE_PCT:
                    df[col] = 0.03  # Default comp raise
                elif col == NEW_HIRE_TERM_RATE:
                    df[col] = 0.25  # Default new hire term rate
                elif col == COLA_PCT:
                    df[col] = 0.02  # Default COLA
                else:
                    df[col] = np.nan
            else:
                df[col] = pd.NA
    df = df[columns]
    # Replace pd.NA with np.nan in numeric columns before astype
    for col, dtype in dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype) and col in df.columns:
            df[col] = df[col].replace({pd.NA: np.nan})
    df = df.astype(dtypes)
    return df

def _extract_hire_details(hire_events: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts static details (role, birth_date) from hire event records.
    Assumes these details are stored in the 'value_json' field of hire events.

    Args:
        hire_events: DataFrame containing only the first 'hire' event per employee.

    Returns:
        A DataFrame indexed by EMP_ID with columns 'role', 'birth_date'.
    """
    details = []
    if hire_events.empty:
        return pd.DataFrame(
            columns=[EMP_ROLE, EMP_BIRTH_DATE], index=pd.Index([], name=EMP_ID)
        )

    for index, event in hire_events.iterrows():
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

def build_full(events: pd.DataFrame, snapshot_year: int) -> pd.DataFrame:
    """
    Builds a complete employee snapshot from the beginning based on all events provided.
    Recommended only for bootstrapping or testing on smaller datasets.

    Args:
        events: DataFrame containing all relevant events, expected to conform
                to EVENT_COLS and schema.

    Returns:
        A pandas DataFrame representing the snapshot state, indexed by EMP_ID.
    """
    # Return empty, correctly typed DF if no events
    if events.empty:
        logger.warning("build_full called with empty events DataFrame.")
        empty_snap = pd.DataFrame(columns=SNAPSHOT_COLS)
        empty_snap = empty_snap.astype(SNAPSHOT_DTYPES)
        empty_snap.index.name = EMP_ID
        return empty_snap

    logger.info(f"Building full snapshot from {len(events)} events...")
    # Sort events primarily by time, secondarily by type (e.g., hire before comp on same timestamp)
    # Using type ignore because sort_values can handle list of bools for ascending
    events = events.sort_values(by=["event_time", "event_type"], ascending=[True, True])  # type: ignore

    # 1. Get First Hire event for each employee
    hires = _filter_events(events, EVT_HIRE).drop_duplicates(subset=EMP_ID, keep="first")
    if hires.empty:
        logger.warning("No hire events found in the provided log for build_full.")
        empty_snap = pd.DataFrame(columns=SNAPSHOT_COLS)
        empty_snap = empty_snap.astype(SNAPSHOT_DTYPES)
        empty_snap.index.name = EMP_ID
        return empty_snap

    # Extract static details (role, birth_date) from first hire events
    hire_details = _extract_hire_details(hires)  # Indexed by EMP_ID

    # Get hire dates
    hire_dates = hires.set_index(EMP_ID)["event_time"].rename(EMP_HIRE_DATE)

    # 2. Get Last Compensation event for each employee
    comps = _filter_events(events, EVT_COMP).drop_duplicates(subset=EMP_ID, keep="last")
    last_comp = comps.set_index(EMP_ID)["value_num"].rename(EMP_GROSS_COMP)

    # 3. Get Last Termination event for each employee
    terms = _filter_events(events, EVT_TERM).drop_duplicates(subset=EMP_ID, keep="last")
    last_term = terms.set_index(EMP_ID)["event_time"].rename(EMP_TERM_DATE)

    # 4. Assemble the snapshot - start with essential hire info (hire date)
    snapshot_df = pd.DataFrame(hire_dates)  # Index = EMP_ID, Col = hire_date
    # Merge other details using left joins to keep all hired employees
    snapshot_df = snapshot_df.merge(hire_details, left_index=True, right_index=True, how="left")
    snapshot_df = snapshot_df.merge(last_comp, left_index=True, right_index=True, how="left")
    snapshot_df = snapshot_df.merge(last_term, left_index=True, right_index=True, how="left")

    # 5. Calculate 'active' status
    # Active if term_date is NaT (Not a Time / Null)
    snapshot_df["active"] = snapshot_df[EMP_TERM_DATE].isna()

    # 6. Ensure all snapshot columns exist and have correct types
    snapshot_df = _enforce_snapshot_columns(snapshot_df)

    # Calculate tenure and tenure_band based on tenure at end of snapshot_year
    if not snapshot_df.empty:
        as_of = pd.Timestamp(f"{snapshot_year}-12-31")
        hire_dates = pd.to_datetime(snapshot_df[EMP_HIRE_DATE], errors="coerce")
        tenure_years = (as_of - hire_dates).dt.days / 365.25
        snapshot_df[EMP_TENURE] = tenure_years.round(3)
        snapshot_df[EMP_TENURE_BAND] = _calculate_tenure_band(snapshot_df[EMP_TENURE])
        for eid, hd, ty in zip(snapshot_df.index, hire_dates, tenure_years):
            logger.debug(f"[DEBUG tenure] {eid} hired {hd.date()}   as_of {as_of.date()}   years={ty:.2f}")
        def band(tenure):
            if pd.isna(tenure): return pd.NA
            if tenure < 1: return '<1'  # Standardized to match hazard table format
            elif tenure < 3: return '1-3'
            elif tenure < 5: return '3-5'
            elif tenure < 10: return '5-10'
            elif tenure < 15: return '10-15'
            else: return '15+'
        snapshot_df[EMP_TENURE_BAND] = snapshot_df[EMP_TENURE].map(band).astype(pd.StringDtype())
    else:
        snapshot_df[EMP_TENURE] = pd.NA
        snapshot_df[EMP_TENURE_BAND] = pd.NA

    # Ensure EMP_ID is a column for output/export
    snapshot_df[EMP_ID] = snapshot_df.index.astype(str)
    # Add simulation_year
    snapshot_df['simulation_year'] = snapshot_year
    # Select final columns in desired order and enforce final dtypes
    # Add EMP_TENURE and simulation_year to output columns and dtypes
    output_cols = SNAPSHOT_COLS + [EMP_TENURE, 'simulation_year']
    output_dtypes = {**SNAPSHOT_DTYPES, EMP_TENURE: 'float64', 'simulation_year': 'int64'}
    snapshot_df = snapshot_df[output_cols]  # Select and order columns
    snapshot_df = snapshot_df.astype(output_dtypes)  # Enforce dtypes
    snapshot_df.index.name = EMP_ID  # Ensure index name is set

    logger.info(f"Full snapshot built. Shape: {snapshot_df.shape}")
    return snapshot_df


def update(
    prev_snapshot: pd.DataFrame, new_events: pd.DataFrame, snapshot_year: int
) -> pd.DataFrame:
    """
    Updates an existing snapshot based on new events that occurred *since*
    the previous snapshot was generated. Designed for efficiency.

    Args:
        prev_snapshot: The snapshot DataFrame from the previous state, indexed by EMP_ID.
                       Expected to conform to SNAPSHOT_COLS/DTYPES.
        new_events: DataFrame containing only the new events since the last snapshot.
                    Expected to conform to EVENT_COLS schema.
        snapshot_year: The year (int) for the new snapshot (used for tenure calculation).

    Returns:
        The updated snapshot DataFrame, indexed by EMP_ID.
    """
    if new_events.empty:
        logger.debug("Update called with no new events. Returning previous snapshot.")
        return prev_snapshot.astype(SNAPSHOT_DTYPES)

    # Ensure previous snapshot has the correct index name
    if prev_snapshot.index.name != EMP_ID:
        logger.warning(
            f"Previous snapshot index name is '{prev_snapshot.index.name}', expected '{EMP_ID}'. Attempting to proceed."
        )

    logger.info(
        f"Updating snapshot ({prev_snapshot.shape}) with {len(new_events)} new events..."
    )
    current_snapshot = prev_snapshot.copy()
    # Ensure simulation_year is set
    current_snapshot['simulation_year'] = snapshot_year

    # Ensure new events are sorted by time and type
    new_events = new_events.sort_values(by=["event_time", "event_type"], ascending=[True, True])

    # --- 1. Identify and Process New Hires ---
    first_hire_events_new = _get_first_event(new_events, EVT_HIRE)
    new_hire_ids = first_hire_events_new[
        ~first_hire_events_new[EMP_ID].isin(current_snapshot.index)
    ][EMP_ID].unique()

    if len(new_hire_ids) > 0:
        logger.debug(f"Processing {len(new_hire_ids)} new hires.")
        events_for_new_hires = new_events[new_events[EMP_ID].isin(new_hire_ids)]
        first_hire_events_for_new_filtered = _get_first_event(events_for_new_hires, EVT_HIRE)
        new_hire_details = _extract_hire_details(first_hire_events_for_new_filtered)
        last_comp_for_new = _get_last_event(events_for_new_hires, EVT_COMP).set_index(EMP_ID)["value_num"]
        last_term_for_new = _get_last_event(events_for_new_hires, EVT_TERM).set_index(EMP_ID)["event_time"]

        new_hire_base = pd.DataFrame(index=pd.Index(new_hire_ids, name=EMP_ID))
        new_hire_base = new_hire_base.merge(
            first_hire_events_for_new_filtered.set_index(EMP_ID)["event_time"].rename(EMP_HIRE_DATE),
            left_index=True, right_index=True, how="left"
        )
        new_hire_base = new_hire_base.merge(
            new_hire_details[[EMP_ROLE, EMP_BIRTH_DATE]],
            left_index=True, right_index=True, how="left"
        )
        new_hire_base = new_hire_base.merge(
            last_comp_for_new.rename(EMP_GROSS_COMP),
            left_index=True, right_index=True, how="left"
        )
        new_hire_base = new_hire_base.merge(
            last_term_for_new.rename(EMP_TERM_DATE),
            left_index=True, right_index=True, how="left"
        )
        new_hire_base["active"] = new_hire_base[EMP_TERM_DATE].isna()
        new_hire_base = _ensure_columns_and_types(new_hire_base, SNAPSHOT_COLS, SNAPSHOT_DTYPES)

        # Calculate tenure and tenure_band for new hires as of snapshot_year
        if not new_hire_base.empty:
            as_of = pd.Timestamp(f"{snapshot_year}-12-31")
            hire_dates = pd.to_datetime(new_hire_base[EMP_HIRE_DATE], errors="coerce")
            tenure_years = (as_of - hire_dates).dt.days / 365.25
            new_hire_base[EMP_TENURE] = tenure_years.round(3)
            new_hire_base[EMP_TENURE_BAND] = new_hire_base[EMP_TENURE].map(_assign_tenure_band).astype(pd.StringDtype())
        else:
            new_hire_base[EMP_TENURE] = pd.NA
            new_hire_base[EMP_TENURE_BAND] = pd.NA
        new_hire_base[EMP_ID] = new_hire_base.index.astype(str)
        new_hire_rows_df = new_hire_base[SNAPSHOT_COLS].astype(SNAPSHOT_DTYPES)
        dfs = [df for df in [current_snapshot, new_hire_rows_df] if not df.empty]
        if dfs:
            current_snapshot = pd.concat(dfs, verify_integrity=True, copy=False)
        else:
            current_snapshot = pd.DataFrame()
        logger.debug(f"Appended {len(new_hire_rows_df)} new hire rows to snapshot.")

        # Recompute tenure & tenure_band for the full updated snapshot
        as_of = pd.Timestamp(f"{snapshot_year}-12-31")
        hire_dates = pd.to_datetime(current_snapshot[EMP_HIRE_DATE], errors='coerce')
        tenure_years = (as_of - hire_dates).dt.days / 365.25
        current_snapshot[EMP_TENURE] = tenure_years.round(3)
        current_snapshot[EMP_TENURE_BAND] = current_snapshot[EMP_TENURE].map(_assign_tenure_band).astype(pd.StringDtype())

    # --- 2. Process Updates for Existing Employees ---
    existing_ids_in_batch = new_events[new_events[EMP_ID].isin(prev_snapshot.index)][EMP_ID].unique()
    if len(existing_ids_in_batch) > 0:
        logger.debug(
            f"Processing updates for {len(existing_ids_in_batch)} potentially existing employees found in new events."
        )
        comp_updates = new_events[new_events["event_type"] == EVT_COMP]
        if not comp_updates.empty:
            last_comp_updates = (
                comp_updates.sort_values("event_time").groupby(EMP_ID).tail(1)
            )
            comp_update_map = last_comp_updates.set_index(EMP_ID)["value_num"]
            active_employees = current_snapshot[current_snapshot["active"]].index
            valid_updates = comp_update_map[
                comp_update_map.index.isin(active_employees)
            ]
            current_snapshot.loc[valid_updates.index, EMP_GROSS_COMP] = valid_updates

        # Process termination updates
        term_updates = new_events[new_events["event_type"] == EVT_TERM]
        if not term_updates.empty:
            last_term_updates = (
                term_updates.sort_values("event_time").groupby(EMP_ID).tail(1)
            )
            term_update_map = last_term_updates.set_index(EMP_ID)["event_time"]
            current_snapshot.loc[term_update_map.index, EMP_TERM_DATE] = term_update_map
            # Update active status
            current_snapshot["active"] = current_snapshot[EMP_TERM_DATE].isna()

        # Recompute tenure and tenure_band for all employees
        as_of = pd.Timestamp(f"{snapshot_year}-12-31")
        hire_dates = pd.to_datetime(current_snapshot[EMP_HIRE_DATE], errors='coerce')
        tenure_years = (as_of - hire_dates).dt.days / 365.25
        current_snapshot[EMP_TENURE] = tenure_years.round(3)
        current_snapshot[EMP_TENURE_BAND] = current_snapshot[EMP_TENURE].map(_assign_tenure_band).astype(pd.StringDtype())

    # Ensure EMP_ID is a column for output/export
    current_snapshot[EMP_ID] = current_snapshot.index.astype(str)
    # Select final columns in desired order and enforce final dtypes
    output_cols = SNAPSHOT_COLS + [EMP_TENURE]
    output_dtypes = {**SNAPSHOT_DTYPES, EMP_TENURE: 'float64'}
    current_snapshot = current_snapshot[output_cols]  # Select and order columns
    current_snapshot = current_snapshot.astype(output_dtypes)  # Enforce dtypes
    current_snapshot.index.name = EMP_ID  # Ensure index name is set

    logger.info(f"Snapshot updated. Shape: {current_snapshot.shape}")
    return current_snapshot

    try:
        """
        Updates an existing snapshot based on new events that occurred *since*
        the previous snapshot was generated. Designed for efficiency.

        Args:
            prev_snapshot: The snapshot DataFrame from the previous state, indexed by EMP_ID.
                           Expected to conform to SNAPSHOT_COLS/DTYPES.
            new_events: DataFrame containing only the new events since the last snapshot.
                        Expected to conform to EVENT_COLS schema.

        Returns:
            The updated snapshot DataFrame, indexed by EMP_ID.
        """
        if new_events.empty:
            logger.debug("Update called with no new events. Returning previous snapshot.")
            # Ensure previous snapshot conforms before returning, just in case
            return prev_snapshot.astype(SNAPSHOT_DTYPES)

        # Ensure previous snapshot has the correct index name
        if prev_snapshot.index.name != EMP_ID:
            logger.warning(
                f"Previous snapshot index name is '{prev_snapshot.index.name}', expected '{EMP_ID}'. Attempting to proceed."
            )
            # Optionally rename: prev_snapshot.index.name = EMP_ID

        logger.info(
            f"Updating snapshot ({prev_snapshot.shape}) with {len(new_events)} new events..."
        )
        current_snapshot = prev_snapshot.copy()

        # Ensure new events are sorted by time and type
        # Using type ignore because sort_values can handle list of bools for ascending
        new_events = new_events.sort_values(by=["event_time", "event_type"], ascending=[True, True])  # type: ignore

        # --- 1. Identify and Process New Hires ---
        # Find first hire event for each ID within the new_events batch
        first_hire_events_new = new_events[
            new_events["event_type"] == EVT_HIRE
        ].drop_duplicates(subset=EMP_ID, keep="first")
        # Identify IDs present in these hire events but NOT in the previous snapshot's index
        new_hire_ids = first_hire_events_new[
            ~first_hire_events_new[EMP_ID].isin(current_snapshot.index)
        ][EMP_ID].unique()

        if len(new_hire_ids) > 0:
            logger.debug(f"Processing {len(new_hire_ids)} new hires.")
            # Filter all events pertaining *only* to these new hires *within the new_events batch*
            events_for_new_hires = new_events[new_events[EMP_ID].isin(new_hire_ids)]

            # Extract static details (role, birth_date) from their first hire event
            first_hire_events_for_new_filtered = events_for_new_hires[
                events_for_new_hires["event_type"] == EVT_HIRE
            ].drop_duplicates(subset=EMP_ID, keep="first")
            new_hire_details = _extract_hire_details(
                first_hire_events_for_new_filtered
            )  # Indexed by EMP_ID

            # Get last comp and term events *within the new_events batch* for the new hires
            last_comp_for_new = (
                events_for_new_hires[events_for_new_hires["event_type"] == EVT_COMP]
                .drop_duplicates(subset=EMP_ID, keep="last")
                .set_index(EMP_ID)["value_num"]
            )
            last_term_for_new = (
                events_for_new_hires[events_for_new_hires["event_type"] == EVT_TERM]
                .drop_duplicates(subset=EMP_ID, keep="last")
                .set_index(EMP_ID)["event_time"]
            )

            # Assemble rows for new hires
            new_hire_base = pd.DataFrame(index=pd.Index(new_hire_ids, name=EMP_ID))
            # Merge hire date from the filtered first hire events
            new_hire_base = new_hire_base.merge(
                first_hire_events_for_new_filtered.set_index(EMP_ID)["event_time"].rename(
                    EMP_HIRE_DATE
                ),
                left_index=True,
                right_index=True,
                how="left",
            )
            new_hire_base = new_hire_base.merge(
                new_hire_details[[EMP_ROLE, EMP_BIRTH_DATE]],
                left_index=True,
                right_index=True,
                how="left",
            )  # Details already indexed
            new_hire_base = new_hire_base.merge(
                last_comp_for_new.rename(EMP_GROSS_COMP),
                left_index=True,
                right_index=True,
                how="left",
            )
            new_hire_base = new_hire_base.merge(
                last_term_for_new.rename(EMP_TERM_DATE),
                left_index=True,
                right_index=True,
                how="left",
            )
            new_hire_base["active"] = new_hire_base[EMP_TERM_DATE].isna()

            # Ensure all columns and dtypes match the main snapshot schema
            for col, dtype in SNAPSHOT_DTYPES.items():
                if col not in new_hire_base.columns:
                    if pd.api.types.is_datetime64_any_dtype(dtype):
                        new_hire_base[col] = pd.NaT
                    elif dtype == pd.BooleanDtype():
                        new_hire_base[col] = pd.NA
                    elif pd.api.types.is_numeric_dtype(dtype):
                        new_hire_base[col] = np.nan  # Use np.nan for Float64Dtype etc.
                    else:
                        new_hire_base[col] = pd.NA  # Use pd.NA for StringDtype

            # Calculate tenure and tenure_band for new hires as of snapshot_year
            if not new_hire_base.empty:
                # Use end-of-year of the snapshot, not the max hire date!
                as_of = pd.Timestamp(f"{snapshot_year}-12-31")
                hire_dates = pd.to_datetime(new_hire_base[EMP_HIRE_DATE], errors="coerce")
                tenure_years = (as_of - hire_dates).dt.days / 365.25
                # DEBUG: log each new hire's tenure for tracing
                for eid, hd, ty in zip(new_hire_base.index, hire_dates, tenure_years):
                    logger.debug(f"[DEBUG tenure] {eid} hired {hd.date()}  ￂￂ as_of {as_of.date()}  ￂￂ years={ty:.2f}")
                def band(tenure):
                    if pd.isna(tenure): return pd.NA
                    if tenure < 1: return '<1'  # Standardized to match hazard table format
                    elif tenure < 3: return '1-3'
                    elif tenure < 5: return '3-5'
                    else: return '5+'
                new_hire_base[EMP_TENURE_BAND] = new_hire_base[EMP_TENURE].map(band).astype(pd.StringDtype())
            else:
                new_hire_base[EMP_TENURE_BAND] = pd.NA
            new_hire_base[EMP_TENURE] = pd.NA
            # Ensure EMP_ID is a column for output/export
            new_hire_base[EMP_ID] = new_hire_base.index.astype(str)
            # Replace pd.NA with np.nan in numeric columns to avoid TypeError
            for col, dtype in SNAPSHOT_DTYPES.items():
                if pd.api.types.is_numeric_dtype(dtype) and col in new_hire_base.columns:
                    new_hire_base[col] = new_hire_base[col].replace({pd.NA: np.nan})
            new_hire_rows_df = new_hire_base[SNAPSHOT_COLS].astype(SNAPSHOT_DTYPES)

            # Append new hires to the main snapshot
            # ignore_index=False keeps the EMP_ID index
            # verify_integrity=True checks for duplicate indices (shouldn't happen if logic is right)
            dfs = [df for df in [current_snapshot, new_hire_rows_df] if not df.empty]
            if dfs:
                current_snapshot = pd.concat(dfs, verify_integrity=True, copy=False)
            else:
                current_snapshot = pd.DataFrame()
            logger.debug(f"Appended {len(new_hire_rows_df)} new hire rows to snapshot.")

            # Recompute tenure & tenure_band for the full updated snapshot
            as_of = pd.Timestamp(f"{snapshot_year}-12-31")
            hire_dates = pd.to_datetime(current_snapshot[EMP_HIRE_DATE], errors='coerce')
            tenure_years = (as_of - hire_dates).dt.days / 365.25
            current_snapshot[EMP_TENURE] = tenure_years.round(3)
            def band(tenure):
                if pd.isna(tenure):    return pd.NA
                elif tenure < 1:       return '<1'  # Standardized to match hazard table format
                elif tenure < 3:       return '1-3'
                elif tenure < 5:       return '3-5'
                else:                  return '5+'
            current_snapshot[EMP_TENURE_BAND] = current_snapshot[EMP_TENURE].map(band).astype(pd.StringDtype())

        # --- 2. Process Updates for Existing Employees ---
        # Find IDs in new_events that existed in the previous snapshot
        existing_ids_in_batch = new_events[new_events[EMP_ID].isin(prev_snapshot.index)][
            EMP_ID
        ].unique()

        if len(existing_ids_in_batch) > 0:
            logger.debug(
                f"Processing updates for {len(existing_ids_in_batch)} potentially existing employees found in new events."
            )
            new_events[new_events[EMP_ID].isin(existing_ids_in_batch)]

            # Process compensation updates
            comp_updates = new_events[new_events["event_type"] == EVT_COMP]
            if not comp_updates.empty:
                # Sort by event time and get the last update for each employee
                last_comp_updates = (
                    comp_updates.sort_values("event_time").groupby(EMP_ID).tail(1)
                )
                comp_update_map = last_comp_updates.set_index(EMP_ID)["value_num"]
                # Only update compensation for active employees
                active_employees = current_snapshot[current_snapshot["active"]].index
                valid_updates = comp_update_map[
                    comp_update_map.index.isin(active_employees)
                ]
                if not valid_updates.empty:
                    current_snapshot.loc[valid_updates.index, EMP_GROSS_COMP] = (
                        valid_updates
                    )
                    logger.debug(
                        f"Applied {len(valid_updates)} compensation updates to active employees."
                    )
                else:
                    logger.debug(
                        "No compensation updates applied - all affected employees were terminated."
                    )

            # Process termination updates
            term_updates = new_events[new_events["event_type"] == EVT_TERM]
            if not term_updates.empty:
                # Sort by event time and get the last update for each employee
                last_term_updates = (
                    term_updates.sort_values("event_time").groupby(EMP_ID).tail(1)
                )
                term_update_map = last_term_updates.set_index(EMP_ID)["event_time"]
                # Update termination date and active status
                current_snapshot.loc[term_update_map.index, EMP_TERM_DATE] = term_update_map
                current_snapshot.loc[term_update_map.index, "active"] = False
                logger.debug(
                    f"Applied {len(term_update_map)} termination updates to existing employees."
                )

        # Ensure final dtypes just in case updates changed them (e.g., float -> int)
        # Although using nullable dtypes should prevent most unwanted casts.
        current_snapshot = current_snapshot.astype(SNAPSHOT_DTYPES)

        # ----------------------------
        # Recompute tenure_band for _all_ employees at year-end
        # ----------------------------
        as_of = pd.Timestamp(f"{snapshot_year}-12-31")
        tenure_years = (
            (as_of - pd.to_datetime(current_snapshot[EMP_HIRE_DATE]))
            .dt.days
            / 365.25
        )
        def _band(t):
            if pd.isna(t):
                return pd.NA
            if t < 1:
                return '<1'
            elif t < 3:
                return '1-3'
            elif t < 5:
                return '3-5'
            elif t < 10:
                return '5-10'
            elif t < 15:
                return '10-15'
            else:
                return '15+'
        current_snapshot[EMP_TENURE_BAND] = tenure_years.map(_band).astype(pd.StringDtype())

        # Sanity check: ensure no duplicate indices after update
        dupes = current_snapshot.index.duplicated().sum()
        if dupes:
            logger.warning(f"Snapshot.update: found {dupes} duplicate indices, dropping extras.")
            # Enforce final dtypes and return
            # Add EMP_TENURE to output columns and dtypes
            output_cols = SNAPSHOT_COLS + [EMP_TENURE]
            output_dtypes = {**SNAPSHOT_DTYPES, EMP_TENURE: 'float64'}
            current_snapshot = current_snapshot[output_cols]
            current_snapshot = current_snapshot.astype(output_dtypes)
            current_snapshot.index.name = EMP_ID
            logger.info(f"Snapshot update complete. New shape: {current_snapshot.shape}")
            return current_snapshot
        return current_snapshot

    except Exception as e:
        logger.error(f"Exception in snapshot.update: {e}", exc_info=True)
        # Return empty DataFrame with correct columns and dtypes
        empty = pd.DataFrame(columns=SNAPSHOT_COLS).astype(SNAPSHOT_DTYPES)
        empty.index.name = EMP_ID
        return empty



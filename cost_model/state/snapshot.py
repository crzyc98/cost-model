# cost_model/state/snapshot.py
"""
Functions to build and update the workforce snapshot DataFrame from the event log.
Provides both a full rebuild capability (for bootstrapping/testing) and
an incremental update capability (for efficient year-over-year processing).
"""

import logging
import pandas as pd
import json  # Needed for parsing value_json

# --- Dependencies ---
# Assumed defined in cost_model/state/event_log.py
try:
    from .event_log import EVT_HIRE, EVT_TERM, EVT_COMP, EVENT_COLS
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


# Assumed defined in cost_model/utils/columns.py
try:
    from ..utils.columns import (
        EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_ROLE, 
        EMP_GROSS_COMP, EMP_TERM_DATE, EMP_DEFERRAL_RATE
    )
except ImportError:
    print(
        "Warning: Could not import EMP_ID or other constants from utils.columns. Defaulting..."
    )
    EMP_ID = "employee_id"
    EMP_HIRE_DATE = "hire_date"
    EMP_BIRTH_DATE = "birth_date"
    EMP_ROLE = "role"
    EMP_GROSS_COMP = "current_comp" # or some other default
    EMP_TERM_DATE = "term_date"
    EMP_DEFERRAL_RATE = "employee_deferral_rate"

logger = logging.getLogger(__name__)

# --- Snapshot Schema Definition ---
# Columns and their desired order in the snapshot
SNAPSHOT_COLS = [
    EMP_ID,          # Ensure EMP_ID is part of the columns
    EMP_HIRE_DATE,
    EMP_BIRTH_DATE,
    EMP_ROLE,
    EMP_GROSS_COMP,  # Was "current_comp", maps to gross compensation
    EMP_TERM_DATE,   # Was "term_date"
    "active",        # pandas BooleanDtype (True/False/NA) - snapshot specific
    EMP_DEFERRAL_RATE, # Was "employee_deferral_rate"
    "tenure_band",   # Snapshot specific for grouping/logic
]

# Corresponding Pandas dtypes using nullable types where appropriate
SNAPSHOT_DTYPES = {
    EMP_ID: pd.StringDtype(),           # Add dtype for EMP_ID
    EMP_HIRE_DATE: "datetime64[ns]",
    EMP_BIRTH_DATE: "datetime64[ns]",
    EMP_ROLE: pd.StringDtype(),          # Nullable string
    EMP_GROSS_COMP: pd.Float64Dtype(),  # Nullable float, was "current_comp"
    EMP_TERM_DATE: "datetime64[ns]",     # Stays datetime, NaT represents null
    "active": pd.BooleanDtype(),         # Nullable boolean - snapshot specific
    EMP_DEFERRAL_RATE: pd.Float64Dtype(),# Was "employee_deferral_rate"
    "tenure_band": pd.StringDtype(),     # Snapshot specific
}

# --- Helper Function ---


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


# --- Snapshot Building Functions ---


def build_full(events: pd.DataFrame) -> pd.DataFrame:
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
    hires = events[events["event_type"] == EVT_HIRE].drop_duplicates(
        subset=EMP_ID, keep="first"
    )
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
    comps = events[events["event_type"] == EVT_COMP].drop_duplicates(
        subset=EMP_ID, keep="last"
    )
    last_comp = comps.set_index(EMP_ID)["value_num"].rename(
        EMP_GROSS_COMP
    )  # Assumes comp is in value_num

    # 3. Get Last Termination event for each employee
    terms = events[events["event_type"] == EVT_TERM].drop_duplicates(
        subset=EMP_ID, keep="last"
    )
    last_term = terms.set_index(EMP_ID)["event_time"].rename(EMP_TERM_DATE)

    # 4. Assemble the snapshot - start with essential hire info (hire date)
    snapshot_df = pd.DataFrame(hire_dates)  # Index = EMP_ID, Col = hire_date
    # Merge other details using left joins to keep all hired employees
    snapshot_df = snapshot_df.merge(
        hire_details, left_index=True, right_index=True, how="left"
    )
    snapshot_df = snapshot_df.merge(
        last_comp, left_index=True, right_index=True, how="left"
    )
    snapshot_df = snapshot_df.merge(
        last_term, left_index=True, right_index=True, how="left"
    )

    # 5. Calculate 'active' status
    # Active if term_date is NaT (Not a Time / Null)
    snapshot_df["active"] = snapshot_df[EMP_TERM_DATE].isna()

    # 6. Ensure all snapshot columns exist and have correct types
    for col, dtype in SNAPSHOT_DTYPES.items():
        if col not in snapshot_df.columns:
            logger.warning(
                f"Column '{col}' missing after build_full merge, adding as NA."
            )
            # Add column with appropriate NA value based on dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                snapshot_df[col] = pd.NaT
            elif dtype == pd.BooleanDtype():  # Check specific nullable boolean type
                snapshot_df[col] = pd.NA
            elif pd.api.types.is_numeric_dtype(dtype):  # Includes Float64Dtype
                snapshot_df[col] = pd.NA  # Use pd.NA for nullable numeric types
            else:  # StringDtype or other objects
                snapshot_df[col] = pd.NA

    # Calculate tenure_band based on tenure at end of latest event year
    if not snapshot_df.empty:
        reference_date = pd.to_datetime(snapshot_df[EMP_HIRE_DATE]).max()
        if pd.notnull(reference_date):
            # Use Dec 31 of the latest event year as reference
            year = reference_date.year
            as_of = pd.Timestamp(f"{year}-12-31")
            tenure_years = (as_of - pd.to_datetime(snapshot_df[EMP_HIRE_DATE])).dt.days / 365.25
            def band(tenure):
                if pd.isna(tenure): return pd.NA
                if tenure < 1: return '<1'
                elif tenure < 3: return '1-3'
                elif tenure < 5: return '3-5'
                else: return '5+'
            snapshot_df['tenure_band'] = tenure_years.map(band).astype(pd.StringDtype())
        else:
            snapshot_df['tenure_band'] = pd.NA
    else:
        snapshot_df['tenure_band'] = pd.NA
    # Ensure EMP_ID is a column for output/export
    snapshot_df[EMP_ID] = snapshot_df.index.astype(str)
    # Select final columns in desired order and enforce final dtypes
    snapshot_df = snapshot_df[SNAPSHOT_COLS]  # Select and order columns
    snapshot_df = snapshot_df.astype(SNAPSHOT_DTYPES)  # Enforce dtypes
    snapshot_df.index.name = EMP_ID  # Ensure index name is set

    logger.info(f"Full snapshot built. Shape: {snapshot_df.shape}")
    return snapshot_df


def update(prev_snapshot: pd.DataFrame, new_events: pd.DataFrame) -> pd.DataFrame:
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
                    new_hire_base[col] = pd.NA  # Use pd.NA for Float64Dtype etc.
                else:
                    new_hire_base[col] = pd.NA  # Use pd.NA for StringDtype

        # Calculate tenure_band for new hires
        if not new_hire_base.empty:
            reference_date = pd.to_datetime(new_hire_base[EMP_HIRE_DATE]).max()
            if pd.notnull(reference_date):
                year = reference_date.year
                as_of = pd.Timestamp(f"{year}-12-31")
                tenure_years = (as_of - pd.to_datetime(new_hire_base[EMP_HIRE_DATE])).dt.days / 365.25
                def band(tenure):
                    if pd.isna(tenure): return pd.NA
                    if tenure < 1: return '<1'
                    elif tenure < 3: return '1-3'
                    elif tenure < 5: return '3-5'
                    else: return '5+'
                new_hire_base['tenure_band'] = tenure_years.map(band).astype(pd.StringDtype())
            else:
                new_hire_base['tenure_band'] = pd.NA
        else:
            new_hire_base['tenure_band'] = pd.NA
        # Ensure EMP_ID is a column for output/export
        new_hire_base[EMP_ID] = new_hire_base.index.astype(str)
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

    # Sanity check: ensure no duplicate indices after update
    dupes = current_snapshot.index.duplicated().sum()
    if dupes:
        logger.warning(f"Snapshot.update: found {dupes} duplicate indices, dropping extras.")
        current_snapshot = current_snapshot[~current_snapshot.index.duplicated(keep='first')]

    logger.info(f"Snapshot update complete. New shape: {current_snapshot.shape}")
    return current_snapshot

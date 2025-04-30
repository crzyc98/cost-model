"""
utils/data_processing.py - Functions for loading and cleaning census data.
"""

import pandas as pd
import os
import logging
import numpy as np
from datetime import datetime
from utils.columns import (
    RAW_TO_STD_COLS,
    DATE_COLS,
    EMP_SSN,
    EMP_GROSS_COMP,
    STATUS_COL,
    EMP_HIRE_DATE,
    EMP_TERM_DATE,
    EMP_BIRTH_DATE,
)
# Avoid enum name collision: alias status_enums EmploymentStatus
from utils.status_enums import EmploymentStatus as PhaseStatus
# Use explicit constants for active vs inactive states
from utils.constants import ACTIVE_STATUS, INACTIVE_STATUS

logger = logging.getLogger(__name__)

def _infer_plan_year_end(filepath: str) -> pd.Timestamp:
    fname = os.path.basename(filepath)
    year_str = fname.rsplit('_', 1)[-1].split('.', 1)[0]
    now = datetime.now().year
    try:
        y = int(year_str)
        if 1980 < y <= now + 1:
            return pd.Timestamp(f"{y}-12-31")
    except ValueError:
        pass
    return pd.Timestamp(f"{now-1}-12-31")


def load_and_clean_census(filepath, expected_cols):
    """Loads a census file and performs basic cleaning."""
    try:
        df = pd.read_csv(filepath)
        logger.info("Loaded census from %s with %d rows", filepath, len(df))
    except FileNotFoundError:
        logger.error("File not found: %s", filepath)
        return None
    except Exception as e:
        logger.error("Error loading %s: %s", filepath, e)
        return None

    # Map raw → standard names & drop duplicates
    df = df.rename(columns=RAW_TO_STD_COLS).drop_duplicates()
    # Check required after renaming
    if not all(col in df.columns for col in expected_cols['required']):
        logger.error("Missing required columns in census:")
        logger.error("  required: %s", expected_cols['required'])
        logger.error("  found:    %s", list(df.columns))
        return None

    # Parse dates only if present
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Remove any rows with missing SSN
    if EMP_SSN in df.columns:
        df = df[df[EMP_SSN].notna()]

    # Translate contributions to standard names
    contribs = expected_cols.get('contributions', [])
    std_contribs = [RAW_TO_STD_COLS.get(c, c) for c in contribs]
    numeric_cols = [EMP_GROSS_COMP] + std_contribs

    # Numeric coercion with warning
    for col in numeric_cols:
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            after = df[col].isna().sum()
            if after > before:
                logger.warning("Coercion created %d new NaNs in %s", after - before, col)

    # Convert SSN to string
    if EMP_SSN in df.columns:
        df[EMP_SSN] = df[EMP_SSN].astype(str)

    # Plan year end date inference
    df['plan_year_end_date'] = _infer_plan_year_end(filepath)

    # Drop any unmapped raw columns
    keep = set(RAW_TO_STD_COLS.values()) | set(expected_cols['required']) | {"plan_year_end_date"}
    df = df.loc[:, df.columns.intersection(keep)]
    return df


def assign_employment_status(df, start_year):
    """Assign employment_status and status columns using PhaseStatus enum."""
    df_copy = df.copy()

    # Initialize all as INACTIVE phase
    status = pd.Series(
        np.full(len(df_copy), PhaseStatus.INACTIVE.value, dtype=object),
        index=df_copy.index
    )

    # Determine phases
    hyear = df_copy[EMP_HIRE_DATE].dt.year
    tdate = df_copy[EMP_TERM_DATE]
    mask_not_hired = hyear > start_year
    mask_new_hire = hyear == start_year
    mask_pre_term = (tdate.notna()) & (tdate.dt.year < start_year)
    mask_active_cont = (hyear < start_year) & (tdate.isna())
    mask_active_init = (hyear == start_year) & (tdate.isna())

    status[mask_not_hired]     = PhaseStatus.NOT_HIRED.value
    status[mask_new_hire]      = PhaseStatus.NEW_HIRE.value
    status[mask_pre_term]      = PhaseStatus.PREV_TERMINATED.value
    status[mask_active_cont]   = PhaseStatus.ACTIVE_CONTINUOUS.value
    status[mask_active_init]   = PhaseStatus.ACTIVE_INITIAL.value

    df_copy['employment_status'] = status

    # Map to active vs inactive categories
    df_copy[STATUS_COL] = np.where(
        status.isin({PhaseStatus.ACTIVE_INITIAL.value, PhaseStatus.ACTIVE_CONTINUOUS.value, PhaseStatus.NEW_HIRE.value}),
        ACTIVE_STATUS,
        INACTIVE_STATUS
    )
    return df_copy

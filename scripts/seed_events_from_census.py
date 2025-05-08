#!/usr/bin/env python3
import json
import logging
from pathlib import Path
import pandas as pd
import uuid

from cost_model.state.event_log import EVT_HIRE, EVT_COMP, EVT_TERM, EVENT_COLS, EVENT_PANDAS_DTYPES
from cost_model.utils.columns import (
    EMP_ID,
    EMP_HIRE_DATE,
    EMP_BIRTH_DATE,
    EMP_TERM_DATE,
    EMP_GROSS_COMP
)

def seed_from_census(
    input_path: Path,
    output_path: Path,
):
    # 1. Load census
    logger.info(f"Reading input file: {input_path}")
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:  # Assume CSV
        df = pd.read_csv(input_path, parse_dates=[EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_TERM_DATE])
    logger.info(f"Loaded {len(df)} records from census")

    records = []
    for _, row in df.iterrows():
        emp_id = str(row[EMP_ID])
        hire_date = pd.to_datetime(row[EMP_HIRE_DATE])
        birth_date = pd.to_datetime(row[EMP_BIRTH_DATE])

        # --- HIRE Event ---
        hire_details_dict = {
            "birth_date": birth_date.strftime('%Y-%m-%d') if pd.notna(birth_date) else None,
            "role": row.get("employee_role", None)
        }
        records.append({
            "event_id": str(uuid.uuid4()),
            "event_time": hire_date,
            EMP_ID: emp_id,
            "event_type": EVT_HIRE,
            "value_num": None,
            "value_json": json.dumps(hire_details_dict),
            "meta": None
        })

        # --- COMP Event ---
        comp_value = row.get(EMP_GROSS_COMP)
        records.append({
            "event_id": str(uuid.uuid4()),
            "event_time": pd.Timestamp("2024-12-31"),
            EMP_ID: emp_id,
            "event_type": EVT_COMP,
            "value_num": float(comp_value) if pd.notna(comp_value) else None,
            "value_json": None,
            "meta": None
        })

        # --- TERM Event ---
        term_date = row.get(EMP_TERM_DATE)
        if pd.notna(term_date) and pd.Timestamp("2024-01-01") <= term_date <= pd.Timestamp("2024-12-31"):
            records.append({
                "event_id": str(uuid.uuid4()),
                "event_time": pd.to_datetime(term_date),
                EMP_ID: emp_id,
                "event_type": EVT_TERM,
                "value_num": None,
                "value_json": None,
                "meta": None
            })

    # 3. Assemble DataFrame
    events = pd.DataFrame(records)
    
    # Ensure all columns from EVENT_COLS are included
    for col in EVENT_COLS:
        if col not in events.columns:
            dtype = event_log.EVENT_PANDAS_DTYPES.get(col)
            if pd.api.types.is_datetime64_any_dtype(dtype):
                events[col] = pd.NaT
            else:
                events[col] = None

    # Ensure correct column order and dtypes
    events = events[EVENT_COLS].astype(EVENT_PANDAS_DTYPES)

    # 4. Sort, dedupe if you like, then write out
    events = events.sort_values(["event_time", EMP_ID])
    events.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(events)} events to {output_path}")

if __name__ == "__main__":
    import argparse

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Seed events from census data")
    parser.add_argument("--census", type=str, required=True, help="Path to census CSV or Parquet file")
    parser.add_argument("--out", type=str, required=True, help="Output path for event log Parquet")
    args = parser.parse_args()

    # Run the main function
    seed_from_census(
        input_path=Path(args.census),
        output_path=Path(args.out)
    )
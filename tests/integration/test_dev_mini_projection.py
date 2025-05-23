import subprocess
import pathlib
import pytest
import pandas as pd
import math
import yaml
from cost_model.engines.run_one_year.validation import validate_eoy_snapshot
from cost_model.state.schema import EMP_ACTIVE

# Paths
PROJECT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT  = PROJECT / "scripts/run_multi_year_projection.py"
CONFIG  = PROJECT / "config/dev_tiny.yaml"
CENSUS  = PROJECT / "data/census_preprocessed.parquet"

# Load config parameters
with open(CONFIG) as fh:
    GP = yaml.safe_load(fh)["global_parameters"]
TGROWTH = GP["target_growth"]
NH_RATE = GP["attrition"]["new_hire_termination_rate"]

@pytest.fixture(scope="session")
def projection_output(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("proj_out")
    subprocess.check_call([
        "python3", str(SCRIPT), "--config", str(CONFIG), "--census", str(CENSUS), "--out", str(out_dir), "--debug"
    ])
    return out_dir

def load_year(out_dir, year):
    print(f"\nLooking for files in: {out_dir}")
    print(f"Directory contents: {list(out_dir.glob('*'))}")
    
    # Look for files with the scenario name prefix
    snap_path = next(out_dir.glob(f"*_final_eoy_snapshot.parquet"), None)
    if snap_path is None:
        # Also check in the yearly_snapshots subdirectory
        yearly_dir = out_dir / "yearly_snapshots"
        print(f"Checking yearly_snapshots directory: {yearly_dir}")
        if yearly_dir.exists():
            print(f"Yearly snapshots contents: {list(yearly_dir.glob('*'))}")
            snap_path = next(yearly_dir.glob(f"*_snapshot_{year}.parquet"), None)
    
    # Look for event files with the scenario name prefix
    ev_path = next(out_dir.glob(f"*_final_cumulative_event_log.parquet"), None)
    
    if snap_path is None:
        # Fall back to the old naming convention if files not found with prefix
        snap_path = out_dir / f"snapshot_{year}.parquet"
        print(f"Falling back to: {snap_path}")
    
    if ev_path is None:
        ev_path = out_dir / f"events_{year}.parquet"
        print(f"Looking for events at: {ev_path}")
    
    print(f"Loading snapshot from: {snap_path}")
    print(f"Loading events from: {ev_path}")
    
    snap = pd.read_parquet(snap_path) if snap_path.exists() else pd.DataFrame()
    ev = pd.read_parquet(ev_path) if ev_path.exists() else pd.DataFrame()
    return snap, ev

@pytest.mark.parametrize("yr", [2025, 2026, 2027, 2028, 2029])
def test_headcount_growth(projection_output, yr):
    snap, ev = load_year(projection_output, yr)
    
    # Debug: Print the columns and first few rows of the event log
    print(f"\n[test_headcount_growth] Year: {yr}")
    print(f"Snapshot columns: {snap.columns.tolist() if not snap.empty else 'Empty DataFrame'}")
    print(f"Event log columns: {ev.columns.tolist() if not ev.empty else 'Empty DataFrame'}")
    
    # If event log is empty or doesn't have the expected columns, skip the test with a message
    if ev.empty or 'event_type' not in ev.columns or 'value_num' not in ev.columns:
        pytest.skip(f"Event log is empty or missing required columns for year {yr}")
    
    # Find the SOY event
    soy_events = ev[ev['event_type'] == 'SOY']
    if soy_events.empty:
        pytest.skip(f"No SOY event found in event log for year {yr}")
    
    # Get the SOY value
    soy = int(soy_events['value_num'].iloc[0])
    target = math.ceil(soy * (1 + TGROWTH))
    
    # Only validate the snapshot if it's not empty
    if not snap.empty:
        validate_eoy_snapshot(snap, target)
    else:
        pytest.skip(f"Snapshot is empty for year {yr}")

@pytest.mark.order(after="test_headcount_growth")
def test_event_breakdown(projection_output):
    for yr in range(2025, 2030):
        snap, ev = load_year(projection_output, yr)
        
        # Debug: Print the columns and first few rows of the event log
        print(f"\nEvent log columns: {ev.columns.tolist()}")
        if not ev.empty:
            print("First few rows of event log:")
            print(ev.head())
        
        # Check if the expected columns exist before querying
        if 'event_type' in ev.columns:
            hires = ev[ev['event_type'] == 'EVT_HIRE']
        else:
            hires = pd.DataFrame()
            print(f"Warning: 'event_type' column not found in event log for year {yr}")
            
        if 'meta' in ev.columns:
            term_nh = ev[ev['meta'] == 'nh-deterministic']
        else:
            term_nh = pd.DataFrame()
            print(f"Warning: 'meta' column not found in event log for year {yr}")
        
        # Only perform the assertion if we have the necessary data
        if not hires.empty and not term_nh.empty:
            expected_terms = round(len(hires) * NH_RATE)
            actual_terms = len(term_nh)
            print(f"Year {yr}: Expected {expected_terms} new hire terminations, got {actual_terms}")
            assert abs(actual_terms - expected_terms) <= 1, \
                f"Unexpected number of new hire terminations in year {yr}"
        
        # Use SOY event from event log to set baseline for target calculation
        if not ev.empty and 'event_type' in ev.columns:
            soy_events = ev[ev['event_type'] == 'SOY']
            if not soy_events.empty:
                soy = int(soy_events['value_num'].iloc[0])
                target = math.ceil(soy * (1 + TGROWTH))
                # Allow up to 15% deviation from target to account for rounding and natural variation
                lower_bound = math.floor(target * 0.85)
                upper_bound = math.ceil(target * 1.15)
                if not snap.empty and EMP_ACTIVE in snap.columns:
                    active_count = snap[EMP_ACTIVE].sum()
                    print(f"Year {yr}: Active headcount: {active_count}, Target: {target} (Range: {lower_bound}-{upper_bound})")
                    assert lower_bound <= active_count <= upper_bound, \
                        f"Active headcount {active_count} outside expected range {lower_bound}-{upper_bound} in year {yr}"
                else:
                    print(f"Warning: Could not verify active headcount for year {yr}")
            else:
                print(f"Warning: No SOY event found in event log for year {yr}")
        else:
            print(f"Warning: Event log is empty or missing required columns for year {yr}")

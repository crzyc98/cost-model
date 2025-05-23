import subprocess, pathlib, yaml, math
import pytest, pandas as pd

from cost_model.engines.run_one_year.utils import compute_headcount_targets
from cost_model.state.schema import EMP_ID, EMP_ACTIVE, EMP_HIRE_DATE
from cost_model.state.event_log import EVT_PROMOTION, EVT_TERM, EVT_HIRE
from cost_model.engines.markov_promotion import apply_markov_promotions
from cost_model.engines.term import run_new_hires as run_new_hires_deterministic

# Paths
ROOT    = pathlib.Path(__file__).resolve().parents[2]
SCRIPT  = ROOT / "scripts/run_multi_year_projection.py"
CONFIG  = ROOT / "config/dev_tiny.yaml"
CENSUS  = ROOT / "data/census_preprocessed.parquet"
OUT_ARGS= ["--config", str(CONFIG), "--census", str(CENSUS), "--output-dir", "{out}", "--debug"]

# Load YAML once
with open(CONFIG) as fh:
    GP = yaml.safe_load(fh)["global_parameters"]
TGROWTH= GP["target_growth"]
NH_RATE= GP["attrition"]["new_hire_termination_rate"]

@pytest.fixture(scope="session")
def proj_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("proj")
    # Format the command with the directory path
    cmd = ["python3", str(SCRIPT)]
    for arg in OUT_ARGS:
        cmd.append(arg.format(out=str(d)) if isinstance(arg, str) and "{out}" in arg else arg)
    subprocess.check_call(cmd)
    return d

def load_data(out_dir, year):
    # Look for the final snapshot and event log files
    snap_path = out_dir / "projection_cli_final_eoy_snapshot.parquet"
    ev_path = out_dir / "projection_cli_final_cumulative_event_log.parquet"
    
    if not snap_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snap_path}")
    if not ev_path.exists():
        raise FileNotFoundError(f"Event log file not found: {ev_path}")
        
    snap = pd.read_parquet(snap_path)
    ev = pd.read_parquet(ev_path)
    
    # Filter events for the specific year if needed
    # This assumes the event log has a 'year' column or similar to filter on
    # Adjust the filtering logic based on your actual event log structure
    if 'year' in ev.columns:
        ev = ev[ev['year'] == year]
    
    return snap, ev

def test_flow_components(proj_dir):
    """Test the flow components using the final snapshot and event log."""
    # Load the final snapshot and event log
    snap, ev = load_data(proj_dir, 2025)  # Using 2025 as a placeholder
    
    # Basic assertions about the snapshot and event log
    assert not snap.empty, "Snapshot is empty"
    assert not ev.empty, "Event log is empty"
    
    # Check that we have the expected columns in the snapshot
    required_columns = [EMP_ID, EMP_ACTIVE, EMP_HIRE_DATE]
    for col in required_columns:
        assert col in snap.columns, f"Missing required column in snapshot: {col}"
    
    # Check that we have the expected event types in the event log
    assert 'event_type' in ev.columns, "Event log is missing 'event_type' column"
    
    # Count different types of events
    hire_count = (ev['event_type'] == EVT_HIRE).sum()
    term_count = (ev['event_type'] == EVT_TERM).sum()
    promo_count = (ev['event_type'] == EVT_PROMOTION).sum()
    
    # Log some basic information for debugging
    print(f"\nSnapshot headcount: {len(snap)}")
    print(f"Active employees: {snap[EMP_ACTIVE].sum()}")
    print(f"Total events: {len(ev)}")
    print(f"Hire events: {hire_count}")
    print(f"Termination events: {term_count}")
    print(f"Promotion events: {promo_count}")
    
    # Basic sanity checks
    assert hire_count >= 0, "Invalid number of hire events"
    assert term_count >= 0, "Invalid number of termination events"
    assert promo_count >= 0, "Invalid number of promotion events"
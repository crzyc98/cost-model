
### Unit Testing – Automating the “dev-mini” Scenario

File(s): tests/integration/test_dev_mini_projection.py (new)
Deps: pytest, yaml, pandas, subprocess

#### 7.1  Fixture – run projection once

import subprocess, pathlib, pytest
PROJECT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT  = PROJECT / "scripts/run_multi_year_projection.py"
CONFIG  = PROJECT / "config/dev_tiny.yaml"
CENSUS  = PROJECT / "data/census_preprocessed.parquet"

@pytest.fixture(scope="session")
def projection_output(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("proj_out")
    subprocess.check_call(
        ["python", SCRIPT, "--config", CONFIG, "--census", CENSUS, "--out", out_dir, "--debug"]
    )
    return out_dir

7.2  Helpers – load snapshots/events

import pandas as pd, math, yaml

def load_year(out_dir, year):
    snap = pd.read_parquet(out_dir / f"snapshot_{year}.parquet")
    ev   = pd.read_parquet(out_dir / f"events_{year}.parquet")
    return snap, ev

with open(CONFIG) as fh:
    GP = yaml.safe_load(fh)["global_parameters"]
TGROWTH = GP["target_growth"]
NH_RATE = GP["attrition"]["new_hire_termination_rate"]

7.3  Headcount target assertion

from cost_model.engines.run_one_year.validation import validate_eoy_snapshot
from cost_model.utils.columns import EMP_ACTIVE

@pytest.mark.parametrize("yr", [2025, 2026, 2027, 2028, 2029])
def test_headcount_growth(projection_output, yr):
    snap, ev = load_year(projection_output, yr)
    soy = int(ev.query("event_type == 'SOY'")["value_num"].iloc[0])  # or derive from prev year
    target = math.ceil(soy * (1 + TGROWTH))
    validate_eoy_snapshot(snap, target)

7.4  Event-level mix checks

@pytest.mark.order(after="test_headcount_growth")
def test_event_breakdown(projection_output):
    start_cnt = 100  # dev_tiny baseline
    for yr in range(2025, 2030):
        snap, ev = load_year(projection_output, yr)
        hires = ev.query("event_type == 'EVT_HIRE'")
        term_nh = ev.query("meta == 'nh-deterministic'")
        assert abs(len(term_nh) - round(len(hires) * NH_RATE)) <= 1
        target = math.ceil(start_cnt * (1 + TGROWTH))
        assert snap[EMP_ACTIVE].sum() == target
        start_cnt = target

Add to pytest.ini:

[pytest]
log_cli = true
addopts = -q --strict-markers

8  CLI / Runner Wiring

  CLI / Runner Wiring
Files: projections/runner.py, projections/cli.py
	•	Swap calls to the new orchestrator:

from cost_model.engines.run_one_year.orchestrator import run_one_year


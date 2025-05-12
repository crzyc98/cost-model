"""Minimal test for the new SnapshotBuilder API and refactor wiring.
This is not a full test suite—just a smoke test for import and basic usage.
"""
import pandas as pd
import pytest

from cost_model.state.builder import SnapshotBuilder
from cost_model.state.schema import EVENT_COLS, EMP_ID


def test_snapshotbuilder_empty_build_and_update():
    builder = SnapshotBuilder(snapshot_year=2025)
    events = pd.DataFrame(columns=EVENT_COLS)
    snap = builder.build(events)
    assert EMP_ID in snap.columns
    # update with nothing should also work
    snap2 = builder.update(snap, events)
    assert snap2.shape == snap.shape

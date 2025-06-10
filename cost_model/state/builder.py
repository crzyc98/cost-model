"""High-level SnapshotBuilder orchestrating build & update flows.

## QuickStart

To build and update workforce snapshots programmatically:

```python
import pandas as pd
from cost_model.state.builder import SnapshotBuilder
from cost_model.state.event_log import EVT_HIRE, EVT_TERM, EVT_COMP

# Initialize the builder for a specific year
snapshot_year = 2025
builder = SnapshotBuilder(snapshot_year)

# Build a snapshot from scratch using an event log
events_df = pd.read_parquet('data/events_2025.parquet')
snapshot_df = builder.build(events_df)
print(f"Built snapshot with {len(snapshot_df)} employees")

# Update an existing snapshot with new events
prev_snapshot = pd.read_parquet('data/snapshot_2024.parquet')
new_events = pd.DataFrame([
    {
        'event_id': 'evt_123',
        'event_time': pd.Timestamp('2025-03-15'),
        'employee_id': 'EMP001',
        'event_type': EVT_COMP,
        'value_num': 75000.0,
        'value_json': None,
        'meta': 'Annual compensation adjustment'
    },
    {
        'event_id': 'evt_124',
        'event_time': pd.Timestamp('2025-06-30'),
        'employee_id': 'EMP002',
        'event_type': EVT_TERM,
        'value_num': None,
        'value_json': None,
        'meta': 'Voluntary termination'
    }
])

updated_snapshot = builder.update(prev_snapshot, new_events)
print(f"Updated snapshot: {updated_snapshot['active'].sum()} active employees")

# Save the snapshot
updated_snapshot.to_parquet(f'data/snapshot_{snapshot_year}.parquet')
```

This demonstrates the core workflow of building snapshots from event logs and updating them with new events.
"""

from __future__ import annotations

import pandas as pd

from . import snapshot_build as _build
from . import snapshot_update as _update

__all__ = ["SnapshotBuilder"]


class SnapshotBuilder:
    """Facade for building or updating workforce snapshots."""

    def __init__(self, snapshot_year: int):
        self.snapshot_year = snapshot_year

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, events: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
        """Return full snapshot from *events* log."""
        return _build.build_full(events, self.snapshot_year)

    def update(self, prev: pd.DataFrame, new_events: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
        """Return updated snapshot from *prev* + *new_events*."""
        return _update.update(prev, new_events, self.snapshot_year)

# Phase 1 – Nail Down Headcount (Experienced Terminations)

Goal: produce an **accurate yearly head-count projection** that correctly classifies “Experienced Terminated” (employees hired in a prior year but terminating in the projection year).

---
## 1  Re-create & observe the bug
1.  Run a *single-year* projection (e.g. 2025) with `--debug` and capture the log.
2.  Open the resulting snapshot for 31 Dec 2025 and focus on columns `employment_status`, `hire_year`, `termination_date`.
3.  Confirm the mismatch – employees that should be `Experienced Terminated` are missing or mis-tagged.

> Tip: quick CLI
> ```bash
> poetry run projection-cli --year 2025 --debug \
>   --output /tmp/one_year_debug
> ```

---
## 2  Trace the data path
```
prev_snapshot ──▶ snapshot_update.update() ──┬─▶ _apply_terminations()
                                            └─▶ _apply_new_hires()  ← InvalidIndexError
                                                   │
                                                   └─ employment_status assignment
```
Tasks
-   [ ] Inspect `snapshot_update._apply_terminations` – ensure **experienced vs new-hire** logic uses `hire_year < year`.
-   [ ] Verify `state/tenure.py.tenure_band()` properly handles terminations (tenure stops at termination date).
-   [ ] Walk through `projections/summaries/core.build_employment_status_summary()` – confirm the summary uses the updated `employment_status` codes.

---
## 3  Fix the concat index clash
Location: `state/snapshot_update._apply_new_hires`

1.  **Replicate** the `pandas.errors.InvalidIndexError` by inserting
    ```python
    _log_index_state(current, new_df)
    ```
    just before the failing `pd.concat`.
2.  Fix options (pick one):
    - `pd.concat([current.reset_index(), new_df.reset_index()]).set_index("employee_id")`
    - or, ensure unique index with
      ```python
      current = current[~current.index.duplicated(keep="last")]
      new_df  = new_df.loc[~new_df.index.isin(current.index)]
      ```
3.  Add a unit-test `tests/state/test_snapshot_update.py::test_apply_new_hires_index_unique`.

---
## 4  Validate status classification
1.  Create a *minimal* synthetic event-log:
    ```csv
    date,event_type,employee_id
    2024-01-01,EVT_HIRE,1
    2025-06-30,EVT_TERM,1
    ```
2.  Build 2024 & 2025 snapshots and assert:
    - 2024-12-31 → `employment_status == "Active"`
    - 2025-12-31 → `employment_status == "Experienced Terminated"`

---
## 5  Add regression tests
- `tests/projections/test_employment_status_summary.py` comparing expected counts vs summary output for 3-year toy dataset.

---
## 6  Definition of Done
- No `InvalidIndexError` during multi-year run.
- `Experienced Terminated` counts match manual calculations in the dashboard.
- All new tests pass, CI green.
- Documentation (this file) updated with any additional gotchas.

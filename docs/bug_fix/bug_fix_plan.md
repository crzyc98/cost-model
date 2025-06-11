# Simulation Bug-Fix Plan — 2025-06-10

## 1. Background  
The 2025 projection run fails with:
- Missing `term_rate` entries for 18 level-tenure combinations.  
- Duplicate & null `EMP_ID`s in the final snapshot (105 each).  
- Missing `start_count` for exact-target hiring.  
- Contribution calculation module disabled/skipped.

These issues block the orchestration pipeline at `run_one_year`.

this is the script to test it 
```bash
python scripts/run_multi_year_projection.py --config config/dev_tiny.yaml --census data/census_preprocessed.parquet --debug
```

## 2. Objectives for Today
1. Eliminate duplicate & null employee IDs in snapshots.  
2. Ensure hazard table has complete `term_rate` coverage for every (level, tenure).  
3. Guarantee `start_count` is supplied when exact targeting is requested.  
4. Decide whether contribution calculation should be enabled or safely bypassed.  
5. Re-run the 2025 simulation end-to-end with zero errors.

## 3. Task Breakdown
| Priority | Task | Owner | Notes |
|----------|------|-------|-------|
| P0 | Investigate ID generation & snapshot merge logic; patch root cause of duplicate/NA `EMP_ID`. | AI-DEV & You | `snapshot_update.py`, merge steps in `run_one_year`. |
| P0 | Audit hazard table loader; fill / impute missing `term_rate`s for all 18 gaps. | You | Verify against `data/hazard*.csv`. |
| P1 | Trace where `start_count` should come from; add parameter propagation & validation. | AI-DEV | `orchestrator.__init__.py` lines ~160-185. |
| P1 | Re-enable or explicitly skip contribution calc with clear logging. | You | Module path: `cost_model/engines/contrib/*.py` (if exists). |
| P2 | Add unit tests / assertions for: (a) duplicate IDs, (b) null IDs, (c) hazard coverage. | AI-DEV | Under `tests/run_one_year/`. |
| P2 | Update docs & changelog. | You | — |

## 4. Timeline (Today)
1. 10:00–11:30 — Snapshot duplicate/NA root-cause analysis & fix.  
2. 11:30–12:30 — Hazard table gaps filled; unit tests added.  
3. 13:30–14:30 — `start_count` propagation fix & validation.  
4. 14:30–15:30 — Contribution calc decision & update.  
5. 15:30–16:00 — Full 2025 run; confirm clean logs & outputs.  
6. 16:00–17:00 — Buffer / documentation.

## 5. Validation Checklist
- [ ] No WARNING/ERROR lines in projection log for 2025 run.  
- [ ] Final snapshot passes `validate_eoy_snapshot`.  
- [ ] Event log counts match snapshot transitions.  
- [ ] All unit tests green.

---

*Prepared by Windsurf AI-DEV — 2025-06-10*
## 2025-05-23: CLI and Runner Wiring Review

- Reviewed `cost_model/projections/cli.py` and `cost_model/projections/runner.py` for integration with the new orchestrator.
- Confirmed that the CLI uses `run_projection_engine` in the runner, which in turn calls `run_one_year` from `cost_model.engines.run_one_year` (the orchestrator).
- All simulation-year logic is routed through the new orchestrator; there are no legacy calls to old hiring/termination logic.
- The runner passes the required arguments (snapshot, hazard_slice, rng, year, config) to the orchestrator. If new parameters are needed, they should be added to the config and passed in.
- Recommendation: If new config options are introduced for hiring/termination, ensure they are present in the YAML config and loaded into the config namespace. Otherwise, no changes are required to CLI/runner wiring.

## 2025-05-23: Archived Legacy Orchestrator

- Archived the legacy orchestrator at `cost_model/projections/runner/orchestrator.py` to `cost_model/projections/archive/legacy_runner/orchestrator.py`
- Added deprecation warnings to the archived file
- Created `docs/archived_files.md` to track archived files and their replacements
- Confirmed no remaining imports of the legacy orchestrator in the codebase

Next: Review test coverage for the new hiring/termination flow to ensure all scenarios are properly tested.

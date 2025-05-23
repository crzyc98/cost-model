# Archived Files

This document tracks files that have been archived but kept for historical reference.

## 2025-05-23: Archived Legacy Orchestrator

### File
- Original: `cost_model/projections/runner/orchestrator.py`
- Archived to: `cost_model/projections/archive/legacy_runner/orchestrator.py`

### Reason for Archiving
- Superseded by the new modular orchestrator at `cost_model/engines/run_one_year/orchestrator.py`
- The new implementation follows the updated architecture and includes all necessary functionality

### Migration Notes
- All imports and usages have been updated to use the new orchestrator
- The archived file includes deprecation warnings to prevent accidental use
- Git history is preserved in the new location

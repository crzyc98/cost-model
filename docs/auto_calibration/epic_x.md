User Story X.1: Develop Dynamic Hazard Table Builder Function

As a Model Developer,
I want to create a new Python function (e.g., build_dynamic_hazard_table in cost_model/projections/dynamic_hazard.py) that takes the global_params object (containing detailed, tuned hazard configurations like termination_hazard, promotion_hazard, raises_hazard, cola_hazard) and other necessary inputs (e.g., simulation years, levels, tenure bands)
So that this function can generate a complete hazard table DataFrame at runtime, with all rates and multipliers (e.g., TERM_RATE, NEW_HIRE_TERM_RATE, merit_raise_pct, cola_pct, promotion_raise_pct) calculated based on the values within global_params.
Acceptance Criteria:
The function is created in the proposed location (e.g., cost_model/projections/dynamic_hazard.py).
The function signature accepts global_params and other necessary parameters (years, levels, tenure_bands).
The function correctly maps and calculates values for all required columns in the hazard table (e.g., term_rate, new_hire_term_rate, merit_raise_pct, promotion_raise_pct, cola_pct) using the corresponding detailed parameters from global_params (e.g., global_params.termination_hazard.base_rate_for_new_hire, global_params.termination_hazard.tenure_multipliers, global_params.raises_hazard.merit_base, global_params.cola_hazard.by_year, etc.).
The output DataFrame schema (columns, dtypes) is identical to the existing static hazard_table.parquet to ensure compatibility with downstream engine processing.
The function includes comprehensive logging to trace how values from global_params are being used to populate the hazard table.
Unit tests are created for this function to verify that different global_params inputs result in correspondingly different values in the generated hazard table.
✅ User Story X.2: Integrate Dynamic Hazard Table into Main Simulation Pipeline

As a Model Developer,
I want to modify the main simulation orchestration logic (in cost_model/simulation.py)
So that it calls the new build_dynamic_hazard_table function at the start of each simulation run (using the current scenario's global_params) instead of loading the hazard table from static files (data/hazard_table.parquet or CSV fallback).
Acceptance Criteria:
✅ The static file loading logic for the hazard table (lines 147-181 in cost_model/simulation.py as per audit) is replaced by a call to build_dynamic_hazard_table.
✅ The global_params (as scenario_cfg) for the current simulation scenario are correctly passed to build_dynamic_hazard_table.
✅ The dynamically generated hazard table DataFrame is then used throughout the simulation (e.g., passed to run_one_year and subsequently sliced for individual engines).
✅ The simulation runs successfully using this dynamically generated hazard table.

IMPLEMENTATION NOTES:
- Fixed column name mismatch: Dynamic hazard table now uses EMP_TENURE_BAND ("employee_tenure_band") to match what simulation engines expect
- Added robust fallback to static loading if dynamic generation fails
- Successfully tested with dev_tiny.yaml - simulation runs for 3 years and produces valid results
- Dynamic hazard table generates 90 rows (3 years × 5 levels × 6 tenure bands) with correct parameter values from global_params
User Story X.3: Validate Parameter Influence with Dynamic Hazard Table

As a Model Developer,
I want to verify that changes made by the auto-tuner to detailed hazard parameters in global_params now correctly influence the simulation outcomes via the dynamically generated hazard table
So that I can confirm the auto-tuning system is effectively connected to the core simulation logic.
Acceptance Criteria:
Add parameter tracing logs (or enhance existing ones) in the simulation engines (term.py, comp.py, markov_promotion.py, etc.) to clearly show the specific hazard rate/multiplier values being used from the hazard_slice (which is now derived from the dynamic table).
Create and run a small number of test simulations using scripts/run_simulation.py with manually modified dev_tiny.yaml files where specific detailed hazard parameters (e.g., global_params.termination_hazard.base_rate_for_new_hire, global_params.raises_hazard.merit_base) are varied.
Confirm through logs and simplified output analysis that these variations in global_params lead to corresponding changes in the values within the hazard_slice used by engines and, ultimately, to directionally expected changes in high-level simulation outputs (e.g., overall termination count, average raise amount).
Assertions can be added to integration tests to verify that hazard table values change when global_params change.


STEP 4
✅ CHECK cost_model/projections/dynamic_hazard.py AND ADD ANY COLUMNS WE NEED TO cost_model/state/schema.py

STEP 5
✅ INTEGRATE DYNAMIC HAZARD TABLE INTO MAIN SIMULATION PIPELINE (cost_model/simulation.py)
- Replaced static hazard table loading with build_dynamic_hazard_table call
- Fixed column name compatibility (EMP_TENURE_BAND vs TENURE_BAND)
- Added robust fallback to static loading
- Successfully tested with dev_tiny.yaml configuration
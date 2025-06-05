# Auto-Tuning Parameter Connection Audit

## Executive Summary

**CRITICAL FINDING**: The auto-tuning system has a fundamental disconnect between the parameters being tuned and the simulation engines. While `tune_configs.py` successfully modifies 38 detailed hazard parameters in the YAML configuration files, **these parameters are NOT being used by the simulation engines** because the hazard table is loaded statically from files rather than being dynamically constructed from `global_params`.

## The Problem

The current simulation pipeline follows this flow:
1. `tune_configs.py` modifies `dev_tiny.yaml` with tuned parameters (✅ Working)
2. `scripts/run_simulation.py` loads the YAML into `global_params` (✅ Working)  
3. `cost_model/simulation.py` **loads hazard table from static files** (❌ **BROKEN LINK**)
4. Engines receive `hazard_slice` with static values, ignoring `global_params` (❌ **BROKEN LINK**)

## Detailed Analysis by Engine

### 1. Termination Engine (term.py, nh_termination.py)

**Age Multipliers**: ✅ **CONNECTED**
- `_extract_termination_hazard_config()` correctly extracts `global_params.termination_hazard.age_multipliers`
- `_apply_age_multipliers()` applies these to termination rates
- **This works because it modifies rates AFTER the hazard table merge**

**Base Rates, Tenure Multipliers, Level Discounts**: ❌ **DISCONNECTED**
- Tuned parameters: `termination_hazard.base_rate_for_new_hire`, `tenure_multipliers.*`, `level_discount_factor`
- **Problem**: These should influence the `TERM_RATE` column in `hazard_slice`, but the hazard table is loaded from static `data/hazard_table.parquet`
- **Impact**: The core termination rates are NOT being influenced by tuning

**New Hire Termination Rate**: ❌ **DISCONNECTED**  
- Tuned parameter: `global_params.attrition.new_hire_termination_rate`
- **Problem**: `hazard_slice[NEW_HIRE_TERM_RATE]` comes from static file, not `global_params`
- **Impact**: New hire termination rates are NOT being influenced by tuning

### 2. Promotion Engine (markov_promotion.py)

**Age Multipliers**: ✅ **CONNECTED**
- `_extract_promotion_hazard_config()` correctly extracts `global_params.promotion_hazard.age_multipliers`
- `_apply_promotion_age_multipliers()` applies these to promotion probabilities
- **This works because it modifies probabilities AFTER the matrix is loaded**

**Base Rate, Tenure Multipliers, Level Dampener**: ❌ **DISCONNECTED**
- Tuned parameters: `promotion_hazard.base_rate`, `tenure_multipliers.*`, `level_dampener_factor`
- **Problem**: These should influence the `promotion_matrix` probabilities, but the matrix is loaded from static files
- **Impact**: Core promotion rates are NOT being influenced by tuning

### 3. Hiring Engine (hire.py)

**Number of Hires**: ✅ **CONNECTED**
- `global_params.target_growth` is used in headcount targeting calculations
- **This works because hiring logic reads directly from `global_params`**

**New Hire Compensation**: ❌ **PARTIALLY DISCONNECTED**
- Tuned parameters: `raises_hazard.*` (merit_base, etc.)
- **Problem**: New hire compensation uses `DefaultSalarySampler` with different parameters
- **Gap**: The `raises_hazard` parameters in SEARCH_SPACE don't directly influence new hire starting salaries

### 4. Compensation Engines (comp.py, cola.py)

**Merit Raises**: ❌ **DISCONNECTED**
- Tuned parameters: `raises_hazard.merit_base`, `merit_tenure_bump_value`, `merit_low_level_bump_value`
- **Problem**: `comp.py` uses `hazard_slice['merit_raise_pct']` from static files, not `global_params`
- **Impact**: Merit raise rates are NOT being influenced by tuning

**COLA**: ❌ **DISCONNECTED**
- Tuned parameters: `cola_hazard.by_year.*` (yearly COLA rates)
- **Problem**: `cola.py` uses `hazard_slice['cola_pct']` from static files, not `global_params`
- **Impact**: COLA rates are NOT being influenced by tuning

**Promotion Raises**: ❌ **DISCONNECTED**
- Tuned parameter: `raises_hazard.promotion_raise`
- **Problem**: `extract_promotion_raise_config_from_hazard()` uses `hazard_slice['promotion_raise_pct']` from static files
- **Impact**: Promotion raise percentages are NOT being influenced by tuning

## Root Cause Analysis

The fundamental issue is in `cost_model/simulation.py` lines 147-181:

```python
# 4. Load hazard table using proper loading function
hazard = load_and_expand_hazard_table('data/hazard_table.parquet')
# ... fallback to CSV if parquet doesn't exist
```

**This loads a static hazard table that completely ignores the tuned parameters in `global_params`.**

## Required Fixes

### Priority 1: Dynamic Hazard Table Construction

Replace the static hazard table loading with dynamic construction using `global_params`:

1. **Create a new function** `build_dynamic_hazard_table(global_params, years, levels, tenure_bands)`
2. **Integrate termination parameters**: Use `termination_hazard.base_rate_for_new_hire`, `tenure_multipliers`, `level_discount_factor` to calculate `term_rate` values
3. **Integrate compensation parameters**: Use `raises_hazard.*` to populate `merit_raise_pct`, `promotion_raise_pct` columns
4. **Integrate COLA parameters**: Use `cola_hazard.by_year.*` to populate `cola_pct` column
5. **Integrate promotion parameters**: Use `promotion_hazard.*` to populate `promotion_rate` column (if used)

### Priority 2: Update Simulation Pipeline

Modify `cost_model/simulation.py` to:
```python
# Replace static loading with dynamic construction
hazard = build_dynamic_hazard_table(
    global_params=scenario_cfg,
    years=range(start_year, start_year + projection_years),
    levels=[1, 2, 3, 4, 5],  # or extract from census
    tenure_bands=['<1', '1-3', '3-5', '5-10', '10-15', '15+']
)
```

### Priority 3: Validation and Testing

1. **Add parameter tracing**: Log which tuned parameters are actually being used
2. **Add assertions**: Verify that hazard table values change when `global_params` change
3. **Add integration tests**: Verify that tuning different parameters produces different simulation outcomes

## Impact Assessment

**Current State**: Only ~10% of tuned parameters are actually influencing the simulation (age multipliers and target_growth)

**After Fix**: 100% of tuned parameters will influence the simulation, making the auto-tuning system effective

**Risk**: This is a significant architectural change that could introduce bugs if not carefully implemented and tested

## Recommended Implementation Approach

1. **Phase 1**: Create `build_dynamic_hazard_table()` function with comprehensive logging
2. **Phase 2**: Add integration tests to verify parameter influence
3. **Phase 3**: Replace static loading in `simulation.py` with dynamic construction
4. **Phase 4**: Run validation campaigns to ensure tuning effectiveness

This fix will transform the auto-tuning system from largely ineffective to fully functional.

## Specific Code Locations to Modify

### 1. Create Dynamic Hazard Table Builder
- **New file**: `cost_model/projections/dynamic_hazard.py`
- **Function**: `build_dynamic_hazard_table(global_params, years, levels, tenure_bands)`
- **Logic**: Mirror `scripts/generate_hazard_template_yaml.py` but use `global_params` instead of static config

### 2. Update Simulation Pipeline
- **File**: `cost_model/simulation.py`
- **Lines**: 147-181 (hazard table loading)
- **Change**: Replace static loading with dynamic construction call

### 3. Add Parameter Validation
- **File**: `cost_model/engines/term.py`, `cost_model/engines/comp.py`, etc.
- **Add**: Logging to show which parameters are being used from hazard_slice vs global_params

This audit reveals that the auto-tuning system needs this critical fix to become effective.

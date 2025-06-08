# Compensation System Troubleshooting Guide

## Current Critical Issue: Negative Compensation Growth

### Problem Statement

All auto-tuning configurations show **negative compensation growth** (-10% to -25%) despite positive configuration parameters that should yield +6% to +8% growth.

**Expected vs Actual**:
- **Expected**: +6.3% (2.6% merit + 1.8% COLA + 3.2% annual increase)
- **Actual**: -13.94% (NEGATIVE!)

This indicates a critical bug in the compensation system that invalidates all auto-tuning results.

## Diagnostic Steps

### Step 1: Verify Configuration Parameters

```bash
# Check compensation parameters in best config
python -c "
from cost_model.config.loaders import load_config_to_namespace
config = load_config_to_namespace('tuned/best_config.yaml')
print('Merit base:', config.global_parameters.raises_hazard.merit_base)
print('COLA 2025:', config.global_parameters.cola_hazard.by_year._2025)
print('Annual increase:', config.global_parameters.annual_compensation_increase_rate)
"
```

### Step 2: Trace Compensation Events

```bash
# Run simulation with detailed compensation logging
python scripts/run_multi_year_projection.py \
  --config tuned/best_config.yaml \
  --census data/census_preprocessed.parquet \
  --debug 2>&1 | grep -E "(COMP|COLA|MERIT|EVT_)"
```

### Step 3: Check Event Generation

Create a test script to verify compensation events are generated correctly:

```python
# test_compensation_events.py
import pandas as pd
from cost_model.engines.comp import bump
from cost_model.projections.hazard import build_hazard_table
from cost_model.config.loaders import load_config_to_namespace

# Load config and create test data
config = load_config_to_namespace('tuned/best_config.yaml')
test_snapshot = pd.DataFrame({
    'employee_id': ['emp_001', 'emp_002'],
    'gross_compensation': [70000.0, 105000.0],
    'employee_level': [1, 2],
    'employee_tenure_band': ['1-3', '3-5'],
    'employee_termination_date': [pd.NaT, pd.NaT]
})

# Build hazard table
hazard_table = build_hazard_table([2025], test_snapshot, config.global_parameters, config.plan_rules)
hazard_slice = hazard_table[hazard_table['simulation_year'] == 2025]

# Generate compensation events
import numpy as np
rng = np.random.default_rng(42)
events = bump(test_snapshot, hazard_slice, pd.Timestamp('2025-01-01'), rng)

print("Generated events:")
for i, event_df in enumerate(events):
    print(f"Event set {i}: {len(event_df)} events")
    if not event_df.empty:
        print(event_df[['employee_id', 'event_type', 'value_num', 'event_time']].head())
```

### Step 4: Check Event Application

Verify that compensation events are properly applied to the snapshot:

```python
# test_event_application.py
from cost_model.state.snapshot_update import update

# Apply events to snapshot
updated_snapshot = update(
    prev_snapshot=test_snapshot,
    new_events=pd.concat(events, ignore_index=True),
    snapshot_year=2025
)

# Compare before/after compensation
print("Before:")
print(test_snapshot[['employee_id', 'gross_compensation']])
print("After:")
print(updated_snapshot[['employee_id', 'gross_compensation']])

# Calculate growth
for emp_id in test_snapshot['employee_id']:
    old_comp = test_snapshot[test_snapshot['employee_id'] == emp_id]['gross_compensation'].iloc[0]
    new_comp = updated_snapshot[updated_snapshot['employee_id'] == emp_id]['gross_compensation'].iloc[0]
    growth = (new_comp / old_comp) - 1
    print(f"{emp_id}: {old_comp:,.0f} → {new_comp:,.0f} ({growth:+.2%})")
```

## Common Root Causes

### 1. Event Application Bug

**Symptoms**: Events generated correctly but compensation doesn't increase

**Check**: 
```python
# Verify events are being applied
print("Events generated:", len(all_events))
print("Snapshot before update:", snapshot['gross_compensation'].sum())
print("Snapshot after update:", updated_snapshot['gross_compensation'].sum())
```

**Potential Fix**: Check `cost_model/state/snapshot_update.py` for compensation event handling

### 2. Event Overwriting

**Symptoms**: Later events overwrite earlier compensation changes

**Check**: Event timestamps and processing order
```python
# Check event timestamps
events_df['event_time'].sort_values()
# Should show: Promotions (00:00), Merit (00:01), COLA (00:02)
```

**Potential Fix**: Ensure events are processed in chronological order

### 3. New Hire Compensation Issues

**Symptoms**: New hires getting incorrect starting compensation

**Check**: New hire compensation initialization
```python
# Check new hire compensation
new_hires = snapshot[snapshot['hire_date'].dt.year == 2025]
print("New hire compensation range:", new_hires['gross_compensation'].describe())
```

**Potential Fix**: Verify new hire compensation sampling logic

### 4. Multiple Engine Conflicts

**Symptoms**: Different compensation engines applying conflicting changes

**Check**: Which compensation engines are active
```bash
# Search for compensation engine calls
grep -r "update_salary\|apply_comp_bump\|bump\|cola" cost_model/engines/run_one_year/
```

**Potential Fix**: Ensure only one compensation engine is used

### 5. Calculation Errors

**Symptoms**: Wrong formulas in merit/COLA calculations

**Check**: Manual calculation verification
```python
# Verify merit calculation
merit_rate = 0.026  # 2.6%
old_comp = 70000
expected_new_comp = old_comp * (1 + merit_rate)
print(f"Expected: {old_comp} → {expected_new_comp} ({merit_rate:.1%})")
```

## Quick Fixes to Try

### Fix 1: Force Positive Compensation Events

Add validation to ensure compensation events are positive:

```python
# In compensation event generation
def validate_compensation_events(events_df):
    """Ensure all compensation events result in increases."""
    if 'value_num' in events_df.columns:
        negative_events = events_df[events_df['value_num'] < 0]
        if not negative_events.empty:
            raise ValueError(f"Found {len(negative_events)} negative compensation events")
    return events_df
```

### Fix 2: Add Compensation Growth Validation

Add end-to-end validation:

```python
# In simulation pipeline
def validate_compensation_growth(start_snapshot, end_snapshot):
    """Validate positive compensation growth."""
    start_total = start_snapshot['gross_compensation'].sum()
    end_total = end_snapshot['gross_compensation'].sum()
    growth = (end_total / start_total) - 1
    
    if growth < 0:
        raise ValueError(f"Negative compensation growth: {growth:.2%}")
    
    return growth
```

### Fix 3: Simplify Compensation System

Temporarily use only the merit base rate:

```yaml
# Simplified compensation config for testing
global_parameters:
  raises_hazard:
    merit_base: 0.04  # 4% simple merit increase
    merit_tenure_bump_value: 0.0  # Disable complexity
    merit_low_level_bump_value: 0.0
  cola_hazard:
    by_year:
      2025: 0.0  # Disable COLA temporarily
  annual_compensation_increase_rate: 0.0  # Disable redundant increases
```

## Investigation Priority

1. **High Priority**: Event application in `snapshot_update.py`
2. **High Priority**: Event generation in `comp.py` and `cola.py`
3. **Medium Priority**: Event ordering and timestamps
4. **Medium Priority**: New hire compensation initialization
5. **Low Priority**: Multiple engine conflicts

## Success Criteria

The compensation system is fixed when:

1. **Positive Growth**: All configurations show positive compensation growth
2. **Expected Range**: Growth rates match configured parameters (±1%)
3. **Consistency**: Similar configurations produce similar growth rates
4. **Validation**: Auto-tuning can achieve target compensation growth (3-4%)

## Next Steps

1. **Run Diagnostic Scripts**: Execute the test scripts above
2. **Identify Root Cause**: Focus on highest-probability issues first
3. **Implement Fix**: Make targeted fixes to identified problems
4. **Validate Fix**: Re-run auto-tuning with fixed system
5. **Update Documentation**: Document the fix and prevention measures

Once the compensation system is fixed, the auto-tuning results will be valid and can be used for production configuration.

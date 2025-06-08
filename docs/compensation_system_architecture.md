# Compensation System Architecture

## Overview

The compensation system manages all aspects of employee compensation growth through multiple coordinated engines. The system is designed to achieve target total compensation growth through configurable parameters that control different types of compensation adjustments.

## System Architecture

### Core Components

1. **Merit Raise Engine** (`cost_model/engines/comp.py`)
2. **COLA Engine** (`cost_model/engines/cola.py`) 
3. **Promotion Engine** (`cost_model/engines/promotion.py`)
4. **Legacy Compensation Engine** (`cost_model/engines/compensation.py`)
5. **Event Application System** (`cost_model/state/snapshot_update.py`)

### Event Types

The system uses distinct event types with specific timestamps to ensure proper ordering:

- **EVT_COMP**: Merit raises (timestamp: 00:01)
- **EVT_COLA**: Cost of living adjustments (timestamp: 00:02) 
- **EVT_PROMOTION**: Role/level changes (timestamp: 00:00)
- **EVT_RAISE**: Legacy raise events

## Compensation Growth Components

### 1. Merit Raises (EVT_COMP)

**Purpose**: Annual performance-based salary increases for active employees.

**Configuration Parameters**:
```yaml
global_parameters:
  raises_hazard:
    merit_base: 0.03                    # Base merit rate (3%)
    merit_tenure_bump_bands: ["<1", "1-3", "3-5"]  # Tenure bands for bonuses
    merit_tenure_bump_value: 0.005      # Additional merit for early tenure (0.5%)
    merit_low_level_cutoff: 2           # Level threshold for low-level bonus
    merit_low_level_bump_value: 0.005   # Additional merit for low levels (0.5%)
```

**Calculation Logic**:
```
Final Merit Rate = merit_base + tenure_bonus + level_bonus
- tenure_bonus = merit_tenure_bump_value (if employee tenure in merit_tenure_bump_bands)
- level_bonus = merit_low_level_bump_value (if employee level <= merit_low_level_cutoff)
```

**Expected Impact**: 3.0% - 4.0% annual increase for most employees

### 2. Cost of Living Adjustments (EVT_COLA)

**Purpose**: Inflation adjustments applied to all employees (active and terminated).

**Configuration Parameters**:
```yaml
global_parameters:
  cola_hazard:
    by_year:
      2025: 0.020  # 2.0% COLA for 2025
      2026: 0.018  # 1.8% COLA for 2026
      2027: 0.016  # 1.6% COLA for 2027
      2028: 0.015  # 1.5% COLA for 2028
      2029: 0.014  # 1.4% COLA for 2029
```

**Calculation Logic**:
```
New Compensation = Current Compensation × (1 + cola_rate)
```

**Expected Impact**: 1.4% - 2.0% annual increase for all employees

### 3. Promotion Raises (EVT_PROMOTION)

**Purpose**: Compensation increases for employees receiving promotions.

**Configuration Parameters**:
```yaml
global_parameters:
  raises_hazard:
    promotion_raise: 0.10  # 10% raise for promotions

# Alternative configuration in compensation section:
global_parameters:
  compensation:
    promo_raise_pct:
      "0_to_1": 0.10  # 10% raise for level 0→1 promotion
      "1_to_2": 0.10  # 10% raise for level 1→2 promotion
      "2_to_3": 0.10  # 10% raise for level 2→3 promotion
      "3_to_4": 0.10  # 10% raise for level 3→4 promotion
```

**Expected Impact**: 10% increase for promoted employees (~8-12% of workforce annually)

### 4. Annual Compensation Increase Rate

**Purpose**: Base annual increase applied through various mechanisms.

**Configuration Parameters**:
```yaml
global_parameters:
  annual_compensation_increase_rate: 0.03  # 3% base annual increase
```

**Note**: This parameter's exact implementation needs clarification - it may be redundant with merit_base.

### 5. Job Level Merit Distributions

**Purpose**: Level-specific merit increase distributions with stochastic variation.

**Configuration Parameters**:
```yaml
global_parameters:
  compensation:
    merit_dist:
      1:  # Staff level
        mu: 0.035    # 3.5% average merit
        sigma: 0.01  # 1% standard deviation
      2:  # Manager level  
        mu: 0.04     # 4.0% average merit
        sigma: 0.012 # 1.2% standard deviation
      3:  # Senior Manager level
        mu: 0.045    # 4.5% average merit
        sigma: 0.015 # 1.5% standard deviation
      4:  # Director level
        mu: 0.05     # 5.0% average merit
        sigma: 0.02  # 2% standard deviation
```

**Expected Impact**: 3.5% - 5.0% average increase by level with normal distribution variation

## Target Total Compensation Growth

### Calculation Formula

For a typical employee receiving all applicable increases:

```
Total Annual Compensation Growth = 
  Merit Rate + COLA Rate + (Promotion Rate × Promotion Probability)

Example with typical parameters:
= 3.5% (merit) + 1.8% (COLA) + (10% × 10% promotion rate)
= 3.5% + 1.8% + 1.0%
= 6.3% total annual growth
```

### Parameter Tuning for Target Growth

To achieve a specific total compensation growth target (e.g., 4.0%):

**Primary Levers**:
1. **merit_base**: Direct impact on all active employees
2. **cola_hazard.by_year**: Direct impact on all employees  
3. **annual_compensation_increase_rate**: Base increase mechanism

**Secondary Levers**:
4. **merit_tenure_bump_value**: Affects early-tenure employees
5. **merit_low_level_bump_value**: Affects lower-level employees
6. **promotion_raise**: Affects promoted employees

**Tuning Strategy**:
```yaml
# For 4.0% target total compensation growth:
merit_base: 0.025           # 2.5% merit (reduced from 3.0%)
cola_hazard.by_year.2025: 0.015  # 1.5% COLA (reduced from 2.0%)
# Total: 2.5% + 1.5% = 4.0% base growth
```

## Event Processing Order

The system processes compensation events in a specific order to ensure correct calculations:

1. **Promotions** (00:00): Level/role changes
2. **Merit Raises** (00:01): Applied to current compensation
3. **COLA Adjustments** (00:02): Applied to post-merit compensation

This ordering ensures COLA applies to the increased post-merit compensation values.

## Auto-Tuning Integration

### Search Space Parameters

The auto-tuning system should include these compensation parameters:

```python
COMPENSATION_SEARCH_SPACE = {
    # Primary levers for total compensation control
    "global_parameters.raises_hazard.merit_base": [0.020, 0.025, 0.030, 0.035, 0.040],
    "global_parameters.annual_compensation_increase_rate": [0.025, 0.030, 0.035],
    
    # COLA rates by year (can be varied independently)
    "global_parameters.cola_hazard.by_year.2025": [0.010, 0.015, 0.020, 0.025],
    "global_parameters.cola_hazard.by_year.2026": [0.010, 0.015, 0.020, 0.025],
    "global_parameters.cola_hazard.by_year.2027": [0.010, 0.015, 0.020, 0.025],
    
    # Secondary levers for fine-tuning
    "global_parameters.raises_hazard.merit_tenure_bump_value": [0.000, 0.005, 0.010],
    "global_parameters.raises_hazard.merit_low_level_bump_value": [0.000, 0.005, 0.010],
    "global_parameters.raises_hazard.promotion_raise": [0.08, 0.10, 0.12],
}
```

### Target Metrics

The auto-tuning system should optimize for:

1. **Total Compensation Growth**: Target 3.0% - 4.0% annually
2. **Compensation Growth Stability**: Low variance across years
3. **Level-Appropriate Growth**: Higher growth for senior levels
4. **Cost Control**: Avoid excessive compensation inflation

## Known Issues and Fixes Needed

### Critical Bug: Negative Compensation Growth

**Problem**: Current system shows -10% to -25% compensation growth despite positive configuration parameters.

**Potential Causes**:
1. **Event Application Bug**: Compensation events not being applied correctly
2. **Calculation Error**: Wrong formulas in merit/COLA calculations
3. **Event Ordering**: Events being applied in wrong order or overwriting each other
4. **New Hire Compensation**: New hires getting incorrect starting salaries
5. **Multiple Engine Conflict**: Different compensation engines conflicting

**Investigation Needed**:
1. Trace compensation events from generation to application
2. Verify event timestamps and ordering
3. Check snapshot update mechanism for compensation events
4. Validate new hire compensation initialization
5. Ensure only one compensation engine is active

### Recommended Fixes

1. **Consolidate Engines**: Use only the modern `comp.py` engine, deprecate legacy engines
2. **Fix Event Application**: Ensure compensation events properly update `EMP_GROSS_COMP`
3. **Add Validation**: Include compensation growth validation in simulation pipeline
4. **Improve Logging**: Add detailed compensation event logging for debugging

## Testing and Validation

### Unit Tests Needed

1. **Merit Calculation Tests**: Verify merit rate calculations with various parameters
2. **COLA Application Tests**: Verify COLA applies correctly to all employees
3. **Event Ordering Tests**: Verify events are processed in correct order
4. **Total Growth Tests**: Verify total compensation growth matches expected values

### Integration Tests Needed

1. **End-to-End Compensation**: Run full simulation and verify positive compensation growth
2. **Parameter Sensitivity**: Test how parameter changes affect total compensation
3. **Auto-Tuning Validation**: Verify auto-tuning can achieve target compensation growth

### Validation Metrics

```python
def validate_compensation_growth(initial_snapshot, final_snapshot):
    """Validate compensation growth meets expectations."""
    initial_total_comp = initial_snapshot['gross_compensation'].sum()
    final_total_comp = final_snapshot['gross_compensation'].sum()
    
    growth_rate = (final_total_comp / initial_total_comp) - 1
    
    assert growth_rate > 0, f"Negative compensation growth: {growth_rate:.2%}"
    assert 0.02 <= growth_rate <= 0.08, f"Compensation growth out of range: {growth_rate:.2%}"
    
    return growth_rate
```

This architecture provides a comprehensive framework for understanding and controlling total compensation growth through the simulation system's configurable parameters.

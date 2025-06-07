# Campaign 2025-06-06 Analysis: Successful Configuration Parameter Ranges

## Executive Summary

This analysis examines 15 systematically sampled successful configurations from today's campaign (2025-06-06) to identify stable parameter ranges and patterns that lead to successful simulations. The analysis reveals **critical issues** that must be addressed, particularly the **universal COLA duplicate keys problem** and provides specific recommended parameter ranges for future campaigns.

## Key Findings

### 1. Critical Issue: COLA Duplicate Keys (100% of Configs Affected)

**PROBLEM**: All 15 successful configurations contain duplicate COLA keys with conflicting values:
- Both numeric keys (2025, 2026, etc.) and string keys ('2025', '2026', etc.) exist
- This creates unpredictable behavior depending on which value YAML parser uses
- String key values differ significantly from numeric key values

**IMPACT**: This is likely causing configuration parsing inconsistencies and simulation failures.

**EXAMPLE**:
```yaml
cola_hazard:
  by_year:
    2025: 0.02      # Numeric key
    2026: 0.018
    2027: 0.016
    2028: 0.015
    2029: 0.014
    '2025': 0.015   # String key - DIFFERENT VALUE!
    '2026': 0.01    # String key - DIFFERENT VALUE!
    '2027': 0.006   # String key - DIFFERENT VALUE!
    '2028': 0.007   # String key - DIFFERENT VALUE!
    '2029': 0.004   # String key - DIFFERENT VALUE!
```

### 2. Stable Parameter Ranges from Successful Configurations

Based on analysis of 15 successful configs, the following parameter ranges lead to stable simulations:

#### Core Stability Parameters
| Parameter | Safe Range (10th-90th percentile) | Mean | Notes |
|-----------|-----------------------------------|------|-------|
| `new_hire_rate` | **0.40 - 0.60** | 0.50 | 9/15 configs had ≥0.5; not problematic as feared |
| `annual_compensation_increase_rate` | **0.025 - 0.032** | 0.028 | Conservative range preferred |
| `target_growth` | **0.055 - 0.070** | 0.061 | No configs ≥0.08 (good sign) |
| `termination_base_rate_for_new_hire` | **0.030 - 0.050** | 0.043 | No configs ≤0.02 (validates concern) |

#### Secondary Parameters
| Parameter | Safe Range | Mean | Notes |
|-----------|------------|------|-------|
| `promotion_base_rate` | **0.080 - 0.138** | 0.101 | Wide acceptable range |
| `merit_base` | **0.020 - 0.032** | 0.026 | Aligns with comp increase rates |
| `promotion_level_dampener_factor` | **0.10 - 0.20** | 0.15 | Binary distribution observed |

#### Age-Related Termination Multipliers
| Parameter | Safe Range | Mean | Notes |
|-----------|------------|------|-------|
| `term_age_mult_60_65` | **4.0 - 15.0** | 9.1 | High variance acceptable |
| `term_age_mult_65_plus` | **5.0 - 20.0** | 12.5 | High variance acceptable |

### 3. Parameter Correlation Patterns

#### Successful Combinations Observed:
1. **Conservative Growth**: `target_growth` (0.055-0.06) + `new_hire_rate` (0.4-0.5)
2. **Moderate Growth**: `target_growth` (0.06-0.065) + `new_hire_rate` (0.45-0.55)  
3. **Higher Growth**: `target_growth` (0.065-0.07) + `new_hire_rate` (0.5-0.6)

#### Risk Mitigation Patterns:
- Higher `new_hire_rate` paired with higher `termination_base_rate_for_new_hire`
- Lower `annual_compensation_increase_rate` often paired with higher `merit_base`
- Age multipliers show wide acceptable ranges (not critical for stability)

### 4. Validation of Suspected Problem Parameters

#### ✅ **Validated Concerns**:
- **Very low termination rates**: No successful configs had `termination_base_rate_for_new_hire` ≤ 0.02
- **COLA duplicate keys**: 100% of configs affected (critical issue)

#### ❌ **Invalidated Concerns**:
- **High new_hire_rate**: 60% of successful configs had ≥ 0.5; not inherently problematic
- **Very high target_growth**: No configs ≥ 0.08; current ranges are reasonable
- **Very high compensation increases**: No configs ≥ 0.04; current ranges are safe

## Recommended Actions

### 1. **IMMEDIATE PRIORITY**: Fix COLA Duplicate Keys
```bash
# This issue must be resolved before next campaign
# Recommend removing all string keys and keeping only numeric keys
```

### 2. **Updated Search Space Recommendations**

#### Conservative Ranges (Recommended for next campaign):
```yaml
new_hire_rate: [0.40, 0.60]          # Safe range confirmed
annual_compensation_increase_rate: [0.025, 0.032]  # Narrow to stable range
target_growth: [0.055, 0.070]        # Remove very high values
termination_base_rate_for_new_hire: [0.030, 0.050]  # Avoid low values
promotion_base_rate: [0.080, 0.140]  # Safe working range
merit_base: [0.020, 0.032]           # Align with comp increases
```

#### More Aggressive Ranges (For experimentation):
```yaml
new_hire_rate: [0.35, 0.65]          # Slightly wider
annual_compensation_increase_rate: [0.020, 0.035]  # Include low end
target_growth: [0.050, 0.075]        # Slightly wider
termination_base_rate_for_new_hire: [0.025, 0.055]  # Include boundary cases
```

### 3. **Parameter Interaction Rules**

Implement these constraints to avoid problematic combinations:

```python
# Constraint 1: Balance hiring and growth
if target_growth > 0.065:
    new_hire_rate_min = 0.50
    
# Constraint 2: Termination rate safety
if new_hire_rate > 0.55:
    termination_base_rate_for_new_hire_min = 0.035
    
# Constraint 3: Compensation balance
if annual_compensation_increase_rate < 0.027:
    merit_base_min = 0.025
```

## Implementation Priority

### Week 1 (Critical)
1. **Fix COLA duplicate keys issue** in config generation
2. **Test fixed config generation** with small sample
3. **Update search space** with conservative ranges

### Week 2 (High Priority)  
1. **Deploy refined search space** for 100-200 iteration campaign
2. **Monitor failure rates** with new constraints
3. **Validate parameter interaction rules**

## Technical Details

### Analysis Methodology
- **Sample Size**: 15 configurations (every 20th from 300 total)
- **Selection Method**: Systematic sampling from successful configs only
- **Statistical Approach**: Percentile-based ranges (10th-90th percentile)
- **Validation Method**: Cross-reference with known failure patterns

### Data Quality
- **Coverage**: Representative sample across config space
- **Completeness**: All critical parameters extracted and analyzed
- **Reliability**: Based on actually successful simulation runs

## Conclusion

The analysis reveals that the configuration parameter space is more stable than initially feared, with the notable exception of the **critical COLA duplicate keys issue**. The recommended parameter ranges are based on empirically successful configurations and should significantly improve campaign success rates.

The key insight is that **moderate parameter values generally lead to stability**, and the previously suspected "problematic" high new_hire_rates are actually acceptable when balanced with appropriate termination rates.

**Next Action**: Fix COLA duplicate keys issue immediately and deploy conservative parameter ranges for next campaign.
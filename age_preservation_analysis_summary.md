# Age Preservation Analysis Summary

## Executive Summary

With the clarification that **age distribution preservation** (not transformation) is the goal, this analysis reveals significant insights about the previous 200-iteration tuning campaign and the path forward.

## Key Findings

### 1. Starting Census Age Distribution (New Baseline Target)

The starting census (`data/census_template.parquet`) has this age distribution that should be preserved:

```
Age Band    Proportion    Count
<30         0.101         12 employees  
30-39       0.202         24 employees
40-49       0.353         42 employees  (largest group)
50-59       0.210         25 employees
60-65       0.050         6 employees
65+         0.084         10 employees
```

**Key insight**: The workforce is mature with 64.7% over age 40, mean age 46.0 years.

### 2. Best Configuration Performance (config_193_20250604_110340.yaml)

**Workforce Flow:**
- Maintained stable headcount (119 → 116, 0% growth vs 3% target)
- Estimated 17.6 annual hires and terminations (15% turnover)
- Achieved 5.3% pay growth (vs 3% target)

**Age Preservation Results:**
- **Age preservation score: 0.777** (1.0 = perfect preservation)
- **Total age error: 0.446** (sum of absolute deviations)

**Major age distribution changes:**
1. **30-39 band: +22.1 percentage points** (20.2% → 42.2%) - MASSIVE INCREASE
2. **40-49 band: -12.9 percentage points** (35.3% → 22.4%) - LARGE DECREASE  
3. **<30 band: -4.9 percentage points** (10.1% → 5.2%) - MODERATE DECREASE

### 3. Root Cause Analysis

**Primary Challenge: New Hire Age Gap**
- New hires average age: **30 years**
- Census average age: **46 years**  
- **16-year age gap** creates systematic downward pressure on age distribution
- With 17.6 annual hires, this significantly skews the workforce younger

**Age Multiplier Effects:**
- Termination hazard favors retention of younger workers (<30: 0.6x, 30-39: 1.0x)
- Promotion hazard accelerates younger worker advancement (<30: 1.6x, 30-39: 1.3x)
- Combined effect: younger workers stay longer and get promoted faster

### 4. Feasibility Assessment

**Age preservation is inherently difficult** given:
1. **Structural age imbalance**: 16-year gap between new hires and workforce average
2. **Growth pressure**: Any positive headcount growth amplifies the age skew
3. **Competing objectives**: Age preservation conflicts with headcount/pay growth targets

**The "best" configuration achieved moderate preservation** but at the cost of:
- Missing headcount growth target (0% vs 3%)
- Overshooting pay growth target (5.3% vs 3%)

## Strategic Recommendations

### 1. Immediate: Update Tuning System for Age Preservation

**Technical Change Required in `tune_configs.py`:**

```python
def load_baseline_distributions() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load baseline distributions with age preservation focus."""
    
    # Load starting census age distribution as baseline target
    census_path = Path("data/census_template.parquet")
    df = pd.read_parquet(census_path)
    
    # Calculate age distribution from starting census
    simulation_start = pd.Timestamp('2025-01-01')
    df['calculated_age'] = ((simulation_start - pd.to_datetime(df['employee_birth_date'])).dt.days / 365.25).round(1)
    df['calculated_age_band'] = df['calculated_age'].map(assign_age_band)
    
    age_dist = df['calculated_age_band'].value_counts(normalize=True).sort_index()
    baseline_age_dist = {str(band): float(count) for band, count in age_dist.items()}
    
    # Load tenure target from config (unchanged)
    baseline_config_path = Path("config/tuning_baseline.yaml")
    if baseline_config_path.exists():
        with open(baseline_config_path, 'r') as f:
            baseline_config = yaml.safe_load(f)
        tenure_dist = baseline_config.get("target_tenure_distribution", {})
    else:
        # Default tenure distribution
        tenure_dist = {
            "<1": 0.20, "1-3": 0.30, "3-5": 0.25, 
            "5-10": 0.15, "10-15": 0.07, "15+": 0.03
        }
    
    return baseline_age_dist, tenure_dist
```

### 2. Consider Search Space Expansion

**Current SEARCH_SPACE limitations:**
- New hire age parameters (`new_hire_average_age`, `new_hire_age_std_dev`) are **not tunable**
- This severely limits the system's ability to achieve age preservation

**Recommendation:** Add new hire age parameters to SEARCH_SPACE:
```python
"global_parameters.new_hire_average_age": (25, 50),
"global_parameters.new_hire_age_std_dev": (3, 15),
```

### 3. Score Weight Rebalancing

**Current challenge:** Age preservation conflicts with growth targets

**Options:**
1. **Increase WEIGHT_AGE** if age preservation is primary goal
2. **Reduce growth targets** to realistic levels given age constraints  
3. **Accept trade-offs** and optimize for best overall balance

### 4. Parameter Strategy Insights

**For better age preservation, consider:**
- **Higher new hire average age** (35-40 vs current 30)
- **More neutral age multipliers** in termination/promotion hazards
- **Lower growth targets** to reduce new hire volume pressure

## Next Steps

1. **Implement the technical change** in `load_baseline_distributions()`
2. **Run validation simulation** with modified baseline to confirm age preservation scoring
3. **Consider search space expansion** for new hire age parameters
4. **Reassess score weights** based on business priorities
5. **Run new tuning campaign** with corrected age baseline (50-100 iterations recommended)

## Conclusion

The previous campaign's "best" configuration actually performed reasonably well at age preservation (77.7% score) given the structural challenges. However, with the corrected understanding that preservation is the goal, we can now:

1. Fix the scoring system to use the correct baseline
2. Potentially expand the search space to include new hire age parameters  
3. Run more targeted campaigns focused on age preservation

The 16-year age gap between new hires and the workforce average remains the fundamental challenge that any tuning campaign must address.

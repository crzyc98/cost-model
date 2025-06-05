# Epic 3 User Story 3.3 - Action Plan

## Executive Summary

The 200-iteration auto-tuning campaign analysis revealed that the system is operationally successful but constrained by search space limitations. The "best" configuration (score 0.1553) achieved significant target misses:

- **Young workforce**: 5.2% vs 25% target (-19.8pp)
- **New hire retention**: 45% vs 20% target (+25pp) 
- **Headcount growth**: 0% vs 3% target (-3pp)

## Root Cause: Search Space Constraints

**Critical Missing Parameters** (were fixed, not tuned):
- `new_hire_rate`: Fixed at 6% (should be 8-18%)
- `new_hire_average_age`: Fixed at 30 (should be 25-30)
- `max_working_age`: Fixed at 65 (should be 62-65)

**Insufficient Ranges**:
- `new_hire_termination_rate`: Min 20% too high (should be 10-25%)
- `age_multipliers.<30`: Min 0.6 insufficient (should be 0.3-0.8)

## Implemented Fixes

✅ **Updated `tuning/tune_configs.py`**:
- Added missing parameters to SEARCH_SPACE
- Expanded ranges for better demographic targeting
- Rebalanced score weights (Age: 40%, Tenure: 30%, HC: 20%, Pay: 10%)
- Added comprehensive documentation

## Next Steps

### 1. Validate Parameter Pathways (Immediate)

Before running the next campaign, verify that tuned parameters actually influence outcomes:

```bash
# Test new_hire_rate impact
python scripts/run_simulation.py --config config/test_hire_rate_low.yaml --output test_low_hire
python scripts/run_simulation.py --config config/test_hire_rate_high.yaml --output test_high_hire

# Compare young workforce percentages in outputs
```

### 2. Run Enhanced Tuning Campaign

```bash
# Execute improved tuning with expanded search space
cd tuning
python tune_configs.py --iterations 300

# Expected improvements:
# - Young workforce: 15-20% (vs current 5.2%)
# - New hire retention: 25-30% (vs current 45%)
# - Headcount growth: 2-3% (vs current 0%)
```

### 3. Monitor Key Metrics

Track these metrics during the campaign:
- **Age <30 percentage**: Target 15-20% minimum
- **Tenure <1 percentage**: Target 25-30% maximum  
- **Headcount growth**: Target 2-3%
- **Score distribution**: Look for scores <0.10

### 4. Post-Campaign Analysis

After the enhanced campaign:
1. Compare best configuration against Epic 3 baseline
2. Validate demographic improvements
3. Test with production census data
4. Document parameter insights for future campaigns

## Expected Timeline

- **Parameter validation**: 1-2 days
- **Enhanced campaign (300 iterations)**: 2-3 days
- **Analysis and documentation**: 1 day
- **Total**: ~1 week

## Success Criteria

The enhanced campaign will be considered successful if:
- Young workforce (<30) reaches 15%+ (vs current 5.2%)
- New hire churn (<1 tenure) drops to 30% or below (vs current 45%)
- Headcount growth achieves 2%+ (vs current 0%)
- Best score improves to <0.10 (vs current 0.1553)

## Risk Mitigation

**If enhanced campaign fails to improve**:
1. Investigate model constraints (hiring logic, termination logic)
2. Consider forced retirement enforcement at max_working_age
3. Explore hiring age distribution controls beyond mean/std
4. Evaluate tenure-based hiring rate parameters

## Files Modified

- ✅ `tuning/tune_configs.py`: Updated search space and score weights
- ✅ `docs/auto_calibration/epic_3.md`: Comprehensive analysis
- ✅ `docs/auto_calibration/epic_3_action_plan.md`: This action plan

## Contact

For questions about this analysis or implementation:
- See detailed analysis in `docs/auto_calibration/epic_3.md`
- Review search space changes in `tuning/tune_configs.py`
- Check baseline targets in `config/tuning_baseline.yaml`

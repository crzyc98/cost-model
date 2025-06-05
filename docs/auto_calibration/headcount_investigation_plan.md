# Headcount Management Logic Investigation Plan

## Executive Summary

Campaign 6 achieved the best-ever calibration score (0.0723) but revealed a critical limitation: persistent -1.69% headcount growth despite aggressive parameter tuning targeting 7% growth. This investigation plan addresses the fundamental model logic issues preventing positive headcount growth.

## 🔍 Investigation Findings

### Code Analysis Results

1. **Exact Targeting Logic is Sound**: The `manage_headcount_to_exact_target()` function is mathematically correct and well-tested
2. **`maintain_headcount` Parameter Exists**: Found in configuration but set to `false` in all campaigns
3. **Execution Sequence Identified**: Clear order of operations in `run_one_year()` orchestrator
4. **Parameter Flow Mapped**: Complete trace from `target_growth` to hiring decisions

### Campaign 6 Best Configuration Analysis

**Key Parameters**:
- `target_growth`: 0.07 (7% growth target)
- `new_hire_rate`: 0.50 (50% of start count)
- `new_hire_termination_rate`: 0.25 (25% of new hires terminate)
- `base_rate_for_new_hire`: 0.04 (4% base termination rate)

**Termination Multipliers**:
- Age bands: <30 (0.4x) to 65+ (8.0x)
- Tenure bands: <1 (0.1x) to 15+ (0.3x)

## 🚨 Root Cause Hypotheses

### Primary Hypothesis: New Hire Termination Engine Failure
Based on User Story Y.2 findings showing **0% actual new hire terminations** vs 25% configured:
- Parameter location mismatch
- Logic bypass or processing failure
- Event generation/application issues

### Secondary Hypothesis: Experienced Attrition Overwhelm
Even with tuned parameters, experienced employee termination rates (especially 60+ age bands with 4x-8x multipliers) may systematically overwhelm hiring capacity.

### Tertiary Hypothesis: Order-of-Operations Timing
Exact targeting calculation happens after experienced terminations but before new hire terminations, potentially causing final headcount mismatches.

## 🔧 Investigation Tasks

### Task 1: Quantify Experienced Employee Attrition Impact
**Script**: `scripts/investigate_headcount_logic.py`
**Objective**: Measure actual magnitude of experienced employee terminations
**Actions**:
- Run diagnostic simulation with Campaign 6 best config
- Extract termination data by age/tenure bands
- Compare experienced vs new hire termination volumes
- Calculate effective vs configured termination rates

### Task 2: Test `maintain_headcount: true` Mode
**Objective**: Determine if this parameter can force positive growth
**Actions**:
- Modify Campaign 6 config to enable `maintain_headcount`
- Run simulation with identical parameters
- Compare headcount outcomes and distribution impacts

### Task 3: Validate New Hire Termination Assumptions
**Objective**: Verify new hire termination engine functionality
**Actions**:
- Add detailed logging to new hire termination processing
- Trace actual vs configured termination rates
- Verify parameter location and access

### Task 4: Order-of-Operations Validation
**Objective**: Verify exact targeting achieves intended results
**Actions**:
- Add comprehensive headcount logging at each orchestrator step
- Track progression: SOY → Terms → Hiring → NH Terms → EOY
- Identify systematic discrepancies

## 🎯 Potential Solutions

### Solution 1: Parameter Adjustment
- Reduce `new_hire_termination_rate` from 0.25 to 0.15-0.20
- Fine-tune experienced termination multipliers
- Add termination rate caps by demographic

### Solution 2: Enhanced Exact Targeting
- Implement iterative targeting with mid-year adjustments
- Add tolerance bands (±1-2 employees)
- Improve rounding precision

### Solution 3: Dynamic Headcount Management
- Quarterly hiring cycles instead of annual
- "True-up" hiring based on actual attrition
- Adaptive termination rate estimation

### Solution 4: Forced Headcount Maintenance
- Utilize `maintain_headcount: true` if effective
- Hybrid approach: exact targeting with headcount floor
- Headcount monitoring and correction mechanisms

## 📊 Success Criteria

1. **Growth Accuracy**: Actual growth within ±0.5% of `target_growth`
2. **Predictable Response**: Linear relationship between target and actual growth
3. **Maintained Calibration**: No degradation in age/tenure/pay metrics
4. **Robustness**: Consistent performance across parameter ranges
5. **Transparency**: Clear logging of headcount decisions

## 🚀 Execution Plan

### Phase 1: Diagnostic Analysis (1-2 days)
```bash
# Run investigation script
python scripts/investigate_headcount_logic.py

# Analyze results
python scripts/analyze_investigation_results.py
```

### Phase 2: Targeted Fix Implementation (2-3 days)
Based on Phase 1 findings:
- Fix new hire termination engine if broken
- Adjust parameter assumptions if mismatched
- Enhance exact targeting logic if needed
- Implement dynamic headcount management if required

### Phase 3: Validation Campaign (1-2 days)
```bash
# Run focused auto-tuning with fix
python tuning/tune_configs.py --iterations 50 --output-dir validation_post_fix

# Verify positive growth achievement
python tuning/analyze_tuning_results.py validation_post_fix/
```

### Phase 4: Production Readiness (1 day)
- Document fix and implications
- Update configuration templates
- Prepare deployment recommendations

## 🔍 Critical Questions

1. **Is the new hire termination engine functional?** (Y.2 showed 0% vs 25% configured)
2. **Are experienced termination rates accurately reflected in hiring calculations?**
3. **Does `maintain_headcount: true` provide a viable workaround?**
4. **What is the actual vs expected termination rate variance?**
5. **Are there systematic rounding or timing issues?**

## 📁 Investigation Outputs

### Generated Files
- `investigation_results/baseline_results.json` - Campaign 6 diagnostic results
- `investigation_results/maintain_headcount_results.json` - Alternative mode results
- `investigation_results/analysis_summary.md` - Detailed findings

### Key Metrics to Track
- Final headcount vs target
- Growth rates by year
- Total hires vs terminations
- Termination breakdown by reason/demographic
- Parameter access and usage validation

## 🎯 Expected Outcomes

### If New Hire Termination Engine is Broken
- **Fix**: Repair parameter access and event processing
- **Impact**: Immediate improvement in growth achievement
- **Timeline**: 1-2 days implementation + validation

### If Experienced Attrition is Overwhelming
- **Fix**: Adjust termination multipliers or add caps
- **Impact**: Gradual improvement with parameter tuning
- **Timeline**: 2-3 days parameter optimization + validation

### If Order-of-Operations is Flawed
- **Fix**: Restructure orchestrator sequence or add corrections
- **Impact**: Systematic improvement in target achievement
- **Timeline**: 3-4 days architectural changes + validation

### If Multiple Issues Exist
- **Fix**: Comprehensive solution addressing all identified problems
- **Impact**: Significant improvement in model reliability
- **Timeline**: 1 week implementation + extensive validation

## 🚨 Risk Mitigation

### Backup Plans
1. **If investigation reveals unfixable issues**: Document limitations and adjust expectations
2. **If fixes break other functionality**: Implement feature flags for rollback
3. **If validation fails**: Return to parameter-only optimization with documented constraints

### Quality Assurance
- Comprehensive unit tests for any logic changes
- Regression testing against previous campaign results
- Golden-run validation with known configurations

This investigation represents a critical transition from parameter optimization to fundamental model validation and enhancement, with the potential to unlock the positive headcount growth that has remained elusive across all previous campaigns.

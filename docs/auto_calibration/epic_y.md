New Epic & User Stories: Model Logic Refinement for Accurate Growth & Demographics

Given the critical discovery, before further large-scale tuning, we need a new Epic focused on addressing the suspected model logic issues.

Epic Y: Model Logic Investigation and Refinement for Headcount Growth & Demographic Accuracy

Goal: To investigate, identify, and implement necessary corrections or enhancements to the core simulation model logic. This will address the current inability to achieve target headcount growth and will improve the model's capability to accurately reflect desired age/tenure distributions when driven by appropriate parameters.

Here are the initial User Stories for this new Epic, based directly on your "Recommended Next Actions":

Prompt for Next Steps (Kicking off Epic Y):

"Team, Campaign 4 was a success in hitting our score target and, crucially, in pinpointing that the persistent headcount growth issue likely stems from model logic rather than just parameter settings. This completes User Story 3.3 for our tuning analysis.

We are now initiating a new critical phase: Epic Y: Model Logic Investigation and Refinement for Headcount Growth & Demographic Accuracy.

Let's begin with the high-priority investigations you recommended:

User Story Y.1: Investigate Target Growth Parameter Translation

As a Model Developer,
I want to thoroughly analyze how the global_params.target_growth input parameter is processed and translated into actual hiring targets or decisions within the simulation code (likely in cost_model/simulation.py or cost_model/engines/run_one_year.py and cost_model/engines/hire.py).
So that I can understand why a 6% target_growth input (as per Campaign 4's SEARCH_SPACE) still resulted in -0.85% actual headcount growth and identify any logical flaws, hardcoded limits, or incorrect calculations in this conversion process.
Tasks for Jules:
Trace the target_growth parameter from its definition in global_params through the simulation orchestrator to where hires_to_make (or equivalent) is determined.
Document the exact formula or logic used.
Identify any points where this logic might cap or incorrectly scale the intended growth.
User Story Y.2: Analyze Hiring vs. Termination Balance Dynamics

As a Model Developer,
I want to investigate the interplay and net balance between the number of hires generated (based on new_hire_rate and growth targets) and the total number of terminations occurring (from all sources: new hire attrition, experienced attrition across age/tenure/level) within a typical simulation year using a 'best' Campaign 4 configuration.
So that I can determine if overall termination volumes are systematically overwhelming even the aggressively increased hiring volumes, thus preventing positive headcount growth.
Tasks for Jules:
Using the outputs from a top-performing Campaign 4 run (e.g., config_060_...), extract:
Total hires per year.
Total terminations per year (broken down by reason/source if possible, e.g., new hire vs. experienced).
Analyze the net headcount change per year and compare it to the target_growth parameter used for that run.
User Story Y.3: Debug New Hire Processing and Initial Tenure Progression

As a Model Developer,
I want to specifically debug the new hire processing, including how their employee_birth_date is handled with extreme parameter values (as noted in Campaign 4 issues), and how these new hires initially populate and transition out of the <1 year tenure band.
So that I can fix any data processing errors related to new hires and better understand the drivers behind the persistently high <1 year tenure percentage.
Tasks for Jules:
Investigate the "birth date processing errors for extreme parameter values" encountered in Campaign 4. What were these errors and how can the logic be made more robust?
For a typical 'best' Campaign 4 config, trace a cohort of new hires from one year to the next. What proportion are terminating within their first year (compare to the tuned new_hire_termination_rate and tenure_multipliers.<1)? How does this contribute to the high <1 tenure band figure?
Let's start with User Story Y.1: Investigate Target Growth Parameter Translation. Please provide an analysis of how target_growth is currently used to derive hiring decisions in the codebase.

## User Story Y.1 Analysis: Target Growth Parameter Translation

### Executive Summary

I have completed a comprehensive analysis of how the `target_growth` parameter flows through the simulation system. The investigation reveals that the **target growth logic is mathematically sound and correctly implemented**, but there are several potential areas where the system might not achieve the intended growth rates.

### Complete Parameter Flow Analysis

#### 1. Parameter Source and Extraction
- **Source**: `global_params.target_growth` (e.g., 0.06 for 6% growth)
- **Extraction Point**: `cost_model/engines/run_one_year/orchestrator/hiring.py:139`
  ```python
  target_growth = getattr(year_context.global_params, 'target_growth', 0.0)
  ```
- **Default Fallback**: 0.0 (no growth) if parameter is missing

#### 2. Target EOY Calculation
- **Formula**: `target_eoy = round(start_count * (1 + target_growth))`
- **Implementation**: `cost_model/engines/run_one_year/utils.py:85`
- **Example**: 100 employees × (1 + 0.06) = 106 employees target

#### 3. Exact Targeting Logic (Core Algorithm)
The system uses the `manage_headcount_to_exact_target()` function which implements sophisticated logic:

**Key Steps:**
1. Calculate target EOY actives: `round(soy_actives * (1 + target_growth_rate))`
2. Determine survivors after experienced terminations: `soy_actives - num_markov_exits_existing`
3. Calculate net actives needed: `target_eoy_actives - survived_soy_actives`
4. Compute gross hires accounting for new hire attrition: `round(net_needed / (1 - new_hire_termination_rate))`

**Mathematical Example (6% growth scenario):**
- Start: 100 employees
- Target: round(100 × 1.06) = 106 employees
- Experienced terminations: 18 employees
- Survivors: 100 - 18 = 82 employees
- Net needed: 106 - 82 = 24 employees
- With 25% new hire termination rate: gross_hires = round(24 / 0.75) = 32 hires

#### 4. Implementation Verification
The exact targeting logic has been **thoroughly tested** with comprehensive unit tests covering:
- Normal growth scenarios
- Excess survivors requiring forced terminations
- Perfect match scenarios
- Edge cases (zero attrition, invalid parameters)

### Potential Issues Identified

#### 1. **New Hire Termination Rate Parameter Location**
The system searches multiple locations for `new_hire_termination_rate`:
```python
# Priority order:
1. global_params.attrition.new_hire_termination_rate
2. global_params.new_hire_termination_rate
3. Default fallback: 0.25 (25%)
```
**Risk**: If the parameter is not found in expected locations, the 25% default may not match the actual termination behavior.

#### 2. **Rounding Effects in Multi-Step Process**
The system uses `round()` at multiple stages:
- Target EOY calculation
- Gross hires calculation
- Individual termination decisions

**Risk**: Cumulative rounding errors could cause small deviations from exact targets.

#### 3. **New Hire Processing Robustness**
The hiring orchestrator logs warnings about "birth date processing errors for extreme parameter values" (mentioned in Campaign 4 issues).

**Risk**: Parameter validation failures could prevent hires from being generated correctly.

#### 4. **Termination Rate Consistency**
The gross hiring calculation assumes a specific new hire termination rate, but the actual termination engine may use different parameters or logic.

**Risk**: Mismatch between assumed and actual termination rates would cause systematic over/under-hiring.

### Diagnostic Recommendations

#### Immediate Actions:
1. **Parameter Audit**: Verify `new_hire_termination_rate` is correctly configured and accessible
2. **Termination Rate Validation**: Compare assumed vs. actual new hire termination rates in Campaign 4 results
3. **Birth Date Processing**: Investigate and fix the "extreme parameter values" errors
4. **End-to-End Validation**: Run controlled simulations with known parameters to verify the complete flow

#### Investigation Queries:
1. What is the actual `new_hire_termination_rate` being used in Campaign 4 configurations?
2. Are there any parameter validation failures in the hire generation process?
3. How do the actual termination rates compare to the configured rates?
4. Are there any systematic biases in the rounding logic?

### Conclusion

The target growth parameter translation logic is **mathematically correct and well-implemented**. The persistent headcount growth issues are likely caused by:
1. **Parameter mismatches** between hiring assumptions and termination reality
2. **Processing errors** in new hire generation with extreme parameter values
3. **Systematic termination rate differences** between configuration and execution

The next investigation (User Story Y.2) should focus on analyzing the actual hiring vs. termination balance in Campaign 4 results to identify where the mathematical expectations diverge from reality.

## User Story Y.2 Analysis: Hiring vs. Termination Balance Dynamics

### Executive Summary

I have completed a comprehensive analysis of the actual hiring and termination flows in the best Campaign 4 configuration (config_004_20250605_103917.yaml). The results reveal **catastrophic failures** in multiple core systems that completely explain the persistent headcount growth issues.

### Configuration Parameters Used
- **Target Growth**: 5.5% per year
- **New Hire Termination Rate**: 25% (configured)
- **New Hire Rate**: 50%
- **Simulation Period**: 3 years (2025-2027)

### Critical Findings

#### 1. **ZERO New Hire Terminations Occurring**
**Most Shocking Discovery**: Despite configuring a 25% new hire termination rate, **ZERO new hire terminations occurred** across all 3 simulation years.

**Evidence**:
- Year 2025: 5 hires, 0 new hire terminations (0.0% vs 25.0% expected)
- Year 2026: 16 hires, 0 new hire terminations (0.0% vs 25.0% expected)
- Year 2027: 23 hires, 0 new hire terminations (0.0% vs 25.0% expected)
- **Total**: 44 hires, 0 new hire terminations (0.0% vs 25.0% expected)

**Impact**: This represents a **-25.0 percentage point difference** between configured and actual new hire termination rates.

#### 2. **Massive Growth Target Failure**
**Target vs Actual**:
- **Expected Growth**: 17.4% total over 3 years (5.5% annually)
- **Actual Growth**: -1.0% (workforce actually shrank)
- **Growth Shortfall**: -18.4 percentage points
- **Headcount Shortfall**: -19 employees (expected 119, actual 100)

#### 3. **Systematic Hiring Insufficiency**
**Net Flow Analysis**:
- **Expected Net Change**: +18 employees
- **Actual Net Change**: -4 employees
- **Difference**: -22 employees

**Yearly Breakdown**:
- 2025: 5 hires, 6 terminations (net -1)
- 2026: 16 hires, 17 terminations (net -1)
- 2027: 23 hires, 25 terminations (net -2)

### Root Cause Analysis

#### **Primary Issue: New Hire Termination Engine Failure**
The complete absence of new hire terminations indicates a **fundamental breakdown** in the new hire termination processing:

1. **Parameter Location Mismatch**: The new hire termination engine is not finding or using the configured `new_hire_termination_rate: 0.25`
2. **Logic Bypass**: New hire termination logic may be completely bypassed or disabled
3. **Event Processing Failure**: New hire termination events are not being generated or applied

#### **Secondary Issue: Hiring Algorithm Compensation Failure**
The hiring algorithm calculates gross hires based on the assumption that 25% of new hires will terminate:
- **Formula**: `gross_hires = net_needed / (1 - 0.25) = net_needed / 0.75`
- **Reality**: Since 0% actually terminate, the system gets 33% more survivors than expected
- **Result**: Systematic over-hiring that the exact targeting cannot compensate for

#### **Tertiary Issue: Experienced Termination Overwhelm**
Even with the hiring issues, experienced employee terminations are overwhelming the hiring capacity:
- Total terminations (48) exceed total hires (44) by 4 employees
- This suggests experienced termination rates may also be higher than the hiring algorithm compensates for

### Implications for User Story Y.1 Findings

This analysis **validates and extends** the Y.1 findings:

1. **Parameter Location Issues Confirmed**: The new hire termination rate is clearly not being found or used by the termination engine
2. **Mathematical Logic Validated**: The exact targeting logic is mathematically correct, but operates on false assumptions
3. **Processing Errors Identified**: The "birth date processing errors for extreme parameter values" may be preventing new hire termination events from being generated

### Immediate Action Items

#### **Critical Priority**:
1. **Debug New Hire Termination Engine**: Investigate why `new_hire_termination_rate: 0.25` is not being applied
2. **Parameter Location Audit**: Verify the termination engine is reading from the correct configuration location
3. **Event Generation Validation**: Ensure new hire termination events are being generated and applied

#### **High Priority**:
1. **End-to-End Parameter Flow**: Trace the complete flow from configuration → hiring assumptions → termination reality
2. **Experienced Termination Audit**: Verify experienced termination rates match hiring algorithm expectations
3. **Birth Date Processing Fix**: Resolve the extreme parameter value errors mentioned in Campaign 4

### Conclusion

The persistent headcount growth issues are **NOT** due to flawed model logic or parameter tuning, but due to **complete system failures**:

1. **New hire termination engine is non-functional** (0% vs 25% configured)
2. **Hiring algorithm operates on false assumptions** about termination rates
3. **Parameter flow is broken** between configuration and execution

This explains why even aggressive parameter tuning in Campaigns 1-4 failed to achieve growth targets. The fundamental employee flow mechanisms are not working as designed.

**Next Steps**: User Story Y.3 should focus on debugging the new hire processing and termination logic to restore basic system functionality before any further parameter tuning.

---

## 🔍 COMPREHENSIVE HEADCOUNT MANAGEMENT INVESTIGATION - CAMPAIGN 6 ANALYSIS

### Executive Summary

Following Campaign 6's achievement of the best-ever calibration score (0.0723) but persistent -1.69% headcount growth, I conducted a comprehensive investigation of the headcount management logic. This analysis reveals **critical insights** that explain the persistent negative growth despite aggressive parameter tuning.

### 🎯 Key Findings from Code Investigation

#### 1. **Exact Targeting Logic is Mathematically Sound**

The `manage_headcount_to_exact_target()` function in `cost_model/engines/run_one_year/utils.py` implements robust logic:

<augment_code_snippet path="cost_model/engines/run_one_year/utils.py" mode="EXCERPT">
````python
def manage_headcount_to_exact_target(
    soy_actives: int,
    target_growth_rate: float,
    num_markov_exits_existing: int,
    new_hire_termination_rate: float
) -> tuple[int, int]:
    # Step 1: Calculate target EOY actives
    target_eoy_actives = round(soy_actives * (1 + target_growth_rate))

    # Step 2: Determine survivors from initial Markov exits
    survived_soy_actives = soy_actives - num_markov_exits_existing

    # Step 3: Calculate net actives needed from new hires
    net_actives_needed_from_hiring_pool = target_eoy_actives - survived_soy_actives

    # Step 4: Calculate gross hires accounting for new hire attrition
    calculated_gross_new_hires = round(
        net_actives_needed_from_hiring_pool / (1 - new_hire_termination_rate)
    )
````
</augment_code_snippet>

#### 2. **Critical Discovery: `maintain_headcount` Parameter Exists**

Found in `cost_model/config/models.py`:

<augment_code_snippet path="cost_model/config/models.py" mode="EXCERPT">
````python
def _check_maintain_headcount_vs_growth(self) -> None:
    """Ensure maintain_headcount and annual_growth_rate are used logically."""
    if self.maintain_headcount and self.annual_growth_rate != 0.0:
        logger.warning(
            f"maintain_headcount is True, but annual_growth_rate is {self.annual_growth_rate:.2%}. Growth rate will be ignored."
        )
````
</augment_code_snippet>

**Status**: All configurations use `maintain_headcount: false`, meaning exact targeting is active.

#### 3. **Execution Sequence Analysis**

The `run_one_year()` orchestrator follows this critical sequence:

1. **Compensation Events** (EVT_COMP, EVT_COLA)
2. **Markov Promotions/Exits** (experienced only)
3. **Hazard-based Terminations** (experienced only) ← **Major attrition source**
4. **Exact Targeting Calculation** (uses survivors after step 3)
5. **Hiring** (adds new employees)
6. **Forced Terminations** (if survivors exceed target)
7. **New Hire Terminations** ← **Critical final step**

### 📊 Campaign 6 Parameter Analysis

**Best Configuration Parameters**:
```yaml
target_growth: 0.07                    # 7% growth target
new_hire_rate: 0.50                   # 50% of start count
new_hire_termination_rate: 0.25       # 25% of new hires terminate
base_rate_for_new_hire: 0.04          # 4% base termination rate

termination_hazard:
  age_multipliers:
    <30: 0.4      # 40% of base rate
    30-39: 0.6    # 60% of base rate
    40-49: 1.0    # 100% of base rate
    50-59: 2.0    # 200% of base rate
    60-65: 4.0    # 400% of base rate
    65+: 8.0      # 800% of base rate
  tenure_multipliers:
    <1: 0.1       # 10% of base (0.4% effective)
    1-3: 0.6      # 60% of base (2.4% effective)
    3-5: 0.5      # 50% of base (2.0% effective)
    5-10: 0.2     # 20% of base (0.8% effective)
    10-15: 0.15   # 15% of base (0.6% effective)
    15+: 0.3      # 30% of base (1.2% effective)
```

### 🚨 Root Cause Hypotheses

#### **Hypothesis 1: Experienced Employee Attrition Overwhelm**
Even with aggressive tuning, experienced employee termination rates (especially 60+ age bands with 4x-8x multipliers) may be systematically overwhelming hiring capacity.

#### **Hypothesis 2: New Hire Termination Rate Mismatch**
The exact targeting assumes 25% new hire termination rate, but:
- Previous Y.2 analysis showed **0% actual new hire terminations** in Campaign 4
- If this persists, the system gets 33% more survivors than expected
- This could cause systematic over-hiring that disrupts the balance

#### **Hypothesis 3: Order-of-Operations Timing Issue**
The exact targeting calculation happens **after** experienced terminations but **before** new hire terminations, creating potential mismatches in final headcount.

#### **Hypothesis 4: Parameter Location/Access Issues**
The `new_hire_termination_rate` parameter is searched in multiple locations:
```python
# Priority order:
1. global_params.attrition.new_hire_termination_rate
2. global_params.new_hire_termination_rate
3. Default fallback: 0.25 (25%)
```

If the parameter isn't found in expected locations, defaults may not match actual behavior.

### 🔧 IMMEDIATE INVESTIGATION PLAN

#### **Task 1: Quantify Experienced Employee Attrition Impact**

**Objective**: Measure the actual magnitude of experienced employee terminations in Campaign 6 best config.

**Action Items**:
1. Run diagnostic simulation with Campaign 6 best config (`campaign_6_results/best_config.yaml`)
2. Extract detailed termination data:
   - Total experienced employee terminations per year
   - Breakdown by age band and tenure band
   - Comparison of experienced vs new hire termination volumes
3. Calculate effective termination rates vs configured parameters

**Expected Outcome**: Quantify if experienced attrition is systematically overwhelming hiring capacity.

#### **Task 2: Test `maintain_headcount: true` Mode**

**Objective**: Determine if the `maintain_headcount` parameter can force positive growth.

**Action Items**:
1. Modify Campaign 6 best config to set `maintain_headcount: true`
2. Run simulation with identical parameters except headcount mode
3. Compare results:
   - Final headcount vs target
   - Impact on age/tenure/pay distributions
   - Any warnings or errors generated

**Expected Outcome**: Understand if this parameter bypasses the exact targeting issues.

#### **Task 3: Validate New Hire Termination Rate Assumptions**

**Objective**: Verify the new hire termination engine is working correctly.

**Action Items**:
1. Add detailed logging to new hire termination processing
2. Trace actual new hire termination rates in Campaign 6 simulation
3. Compare configured (25%) vs actual termination rates
4. Verify parameter location and access in termination engine

**Expected Outcome**: Confirm if the Y.2 finding (0% new hire terminations) persists in Campaign 6.

#### **Task 4: Order-of-Operations Validation**

**Objective**: Verify the exact targeting logic achieves intended final headcount.

**Action Items**:
1. Add comprehensive headcount logging at each orchestrator step
2. Track headcount progression: SOY → Experienced Terms → Hiring → New Hire Terms → EOY
3. Verify exact targeting assertions are met within tolerance
4. Identify any systematic discrepancies

**Expected Outcome**: Confirm the orchestrator sequence produces expected results.

### 🎯 POTENTIAL SOLUTIONS

#### **Solution 1: Parameter Adjustment Strategy**
- Reduce `new_hire_termination_rate` from 0.25 to 0.15-0.20
- Fine-tune experienced employee termination multipliers
- Add caps on total termination rates by age/tenure band

#### **Solution 2: Enhanced Exact Targeting Logic**
- Implement iterative targeting with mid-year adjustments
- Add tolerance bands for target achievement (±1-2 employees)
- Improve rounding precision in multi-step calculations

#### **Solution 3: Dynamic Headcount Management**
- Implement quarterly hiring cycles instead of annual
- Add "true-up" hiring based on actual (not projected) attrition
- Consider adaptive termination rate estimation

#### **Solution 4: Forced Headcount Maintenance**
- Utilize `maintain_headcount: true` mode if it proves effective
- Implement hybrid approach: exact targeting with headcount floor
- Add headcount monitoring and correction mechanisms

### 📊 SUCCESS CRITERIA

A "fixed" model should achieve:

1. **Headcount Growth Accuracy**: Actual growth within ±0.5% of `target_growth` parameter
2. **Predictable Response**: Linear relationship between `target_growth` and actual growth
3. **Maintained Calibration**: No degradation in age/tenure/pay growth metrics
4. **Robustness**: Consistent performance across different parameter ranges
5. **Transparency**: Clear logging of all headcount management decisions

### 🚀 EXECUTION ROADMAP

#### **Phase 1: Diagnostic Analysis (1-2 days)**
1. Execute Tasks 1-4 to gather comprehensive data
2. Synthesize findings to identify primary limiting factor
3. Determine if issue is parameter-based or logic-based

#### **Phase 2: Targeted Fix Implementation (2-3 days)**
1. Implement the most promising solution based on Phase 1 findings
2. Add enhanced logging and validation
3. Create unit tests for new logic

#### **Phase 3: Validation Campaign (1-2 days)**
1. Run focused auto-tuning with 50-100 iterations
2. Verify fix achieves positive headcount growth
3. Confirm maintained calibration quality

#### **Phase 4: Production Readiness (1 day)**
1. Document the fix and its implications
2. Update configuration templates
3. Prepare deployment recommendations

### 🔍 CRITICAL QUESTIONS TO RESOLVE

1. **Is the new hire termination engine functional in Campaign 6?** (Building on Y.2 findings)
2. **Are experienced termination rates accurately reflected in hiring calculations?**
3. **Does `maintain_headcount: true` provide a viable workaround?**
4. **What is the actual vs expected termination rate variance?**
5. **Are there systematic rounding or timing issues in the orchestrator?**

This investigation represents a **critical transition** from parameter optimization to fundamental model validation and enhancement. The findings will determine whether the headcount growth challenge requires parameter adjustments, logic fixes, or architectural changes to the simulation engine.
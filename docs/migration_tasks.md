## 8 Migration Task Breakdown – **File-Tree Aware**

This checklist maps each design change clearly to your exact repository layout, ensuring the engineer knows precisely where to edit and which new modules/tests to add. All paths are relative to the project root.

### Step-by-Step Migration Guide

**1. Update Workflow Sequence**

**File:**
`cost_model/engines/run_one_year/orchestrator.py`

**Actions:**

* Replace existing orchestration logic with the ordered sequence:

  1. `markov_promotion.apply_markov_promotions()` (experienced only)
  2. `term.run()` (experienced hazard exits, excluding Markov IDs)
  3. Update snapshot → survivors
  4. Compute targets and calculate `gross_hires` (see Step 2)
  5. `hire.run()` (pass `gross_hires`)
  6. Merge hires into snapshot
  7. `term.run_new_hires_deterministic()` (see Step 3)
  8. Final snapshot → `validation.validate_eoy_snapshot()`

**Snippet:**

```python
from .validation import validate_eoy_snapshot
```

---

**2. Headcount Calculation Logic**

**File:**
`cost_model/engines/run_one_year/utils.py` (create or update)

**Actions:**

* Add precise calculation function:

```python
def compute_headcount_targets(start_count, survivor_count, target_growth, nh_term_rate):
    target_eoy = math.ceil(start_count * (1 + target_growth))
    net_hires = max(0, target_eoy - survivor_count)
    gross_hires = math.ceil(net_hires / (1 - nh_term_rate)) if nh_term_rate < 1 else net_hires
    return target_eoy, net_hires, gross_hires
```

* Call this from the orchestrator, logging results clearly:

```python
logger.info(f"[DEBUG-HIRE] Start: {start_count}, Survivors: {survivor_count}, Net Hires: {net_hires}, Gross Hires: {gross_hires}")
```

---

**3. Deterministic New-Hire Terminations**

**File:**
`cost_model/engines/run_one_year/nh_termination.py` (new file) or include in `comp_term.py`

**Actions:**

* Add deterministic termination helper:

```python
def run_new_hires_deterministic(snapshot, hazard_slice, rng, year):
    nh_mask = snapshot[EMP_HIRE_DATE] >= pd.Timestamp(f"{year}-01-01")
    nh_rate = float(hazard_slice[NEW_HIRE_TERM_RATE].iloc[0])
    num_exits = round(nh_mask.sum() * nh_rate)
    exit_ids = rng.choice(snapshot.loc[nh_mask, EMP_ID], size=num_exits, replace=False)
    return create_term_events(exit_ids, comment="nh-deterministic")
```

* Integrate into orchestrator at the new-hire termination step.

---

**4. Snapshot Validation & Enhanced Logging**

**File:**
`cost_model/engines/run_one_year/validation.py`

**Actions:**

* Add final snapshot validation:

```python
def validate_eoy_snapshot(final_snapshot, target_eoy):
    assert final_snapshot[EMP_ACTIVE].sum() == target_eoy, (
        f"EOY headcount {final_snapshot[EMP_ACTIVE].sum()} ≠ {target_eoy}")
    assert not final_snapshot[EMP_ID].duplicated().any(), "Duplicate EMP_IDs detected!"
```

* Log the outcome clearly:

```python
logger.info(f"[RESULT] EOY={final_snapshot[EMP_ACTIVE].sum()} (target={target_eoy})")
```

---

**5. Configuration Parameters Cleanup**

**Files:**
`cost_model/config/models.py`, `config/dev_tiny.yaml`

**Actions:**

* YAML: Ensure a single `attrition` block with keys clearly defined:

```yaml
attrition:
  annual_termination_rate: 0.15
  new_hire_termination_rate: 0.25
  use_expected_attrition: false
```

* Python: Update the Pydantic or Typed models to match.

---

**6. Edge-Case Handling**

**File:**
`cost_model/engines/run_one_year/orchestrator.py`

**Actions:**

* Handle flags for special cases:

```python
if params.maintain_headcount:
    target_growth = 0.0

if params.prevent_all_hiring:
    gross_hires = 0
```

---

**7. Unit Testing**

**File:**
`tests/engines/test_headcount.py` (new test file)

**Actions:**

* Create fixtures with controlled scenarios (100 employees, 3% growth, 25% NH attrition)
* Validate year-by-year headcount consistency over multiple years
* Ensure no duplicate IDs appear

---

**8. CLI & Runner Wiring**

**Files:**
`cost_model/projections/runner.py`, `cost_model/projections/cli.py`

**Actions:**

* Replace existing direct calls to old dynamics functions with the new orchestrator:

```python
from cost_model.engines.run_one_year.orchestrator import run_one_year
# Use this function consistently in place of dynamics/engine.run_dynamics_for_year()
```

* Ensure SimpleNamespace configurations are passed correctly.

---

### Deliverables Checklist

* [ ] PR with above changes clearly documented
* [ ] Successful CI pipeline run including new tests
* [ ] Logs demonstrating `[CONFIG]`, `[DEBUG-HIRE]`, and `[RESULT]` for years 2025-2029

Ping me once the branch is ready for review or if further clarifications are needed.
## 8 Migration Task Breakdown – **File-Tree Aware**

This checklist maps each design change clearly to your exact repository layout, ensuring the engineer knows precisely where to edit and which new modules/tests to add. All paths are relative to the project root.

### Step-by-Step Migration Guide

**1. Update Workflow Sequence**

**File:**
`cost_model/engines/run_one_year/orchestrator.py`

**Actions:**

* Replace existing orchestration logic with the ordered sequence:

  1. `markov_promotion.apply_markov_promotions()` (experienced only)
  2. `term.run()` (experienced hazard exits, excluding Markov IDs)
  3. Update snapshot → survivors
  4. Compute targets and calculate `gross_hires` (see Step 2)
  5. `hire.run()` (pass `gross_hires`)
  6. Merge hires into snapshot
  7. `term.run_new_hires_deterministic()` (see Step 3)
  8. Final snapshot → `validation.validate_eoy_snapshot()`

**Snippet:**

```python
from .validation import validate_eoy_snapshot
```

---

**2. Headcount Calculation Logic**

**File:**
`cost_model/engines/run_one_year/utils.py` (create or update)

**Actions:**

* Add precise calculation function:

```python
def compute_headcount_targets(start_count, survivor_count, target_growth, nh_term_rate):
    target_eoy = math.ceil(start_count * (1 + target_growth))
    net_hires = max(0, target_eoy - survivor_count)
    gross_hires = math.ceil(net_hires / (1 - nh_term_rate)) if nh_term_rate < 1 else net_hires
    return target_eoy, net_hires, gross_hires
```

* Call this from the orchestrator, logging results clearly:

```python
logger.info(f"[DEBUG-HIRE] Start: {start_count}, Survivors: {survivor_count}, Net Hires: {net_hires}, Gross Hires: {gross_hires}")
```

---

**3. Deterministic New-Hire Terminations**

**File:**
`cost_model/engines/run_one_year/nh_termination.py` (new file) or include in `comp_term.py`

**Actions:**

* Add deterministic termination helper:

```python
def run_new_hires_deterministic(snapshot, hazard_slice, rng, year):
    nh_mask = snapshot[EMP_HIRE_DATE] >= pd.Timestamp(f"{year}-01-01")
    nh_rate = float(hazard_slice[NEW_HIRE_TERM_RATE].iloc[0])
    num_exits = round(nh_mask.sum() * nh_rate)
    exit_ids = rng.choice(snapshot.loc[nh_mask, EMP_ID], size=num_exits, replace=False)
    return create_term_events(exit_ids, comment="nh-deterministic")
```

* Integrate into orchestrator at the new-hire termination step.

---

**4. Snapshot Validation & Enhanced Logging**

**File:**
`cost_model/engines/run_one_year/validation.py`

**Actions:**

* Add final snapshot validation:

```python
def validate_eoy_snapshot(final_snapshot, target_eoy):
    assert final_snapshot[EMP_ACTIVE].sum() == target_eoy, (
        f"EOY headcount {final_snapshot[EMP_ACTIVE].sum()} ≠ {target_eoy}")
    assert not final_snapshot[EMP_ID].duplicated().any(), "Duplicate EMP_IDs detected!"
```

* Log the outcome clearly:

```python
logger.info(f"[RESULT] EOY={final_snapshot[EMP_ACTIVE].sum()} (target={target_eoy})")
```

---

**5. Configuration Parameters Cleanup**

**Files:**
`cost_model/config/models.py`, `config/dev_tiny.yaml`

**Actions:**

* YAML: Ensure a single `attrition` block with keys clearly defined:

```yaml
attrition:
  annual_termination_rate: 0.15
  new_hire_termination_rate: 0.25
  use_expected_attrition: false
```

* Python: Update the Pydantic or Typed models to match.

---

**6. Edge-Case Handling**

**File:**
`cost_model/engines/run_one_year/orchestrator.py`

**Actions:**

* Handle flags for special cases:

```python
if params.maintain_headcount:
    target_growth = 0.0

if params.prevent_all_hiring:
    gross_hires = 0
```

---

**7. Unit Testing**

**File:**
`tests/engines/test_headcount.py` (new test file)

**Actions:**

* Create fixtures with controlled scenarios (100 employees, 3% growth, 25% NH attrition)
* Validate year-by-year headcount consistency over multiple years
* Ensure no duplicate IDs appear

---

**8. CLI & Runner Wiring**

**Files:**
`cost_model/projections/runner.py`, `cost_model/projections/cli.py`

**Actions:**

* Replace existing direct calls to old dynamics functions with the new orchestrator:

```python
from cost_model.engines.run_one_year.orchestrator import run_one_year
# Use this function consistently in place of dynamics/engine.run_dynamics_for_year()
```

* Ensure SimpleNamespace configurations are passed correctly.

---

### Deliverables Checklist

* [ ] PR with above changes clearly documented
* [ ] Successful CI pipeline run including new tests
* [ ] Logs demonstrating `[CONFIG]`, `[DEBUG-HIRE]`, and `[RESULT]` for years 2025-2029

Ping me once the branch is ready for review or if further clarifications are needed.

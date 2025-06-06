Workforce Simulation – Hiring & Termination Flow

Purpose  Provide a developer‑readable specification of exactly when and how hires and terminations (including Markov exits) are generated each projection year, so the engine stays numerically consistent with the configured growth and attrition targets.

⸻

1  Glossary

Term	Meaning
SOY	Start‑Of‑Year (snapshot as of 1 Jan Y)
EOY	End‑Of‑Year (snapshot as of 31 Dec Y)
Target Growth	Desired % increase in total active headcount vs SOY
Annual Termination Rate	Hazard‑based attrition rate applied to experienced employees
NH Termination Rate	Probability a new hire exits before year‑end
Net Hires	Hires needed after replacing attrition to reach headcount target
Gross Hires	Net Hires grossed‑up by NH attrition: ⌈ net / (1 − NH‑rate) ⌉


⸻

2  High‑Level Timeline (per projection year Y)
	1.	Load config + hazard slice for Y.
	2.	Markov promotions & exits (experienced only).
	3.	Hazard‑based terminations (experienced only).
	4.	Snapshot update → survivors.
	5.	Compute target headcount & hires.
	6.	Generate hires (gross).
	7.	Apply hires to snapshot.
	8.	Run new‑hire terminations.
	9.	Final snapshot + integrity checks.

The diagram below shows the ordering; terminations always happen before hires so growth math is based on actual survivors, and new‑hire terminations run after hires so the EOY headcount lands exactly on target.

SOY  ──► Markov exits ─► Hazard exits ─► survivors
                              │
                              ▼
            Compute target ► net ► gross ► EVT_HIREs
                              │
                              ▼
            Apply hires ─► NH exits ─► EOY snapshot


⸻

3  Detailed Steps & Formulas

3.1   Initial Counts

start_count = prev_snapshot[EMP_ACTIVE].sum()  # all actives, experienced + NH

3.2   Markov Promotions & Exits
	•	Run apply_markov_promotions() on experienced subset.
	•	Promotions → EVT_PROMOTION, optional EVT_RAISE.
	•	Exits → EVT_TERM (meta="markov‑exit").
	•	Deduplicate: keep a set markov_exit_ids to ignore duplicates later.

3.3   Experienced Attrition

term_frames = term.run(snapshot_exp_non_markov, hazard_slice, rng, deterministic_term)

	•	Create EVT_TERM for each departure.

3.4   Post‑Termination Snapshot

temp_snap = snapshot.update(prev_snapshot, core_events, year)

	•	survivors = temp_snap[EMP_ACTIVE & not_terminated]
	•	survivor_count = survivors.sum()
	•	total_attrition = start_count - survivor_count

3.5   Headcount Targets & Hires

target_eoy = ceil(start_count * (1 + target_growth))
net_hires  = max(0, target_eoy - survivor_count)
gross_hires = ceil(net_hires / (1 - nh_term_rate))  # if nh_term_rate < 1

3.6   Generate Hires

hire_events = hire.run(temp_snap, gross_hires, hazard_slice, rng, census_template, params, terminated_events)

	•	Each hire gets an EVT_HIRE + matching EVT_COMP.
	•	Compensation sampling uses current level distribution via sample_mixed_new_hires().

3.7   Apply Hires & Annual Comp Bump
	1.	Merge hire_events into snapshot.
	2.	Identify new_hire_ids (to exclude from comp bump).
	3.	Apply COLA → merit bumps to experienced cohort via apply_comp_bump().

3.8   New‑Hire Terminations

nh_term_df = term.run_new_hires(snap_with_hires, hazard_slice, rng, year, deterministic=True)

	•	Removes exactly round(gross_hires * nh_term_rate) heads so EOY lands on target.

3.9   Final Snapshot & Validation

final_snapshot = snapshot.update(snap_with_hires, nh_term_df, year)
assert final_snapshot[EMP_ACTIVE].sum() == target_eoy
assert not final_snapshot[EMP_ID].duplicated().any()


⸻

4  Configuration Parameters

YAML Path	Used In	Notes
global_parameters.target_growth	step 3.5	decimal (0.03 = 3 %)
global_parameters.attrition.annual_termination_rate	hazard pre‑calc	drives experienced hazard rates
global_parameters.attrition.new_hire_termination_rate	step 3.5, 3.8	for gross‑up & NH exits
global_parameters.maintain_headcount	overrides growth to 0	pure replacement mode
global_parameters.prevent_all_hiring	skips steps 3.5‑3.8	freeze mode


⸻

5  Logging & Observability
	•	[CONFIG] – log growth & attrition parameters on load.
	•	[YR=Y] Workforce Plan – summary of start, attrition, hires, target.
	•	[DEBUG‑HIRE] – inputs to hire.run() (sanity for gross vs net).
	•	Integrity asserts – headcount equality and duplicate EMP_ID check.

⸻

6  Example Walk‑Through (Growth = 3 %, NH term = 25 %)

Metric	Value
SOY actives	100
Markov exits	 5
Hazard exits	 12
Survivors	83
Target EOY (3 %)	⌈100 × 1.03⌉ = 103
Net hires	 20
Gross hires	⌈20 / 0.75⌉ = 27
NH exits (deterministic)	round(27 × 0.25) = 7
EOY actives	83 + 27 − 7 = 103 ✔︎


⸻

7  Edge‑Case Rules
	1.	If maintain_headcount=True ⇒ set target_growth = 0.
	2.	If growth < 0 (shrink) ⇒ net_hires may be negative; do not create hires, optionally trigger layoffs.
	3.	If prevent_all_hiring=True ⇒ skip hire logic entirely.

⸻

8  Migration Task Breakdown (Developer-Ready)

Below is a concrete checklist that maps each section of this spec to the files you need to touch, the code you should paste, and the tests that must pass.

Step	File(s) / Function(s)	What to change	Drop-in snippet
1. Sequencing	dynamics/engine.py → run_dynamics_for_year()	Replace current mixed flow with ordered calls	```python

new canonical order

markov_events  = markov.apply_markov_promotions(snapshot_exp, promo_time, rng, promo_cfg)
hazard_terms   = term.run(snapshot_exp - markov_exits, hazard_slice, rng, deterministic)
core_events    = pd.concat([markov_events, hazard_terms], ignore_index=True)
update_snapshot(core_events)  # survivors

compute targets …

hire_events    = hire.run(temp_snap, gross_hires, hazard_slice, rng, template, params)
apply_snapshot(hire_events)
nh_exit_events = term.run_new_hires_deterministic(snap_with_hires, hazard_slice, rng, year)
final_snapshot = apply_snapshot(nh_exit_events)

| **2. Headcount maths** | same file | Insert after survivors calc | ```python
target_eoy  = math.ceil(start_count * (1 + target_growth))
net_hires   = max(0, target_eoy - survivor_count)
gross_hires = math.ceil(net_hires / (1 - nh_term_rate))
``` |
| **3. Deterministic NH exits** | `dynamics/termination.py` | Add helper | ```python
def run_new_hires_deterministic(snapshot, hazard_slice, rng, year):
    nh_mask  = snapshot[EMP_HIRE_DATE] >= pd.Timestamp(f"{year}-01-01")
    nh_ids   = snapshot.loc[nh_mask, EMP_ID]
    k        = round(len(nh_ids) * hazard_slice[NEW_HIRE_TERMINATION_RATE].iloc[0])
    exit_ids = rng.choice(nh_ids, size=k, replace=False)
    return create_term_events(exit_ids, reason="nh-deterministic", time=f"{year}-12-31")
``` |
| **4. Validation & logging** | wherever snapshot finalised | Add asserts + INFO lines | ```python
assert final_snapshot[EMP_ACTIVE].sum() == target_eoy,
       f"EOY headcount {final_snapshot[EMP_ACTIVE].sum()} ≠ {target_eoy}"
logger.info("[RESULT] EOY=%d (target=%d)", final_snapshot[EMP_ACTIVE].sum(), target_eoy)
``` |
| **5. Config clean-up** | `config/dev_tiny.yaml` | Remove duplicate `attrition` block, keep: | ```yaml
attrition:
  annual_termination_rate: 0.15
  new_hire_termination_rate: 0.25
  use_expected_attrition: false
``` |
| **6. Edge-case switches** | `engine.py` | implement flags | ```python
if params.maintain_headcount:
    target_growth = 0.0
if params.prevent_all_hiring:
    gross_hires = 0
``` |
| **7. Unit tests** | `tests/dynamics/test_headcount.py` | new fixture: growth = 3 %, nh_term = 25 % | ensure `EOY == ceil(start*1.03)` for 5-year run |

### Extra Logging Hooks
Add at boot:
```python
logger.info("[CONFIG] growth=%.3f nh_term=%.2f exp_term=%.2f", target_growth, nh_term_rate, annual_term_rate)


⸻

Deliverables
	1.	PR with refactored dynamics/engine.py, new deterministic NH exit helper, and updated YAML.
	2.	Green test suite with new test_headcount.py.
	3.	Screenshot of log excerpt showing [RESULT] lines hitting target for 2025-2029.

⸻

Ping me once you have the first draft branch; I’ll review the code diff and the run-time logs.
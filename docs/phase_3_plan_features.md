# Phase 3 – Layer in Plan Features (Eligibility, Deferrals, AE/AI)

With headcount & comp locked, introduce plan-specific mechanics to project participation and employer cost.

---
## 1  Eligibility Engine
1.  Inputs: `age`, `service_years`, `config.plan_rules.eligibility.{age,service}`.
2.  Output: flag/event `ELIGIBLE_DATE` per employee.
3.  Update snapshots with `is_plan_eligible` boolean.

Tasks
- [ ] Flesh out `plan_rules/eligibility.py.determine_eligibility()` to emit EVT_ELIGIBLE events.
- [ ] Add test `tests/plan_rules/test_eligibility.py` using edge ages/service.

---
## 2  Auto-Enrollment (AE)
1. Use eligibility events → when employee crosses eligibility, trigger AE after `window_months` unless opted-out.
2. `auto_enrollment.py` already has constants; ensure event types `EVT_AE_ENROLL` exist.

Tasks
- [ ] Complete `plan_rules/auto_enrollment.apply_auto_enrollment()`.
- [ ] Add snapshot columns `is_enrolled`, `deferral_rate`.

---
## 3  Deferrals & Auto-Increase (AI)
1. Rules in config: `auto_increase_pct`, `max_deferral`, `deferral_cap_pct_of_comp`.
2. Events: `EVT_DEFERRAL_CHANGE` yearly.

Tasks
- [ ] Implement `plan_rules/auto_increase.apply_auto_increase()`.
- [ ] Ensure yearly step in dynamics engine calls it after comp events.

---
## 4  Employer Match / NEC
- [ ] Expand `plan_rules/contributions.calculate_contributions()` to use tiered match formulas (see `docs/match_formulas.md`).
- [ ] Unit tests with IRS limit edge cases.

---
## 5  Definition of Done
- Projection outputs include participation & contribution dollars.
- Dashboards show participation rate, avg deferral, employer cost.
- All tests green; CI passes.

# Phase 2 – Get Compensation Right

Once headcount classification is rock-solid, tackle **compensation accuracy** so cost projections and employer match calcs are meaningful.

---
## 1  Clarify compensation sources
| Component | Source | Engine/File |
|-----------|--------|-------------|
| Base pay  | `event_log` → `EVT_COMP_CHANGE` | `dynamics/compensation.py` |
| Hiring comp | EVT_HIRE (starting comp) | `dynamics/hiring.py` |
| Proration | snapshot_build + utils | `state/snapshot_build.py` |

Ensure every comp-related event lands in the event-log and flows to the snapshot via `snapshot_update` in chronological order.

---
## 2  Audit compensation change logic
-   [ ] Review `dynamics/compensation.generate_comp_events()` – verify raise schedule, distribution (flat vs percent), and rate sources (`config.comp_increase_*`).
-   [ ] Confirm proration formula in `compensation._prorate_comp()` uses `days_worked / days_in_year` after **termination date** enforcement.
-   [ ] Validate employer-match rules reference **prorated comp** (`employee_plan_year_compensation`).

---
## 3  Build golden dataset
Create a tiny census & event-log where comp changes are known:
```csv
# census.csv
employee_id,start_date,comp
1,2024-01-01,100000

# events.csv
2024-07-01,EVT_COMP_CHANGE,1,110000
2025-06-30,EVT_TERM,1
```
Expected plan-year comp for 2024: `100000 * 0.5 + 110000 * 0.5 = 105000`
Expected for 2025: `110000 * 0.5 = 55000`
Add assertions in `tests/comp/test_prorated_comp.py`.

---
## 4  Dashboards
Update `notebooks/headcount_dashboard.ipynb` or create `compensation_dashboard.ipynb` with:
* Total payroll $ vs headcount
* Avg comp per FTE vs year
* Distribution of raises (%)

---
## 5  Definition of Done
- Snapshot contains columns `annual_base_comp`, `employee_plan_year_compensation` populated and matching golden tests.
- Employer match calc uses correct comp.
- All comp unit tests green.

Refactor roadmap for utils/plan_rules.py

# Refactor Complete
**Status**: Completed as of 2025-04-22
**Module Layout**:
```
utils/
├── plan_rules.py       # Facade orchestrating rule modules
└── rules/
    ├── eligibility.py      # Eligibility logic
    ├── auto_enrollment.py  # Auto-enrollment logic
    ├── auto_increase.py    # Auto-increase logic
    ├── formula_parsers.py  # Match/NEC parsing helpers
    ├── contributions.py    # Contributions & limit engine
    └── response.py         # Plan-change deferral response
```
**Public API (`utils.plan_rules`)**:
- `determine_eligibility(df, scenario_config, simulation_year_end_date)`
- `apply_auto_enrollment(df, plan_rules, year_dates)`
- `apply_auto_increase(df, plan_rules, year)`
- `apply_plan_change_deferral_response(df, current_cfg, baseline_cfg, year, start_year)`
- `calculate_contributions(df, scenario_config, simulation_year, start_date, end_date)`
- `process_year(df, scenario_cfg, baseline_cfg, year_dates)`

**Configuration Example** (`match_change_response`):
```yaml
plan_rules:
  ...
  match_change_response:
    enabled: true
    increase_probability: 0.25  # Probability of bump to optimal
    increase_target: optimal    # Only 'optimal' supported currently
```

(lean version – puts the code separation first, leaves full pytest coverage for a later sprint)

⸻

1 · Why refactor now, test later?
	•	The file is ~1 700 LOC, mixing five distinct rule sets.
	•	Clean separation will unblock new features (stretch match, rule engine).
	•	We’ll split the code first, keep the public API working, and add deeper unit‑tests once the dust settles.

⸻

2 · Target module layout

utils/
├── rules/
│   ├── __init__.py
│   ├── eligibility.py          # age/service/hours + entry‑date calc
│   ├── auto_enrollment.py      # AE logic & small helper funcs
│   ├── auto_increase.py        # AI logic
│   ├── formula_parsers.py      # match & NEC parsing helpers
│   ├── contributions.py        # contributions + 415 limit
│   └── response.py             # plan‑change deferral bump
└── plan_rules.py               # THIN façade that orchestrates above

No tests are required in this refactor PR—focus on moving code and keeping current CLI/runner working.

⸻

3 · Step‑by‑step tasks

Phase 1 – Extract helper parser (½ day)
	1.	Create utils/rules/formula_parsers.py.
	2.	Move parse_match_formula() plus the tier parser block (inside contributions) into that file.
	3.	Update import in contributions section:
from utils.rules.formula_parsers import parse_match_formula.

Phase 2 – Eligibility split (½ day)
	1.	Copy determine_eligibility() into utils/rules/eligibility.py.
	2.	Replace print calls with logger.info/debug (import logging).
	3.	In plan_rules.py, from utils.rules.eligibility import apply as apply_eligibility.

Phase 3 – Auto‑Enrollment & Auto‑Increase (1 day)
	1.	Copy apply_auto_enrollment() → auto_enrollment.py.
	2.	Copy apply_auto_increase() → auto_increase.py.
	3.	Both modules accept (df, plan_rules, year_dates) to decouple from scenario blob.

Phase 4 – Contributions engine (1 day)
	1.	Slice contribution logic into utils/rules/contributions.py::apply.
	2.	Keep IRS‑limit parsing inside that module.
	3.	Ensure plan_rules.py still calls contributions last.

Phase 5 – Plan‑change response (½ day)
	1.	Move apply_plan_change_deferral_response() → utils/rules/response.py.

Phase 6 – Thin façade (½ day)

# utils/plan_rules.py – new look
from utils.rules import (
    eligibility, auto_enrollment, auto_increase,
    contributions, response, formula_parsers  # noqa: F401 – keep export
)
import logging
logger = logging.getLogger(__name__)

def process_year(df, scenario_cfg, baseline_cfg, year_dates):
    df = eligibility.apply(df, scenario_cfg['plan_rules'], year_dates.end)
    df = auto_enrollment.apply(df, scenario_cfg['plan_rules'], year_dates)
    df = auto_increase.apply(df, scenario_cfg['plan_rules'], year_dates.year)
    df = response.apply(df, scenario_cfg, baseline_cfg, year_dates.year, scenario_cfg['start_year'])
    df = contributions.apply(df, scenario_cfg, year_dates)
    return df

Old runners keep calling plan_rules.process_year()—no signature change.

⸻

4 · Deferred items (future sprint)

Backlog item	Rationale
Pytest coverage for each sub‑module	Add once code stabilises.
Type hints / mypy	Optional, improves IDE help.
Rules engine (YAML‑driven logic)	Easier after code is modular.



⸻

5 · Developer notes & gotchas
	•	Logger: import logging in each new module; avoid print.
	•	Circular imports: keep sub‑modules self‑contained; only façade imports them.
	•	Git history: use git mv when splitting files so history tracks.
	•	Smoke run: after each phase, run the existing CLI (run_projection.py) on a small census to ensure no regressions.

⸻

Ready‑made kickoff ticket (copy into Jira)

Title: Phase 1 – Extract formula parsers from plan_rules.py
Tasks:
	1.	Add utils/rules/formula_parsers.py with parse_match_formula() and tier parser.
	2.	Replace imports in contributions section.
	3.	Update any relative imports to absolute.
Acceptance: CLI runs successfully; no functional change.

Hand this plan to the engineer, let them tackle Phase 1–6, and circle back for test coverage when time allows.
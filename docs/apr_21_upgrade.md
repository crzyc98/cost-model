# Windsurf AI Implementation Plan (rev Apr‑25‑2025)

*A step‑by‑step script you can paste into Windsurf AI (VS Code) so the o4‑mini‑high agent can generate the Monte‑Carlo scenario engine inside **your existing `cost-model/` repo layout.***

---

## 0 · Objective
Extend the **cost‑model** project so an analyst can:
1. Drop a **baseline** plan‑rule YAML plus 2‑4 **alternate** designs.
2. Pick a **Monte‑Carlo T‑shirt package** (XS–XL) that defines growth / attrition / salary inflation draws.
3. Run a single CLI command that executes *N* MC paths for every design with common random numbers.
4. Receive CSV & plots (fan chart, delta table) in `reports/`.

---

## 1 · Updated Repository Skeleton (matches your tree)
```
cost-model/
├── configs/                # <- parameter files
│   ├── config.yaml         # simulation defaults
│   └── mc_packages.json    # XS–XL definitions
├── scenarios/              # NEW – analyst‑authored plan overrides
│   ├── baseline.yaml
│   ├── alt1.yaml
│   └── alt2.yaml
├── src/
│   └── cost_model/
│       ├── __init__.py
│       ├── agent.py
│       ├── model.py
│       ├── plan_rules.py
│       ├── mc.py           # NEW – Monte‑Carlo driver
│       ├── scenario_loader.py  # NEW – YAML extends logic
│       ├── sampler.py      # NEW – ParameterSampler helper
│       ├── reporting.py    # NEW – aggregation + charts
│       └── run_compare.py  # NEW – CLI entry‑point
├── tests/
│   └── … (add tests as new modules appear)
└── (other existing files …)
```

> **Why add `scenarios/`?**  Keeps analyst designs separate from `configs/`, which holds global defaults & MC packages.  Nothing else in your repo changes.

---

## 2 · Phased Windsurf AI Prompts
Each bullet is a *single prompt* you’ll paste into Windsurf AI chat.  o4‑mini‑high will respond with code changes.

### Phase 1 – Scenario Loader
1. **Prompt 1** → *Create `src/cost_model/scenario_loader.py` with `load(path)` that supports an `extends` key and returns a `dict` of merged configs.*
2. **Prompt 2** → *Add `tests/test_scenario_loader.py` to cover success and missing‑parent error.*

### Phase 2 – MC Parameter Sampler
1. *Build `src/cost_model/sampler.py` with `ParameterSampler` that reads `configs/mc_packages.json`.*
2. *Unit‑test edge ranges in `tests/test_sampler.py`.*

### Phase 3 – Monte‑Carlo Driver
1. *Implement `src/cost_model/mc.py` that loops `runs` times and executes `RetirementPlanModel` for every scenario using common RNG seeds.*
2. *Expose CLI in `src/cost_model/run_compare.py` (`argparse` args: `--scenario-dir`, `--mc-package`, `--runs`, `--output`).*

### Phase 4 – Reporting
1. *Create `src/cost_model/reporting.py` that writes `summary.csv` and, if `matplotlib` available, fan‑chart PNGs to `reports/`.*
2. *Add P‑50 / P‑90 delta calc vs. baseline.*

### Phase 5 – Docs & CI
1. *Update `docs/architecture.md` with the new MC layer.*
2. *Extend `.github/workflows/ci.yml` to run new tests.*

---

## 3 · YAML & JSON Templates

`scenarios/baseline.yaml` (minimal):
```yaml
match:
  type: stretch
  tiers:
    - rate: 0.50
      up_to: 0.04
vesting:
  type: graded
  schedule: [0, 20, 40, 60, 80, 100]
```

`configs/mc_packages.json` excerpt (keep existing key, add XS‑XL if missing):
```json
{
  "XS": {"g": 0.03, "h_ex": {"mean": 0.12, "sd": 0.01}},
  "S" : {"g": 0.02, "h_ex": {"mean": 0.13, "sd": 0.01}},
  "M" : {"g": 0.01, "h_ex": {"mean": 0.14, "sd": 0.01}},
  "L" : {"g": 0.00, "h_ex": {"mean": 0.15, "sd": 0.015}},
  "XL": {"g": -0.01,"h_ex": {"mean": 0.16, "sd": 0.02}}
}
```

---

## 4 · Analyst How‑To (updated)
```bash
# 1.  Put YAML designs in scenarios/
# 2.  Select a package key from configs/mc_packages.json (e.g. standard)
python -m cost_model.run_compare \
   --scenario-dir scenarios \
   --mc-package standard \
   --runs 1000 \
   --output reports/
```
Outputs land in `reports/summary.csv` and `reports/plots/`.

---

## 5 · First Windsurf AI Prompt (ready to copy)
> **“Create `src/cost_model/scenario_loader.py` with a `load_scenarios(dir_path)` function that supports an optional `extends` key, performs recursive merges, and returns a `dict[name, dict]`. Also generate `tests/test_scenario_loader.py` with fixtures for (a) simple load, (b) extends chain, (c) missing parent error.”**

Paste that in Windsurf and start the cycle.

---
*Document updated to match your existing repo structure. Continue prompting Windsurf AI through the phases above.*


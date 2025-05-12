# Phase 4 – Validation & CI Pipeline

Ensure long-term confidence with reproducible tests and continuous integration.

---
## 1  Unit-Test Coverage Targets
| Module | Coverage % |
|--------|-----------|
| state & snapshot* | 90 |
| dynamics | 80 |
| plan_rules | 80 |

---
## 2  Scenario Regression Suite
Create `tests/regression/` containing small YAML scenarios (Baseline, AE_AI) and expected KPI CSVs.
Use `pytest --scenario <file>` paramized fixture to run projection and compare summaries.

---
## 3  GitHub Actions Workflow `.github/workflows/ci.yml`
1.  Setup Python 3.11, poetry install.
2.  Run `pytest -q` with coverage.
3.  Build docs via MkDocs.
4.  On `main` merge, deploy docs to GitHub Pages.

---
## 4  Data Validation Rules
- Use `pandera` schemas for snapshots & summaries.
- Validate no negative headcount/comp, unique employee_id per snapshot, etc.

---
## 5  Definition of Done
- CI passes on PR with full test & lint.
- Docs site auto-deploys.
- Regression suite flags breaking changes.

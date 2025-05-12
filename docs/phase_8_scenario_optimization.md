# Phase 8 – Scenario Exploration & Optimization

Unleash *what-if* power by letting analysts run hundreds of scenario variations and surface the most impactful levers.

---
## 1  Batch Scenario Runner
- CLI `projection-batch --matrix configs/matrix.yaml` where YAML lists parameter grids.
- Generates a **Cartesian product** of values (e.g. termination_rate × match_formula).
- Uses `concurrent.futures.ProcessPoolExecutor` to fan out runs; stores each result under `output/{scenario_id}`.

Example `matrix.yaml`:
```yaml
base: baseline.yaml
sweeps:
  turnover_rate: [0.12, 0.15, 0.18]
  match_formula: ["50% up to 6%", "100% up to 4%"]
```

---
## 2  Sensitivity Analysis Toolkit
1. Import results into `notebooks/sensitivity.ipynb`.
2. Compute **tornado charts** for KPIs (employer_cost, participation).
3. Use **Morris method** or **Sobol indices** via `SALib` to quantify factor importance.

---
## 3  Optimization Loop
Goal: minimize employer cost subject to ≥ 75% participation.

Algorithm options:
| Approach | Library | Notes |
|----------|---------|-------|
| Bayesian Opt | `scikit-optimize` | Handles expensive functions |
| Genetic Alg | `DEAP` | Good for discrete choices |
| Grid Search | NA | Baseline |

Expose API `/api/optimizations/` that accepts objectives & constraints, launches optimization job, streams Pareto frontier.

---
## 4  Visualization
Add UI tab **“Optimizer”**:
- Select baseline scenario.
- Define objective (drop-down) & constraints.
- Start run → real-time scatter plot of candidate solutions.

---
## 5  Definition of Done
- Can sweep 100 scenarios in < 2min on 8-core.
- Sensitivity notebook auto-refreshes after batch.
- Optimizer finds cost-saving config within constraint.

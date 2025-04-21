Below is a workflow upgrade that lets an analyst load any mix of plan designs (baseline + 2‑4 alternates), compare them head‑to‑head, and wrap the whole thing inside “T‑shirt” Monte‑Carlo packages that push each design through mild, moderate, and severe stress.

⸻

1 . Scenario‑Driven Workflow

1.1 Folder & file convention

scenarios/
├── baseline.yaml
├── alt1_stretch_match.yaml
├── alt2_shorter_vesting.yaml
└── alt3_auto_enrol.yaml

Each YAML is a partial plan‑rule file—overrides to the common baseline.

# alt2_shorter_vesting.yaml
extends: baseline            # inherit everything else
vesting:
  type: graded
  schedule: [20, 40, 60, 80, 100]   # 5‑yr graded instead of 6‑yr

1.2 CLI / notebook entry point

python run_compare.py \
  --scenario-dir scenarios \
  --mc-package standard \
  --runs 1000 \
  --output results/

	•	--mc-package standard = “T‑shirt” distributions below
	•	--runs 1000 = # of MC paths per scenario

1.3 “Common random numbers”

Seed the RNG once per run and reuse across scenarios.
That way the cost deltas are design‑driven, not noise‑driven.

⸻

2 . T‑Shirt Monte‑Carlo Packages

Package	Growth (g)	Attrition (h_ex)	Market return*	Salary infl.
XS	3 %	–1 σ	Mean	2 %
S	2 %	Mean	Mean-σ	3 %
M	1 %	+1 σ	Mean-1.5 σ	3.5 %
L	0 %	+2 σ	Mean-2 σ	4 %
XL	–1 %	+3 σ	2008 replay	5 %

*Only if you model asset growth.
	•	“XS” ≈ optimistic, “S” ≈ base, “M” ≈ mildly adverse, etc.
	•	The package loader expands each YAML into 5 sub‑scenarios (alt1_S, alt1_L, …).

⸻

3 . Monte‑Carlo Engine Skeleton

def run_scenario(cfg, mc_params, n_runs, rng_seed):
    rng = np.random.RandomState(rng_seed)
    kpis = []
    for i in range(n_runs):
        # draw param set from package
        draw = mc_params.sample(rng)
        model_cfg = {**cfg, **draw}           # merge YAML + MC draw
        model     = RetirementPlanModel(model_cfg, census_df, rng)
        model.run()
        kpis.append(model.kpi)
    return pd.DataFrame(kpis)

def main(dir_path, package, runs):
    scenarios = load_yaml_dir(dir_path)
    mc_params = load_package(package)        # XS–XL distributions
    seed      = 42                           # common across designs
    outputs   = {}
    for name, cfg in scenarios.items():
        outputs[name] = run_scenario(cfg, mc_params, runs, seed)
    save(outputs, "results/")



⸻

4 . Result Aggregation

4.1 KPI table

Design	Package	P‑50 Cost	Δ vs. Baseline	P‑90 Cost	Δ90 vs. BL
alt1	S	$47.2 M	+1.4 M	53.5 M	+1.7 M
alt2	S	45.9 M	‑0.0 M	52.0 M	+0.2 M
alt3	S	46.8 M	+0.9 M	52.9 M	+1.1 M

4.2 Fan charts & tornado
	•	One fan chart per design (headcount & employer cost over time).
	•	One tornado: sensitivity of cost to each random driver across all designs.

4.3 “Traffic‑light” delta

Map Δ‑cost percentile bands to colours or T‑shirt size:

Δ ≤ 0        => “Savings”  (green, size XS/S)
0 < Δ ≤ 2%   => “Neutral”  (amber, size M)
Δ > 2%       => “Cost ↑”   (red, size L/XL)



⸻

5 . Phase Roll‑out

Phase	Deliverable	Owner	Notes
1	Scenario YAML “extends” + CLI	Dev	1–2 sprints
2	MC package loader + RNG seeding	Dev	reuse engine
3	Aggregator & delta logic	Data Eng	pandas groupby
4	Fan chart & tornado visuals	Analyst / BI	matplotlib or Plotly
5	QA & regression suite	QA	baseline YAMLs fixed
6	Analyst training deck	Product	30‑min session



⸻

6 . Next‑Step Checklist
	1.	Lock YAML schema (extends, match, vesting, etc.).
	2.	Implement common‑seed MC driver with package JSON.
	3.	Build delta‑calculator that:
	•	Aligns runs by seed;
	•	Computes P‑50 and P‑90 deltas vs. baseline.
	4.	Prototype output on a small census; measure runtime.
	5.	Refine T‑shirt thresholds with Finance.
	6.	Publish analyst how‑to (“drop YAMLs, run CLI, read Excel”).

⸻

Need code samples, a Jupyter template, or help defining the MC packages in JSON?

Just tell me which piece you’d like fleshed out, and we’ll build it.
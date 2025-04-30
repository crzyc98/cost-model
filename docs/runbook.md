Below is a “soup-to-nuts” run-sheet you (and any teammate) can follow every time you pull a new branch, change a rule, or tweak an assumption.  It is opinionated but turnkey: copy it to docs/runbook.md, tweak paths once, and you’ll never be stuck wondering “what do I do next?”

⸻

0  Bootstrap (one-time per laptop / CI runner)

step	command	goal
0-a	python -m venv .venv && source .venv/bin/activate	isolated Python
0-b	pip install -r requirements.txt -r requirements-dev.txt	all deps
0-c	pre-commit install	auto-format & lint before every commit
0-d	export PYTHONPATH=$PWD (add to shell profile)	repo modules resolvable everywhere



⸻

1  Fast-fail sanity check (<< 30 s)

pytest -q tests/quick
python scripts/smoke_run.py         # 1-yr, tiny census, DEBUG logging

What it tells you: code still imports, key rules run, output folders created.

⸻

2  Full Phase 1 HR-snapshot generation
	1.	Edit/Tee configs/dev_local.yaml
Use the real census but only 2–3 projection years for speed.
	2.	Run

python scripts/run_hr_snapshots.py \
  --config configs/dev_local.yaml \
  --census data/census_preprocessed.parquet \
  --output output/hr_snapshots \
  --seed 42


	3.	Quick smoke-metrics

python utils/qa/check_headcount.py output/hr_snapshots \
       --expected-growth 0.03 --tolerance 0.5



⸻

3  Full Phase 2 plan-rule pass

python -m scripts.run_plan_rules \
       --config configs/dev_local.yaml \
       --snapshots-dir output/hr_snapshots \
       --output-dir    output/plan_outputs \
       | tee logs/plan_rules_run.log

Pro-tip: Until you’re happy with the log noise, run with
LOG_LEVEL=INFO python -m ….

⸻

4  Automated “red-flag” scoreboard  (1 min)

Create a single notebook notebooks/00_scoreboard.ipynb or a CLI utility:

python utils/qa/scoreboard.py output/plan_outputs/*_metrics.csv

Key checks (they print ✅ / ❌):

metric	test
headcount	matches growth assumptions ±0.25 %
eligible %	stays within 2 pp of prior census
participation %	same logic
avg_deferral %	non-zero when AE on; monotonic if AI on
total_plan_cost	not NaN / inf; monotonic w/ match increases
comp growth	row-level median ≈ configured comp Δ



⸻

5  Deep-dive interactive workbook  (≈ 5 min once)

Spin up JupyterLab:

jupyter lab --notebook-dir notebooks

Two canonical notebooks:
	1.	01_population_walkthrough.ipynb – loads the yearly Parquet snapshots; scatter of age vs. tenure; churn waterfall.
	2.	02_financials_walkthrough.ipynb – loads plan-rule output and produces:
	•	side-by-side scenario tables
	•	heat-maps of deferral vs. comp quantile
	•	cohort traces (new-hire cohort participation over time)

⸻

6  Log summariser (look for behavioural bugs fast)

Add one helper:

# utils/qa/log_summary.py
import re, pathlib, collections, pandas as pd

PAT = re.compile(
    r'INFO (Applying Auto Enrollment|AE Applied|Eligibility determined): (.*)'
)

def summarise(log_path):
    rows = []
    for line in pathlib.Path(log_path).read_text().splitlines():
        m = PAT.search(line)
        if m:
            rows.append(m.groups())
    return pd.DataFrame(rows, columns=['event', 'payload'])

Usage:

python - <<'PY'
from utils.qa.log_summary import summarise
print(summarise("logs/plan_rules_run.log").head(20))
PY



⸻

7  Debug / iterate loop
	•	☑ Something looks off (e.g. headcount growth too low)
→ open snapshot parquet → query hires & terms → adjust termination-rate formula or hire calc → re-run Phase 1 only.
	•	☑ Participation crazy
→ grep AE Applied: lines → mismatch? → tweak apply_plan_rules.auto_enroll().

Each cycle is Phase 1 + Phase 2 only, not a full reinstall, so iteration is < 2 minutes.

⸻

8  CI / PR gate

Add .github/workflows/regression.yml:

name: regression
on: [pull_request]
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest -q tests/quick
      - run: python scripts/run_hr_snapshots.py \
              --config configs/ci.yaml \
              --census data/sample_census.parquet \
              --output output/hr_snapshots --seed 7
      - run: python -m scripts.run_plan_rules \
              --config configs/ci.yaml \
              --snapshots-dir output/hr_snapshots \
              --output-dir output/plan_outputs
      - run: python utils/qa/scoreboard.py output/plan_outputs/*_metrics.csv

Fail the workflow if any red-flag check returns ❌.

⸻

9  Documentation for the next human

Put the above in docs/runbook.md and link it from README.md#quick-start.
Add a short Loom or GIF showing Notebook 01 & 02 in action.

⸻

10  Where you come in (“coach” feedback loop)
	1.	You run the pipeline (make run_dev if you wrap the commands).
	2.	You glance at scoreboard.py output – if any ❌, click the generated HTML diff (store it in output/qa/).
	3.	You open the notebooks, filter by employee_role == "Staff" & year == 2028, see participation 2 pp low.
	4.	File a GitHub issue, attach the slice as CSV + screenshot, label bug:AE.
	5.	Dev fixes logic, pushes; CI turns ✅.

⸻

Summary checklist

done?	item
☐ constants imported & used in run_plan_rules.py	
☐ scoreboard.py returns all ✅ on baseline	
☐ notebooks render without manual path edits	
☐ CI workflow green on pull-request	

Follow these steps and you’ll have a repeatable, sub-5-minute loop to prove (or disprove) that the model is behaving—as well as a paper-trail you can hand to any engineer for fast fixes.
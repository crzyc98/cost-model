# Retirement Plan Cost Model

An agent-based simulation model of a retirement plan built using the Mesa framework. This project estimates participation, deferral behavior, employer contributions, and overall plan cost over a multi-year projection.

## Features

- **Agent-Based Modeling** with Mesa to simulate individual employee behaviors
- Configurable plan rules (eligibility, auto-enrollment, auto-increase) via `config.yaml`
- Handles population dynamics (new hires and terminations)
- Precise financial calculations using Python's `Decimal`
- Output of model- and agent-level results to CSV/Excel

## Requirements

- Python 3.8 or higher
- See [requirements.txt](requirements.txt) for full dependency list

## Installation

```bash
# Clone the repository
git clone <repo_url> cost-model
cd cost-model

# (Optional) create a virtual environment
env=$(python3 -m venv venv) && source $env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to adjust:

- Simulation start year and projection length
- Plan eligibility criteria
- Auto-enrollment (AE) and auto-increase (AI) rates
- Contribution formulas and IRS limits
- Demographic and turnover rates
- Veteran attrition handling: `use_expected_attrition` (boolean) to toggle expected vs. realized veteran terminations in hires
- Monthly attrition transitions: `monthly_transition` (boolean) to enable monthly hazard-based terminations using fitted hazard model
- Hazard model parameters: `hazard_model_params.file` pointing to YAML file with Cox coefficients and Kaplan–Meier median time

## Data

Place your initial census data in CSV format as `census_data.csv`. The expected columns include:

- `employee_id`, `birth_date`, `start_date`, `salary`, etc.

Example data file is provided in the root directory.

## Usage

Run the retirement plan simulation:

```bash
python scripts/run_retirement_plan_abm.py \
  --config config.yaml \
  --census census_data.csv \
  --output results/
```

- `--config`: path to your YAML configuration file
- `--census`: path to your initial census CSV
- `--output`: directory to write model and agent result files

**Monthly attrition** can be activated by setting `monthly_transition: true` and specifying `hazard_model_params.file` in your `config.yaml`. The simulation CLI remains the same.

After completion, results will be saved as:

- `results/model_results.csv`
- `results/agent_results.csv`

## Reporting Dynamics
After running the ABM simulation, generate visual reports with:

```bash
python3 scripts/report_dynamics.py \
  --model_csv output/<prefix>_model_results.csv \
  --agent_csv output/<prefix>_agent_results.csv \
  --output_dir output
```

This will produce:

- `output/net_growth_decomposition.png`
- `output/cohort_counts.png`
- `output/hire_termination_dynamics.png`

## Project Structure

```
cost-model/
├── agents/
│   └── employee_agent.py       # EmployeeAgent class definition
├── model/
│   └── retirement_model.py     # RetirementPlanModel implementation
├── scripts/
│   └── run_retirement_plan_abm.py  # Main simulation runner
├── config.yaml                 # Simulation parameters and plan rules
├── census_data.csv             # Sample initial employee data
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Contributing

Pull requests are welcome. Please open an issue to discuss major changes.

## License

This project is released under the MIT License.

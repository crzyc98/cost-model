cost_model/
├── config/
│   ├── loaders.py
│   └── models.py
├── data/
│   └── processing.py
├── dynamics/
│   ├── compensation.py
│   ├── engine.py
│   ├── hiring.py
│   ├── sampling/
│   │   ├── salary.py
│   │   └── terminations.py
│   └── termination.py
├── engines/
│   ├── __init__.py
│   ├── cola.py
│   ├── comp.py
│   ├── compensation.py
│   ├── hire.py
│   ├── markov_promotion.py
│   ├── promotion.py
│   ├── run_one_year/
│   │   ├── __init__.py
│   │   ├── comp_term.py
│   │   ├── finalize.py
│   │   ├── hires.py
│   │   ├── orchestrator.py
│   │   ├── plan_rules.py
│   │   ├── utils.py
│   │   └── validation.py
│   ├── run_one_year_engine_old.py
│   └── term.py
├── ml/
│   └── ml_utils.py
├── plan_rules/
│   ├── auto_increase.py
│   ├── contributions.py
│   ├── eligibility.py
│   ├── enrollment.py
│   ├── response.py
│   └── validators.py
├── projections/
│   ├── cli.py
│   ├── config.py
│   ├── event_log.py
│   ├── hazard.py
│   ├── reporting.py
│   ├── runner/
│   │   └── (runner module files)
│   ├── runner.py
│   ├── snapshot.py
│   └── summaries/
│       ├── core.py
│       ├── employment.py
│       └── reporting.py
├── reporting/
│   └── (reporting module files)
├── rules/
│   └── (various rule files)
├── simulation.py
├── state/
│   ├── builder.py
│   ├── event_log.py
│   ├── job_levels/
│   │   ├── assign.py
│   │   ├── defaults.py
│   │   ├── engine.py
│   │   ├── init.py
│   │   ├── intervals.py
│   │   ├── loader.py
│   │   ├── models.py
│   │   ├── sampling.py
│   │   ├── state.py
│   │   ├── transitions.py
│   │   └── utils.py
│   ├── schema.py
│   ├── snapshot.py
│   ├── snapshot/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── details.py
│   │   ├── helpers.py
│   │   ├── snapshot_build.py
│   │   ├── snapshot_update.py
│   │   └── tenure.py
│   ├── snapshot_build.py
│   ├── snapshot_update.py
│   ├── snapshot_utils.py
│   └── tenure.py
└── utils/
    ├── census_generation_helpers.py
    ├── columns.py
    ├── compensation/
    │   ├── __init__.py
    │   └── bump.py
    ├── constants.py
    ├── data_processing.py
    ├── date_utils.py
    ├── decimal_helpers.py
    ├── id_generation.py
    ├── labels.py
    ├── ml_logic.py
    ├── plan_rules.py
    ├── plan_rules_engine.py
    ├── preprocess_census.py
    ├── projection_utils.py
    ├── rules/
    │   └── (rule utilities)
    ├── simulation_utils.py
    ├── status_enums.py
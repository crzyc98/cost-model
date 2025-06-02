Product Requirements Document (PRD): Workforce Model Age Sensitivity & Auto-Calibration Enhancement
1. Introduction & Purpose
To enhance the workforce simulation model with comprehensive age-based dynamics for attrition and promotion, and to develop and operationalize a robust auto-tuning system. This system will calibrate the model to accurately preserve key metrics (headcount growth, compensation growth, and the shapes of age and tenure distributions) year-over-year, ensuring more realistic simulations and enabling efficient, data-driven parameterization for workforce planning and analysis.

2. Goals

Implement realistic age-sensitive logic for both employee terminations and promotions.
Achieve an automated, repeatable, and effective model calibration process using the auto-tuning script.
Ensure simulation outputs (especially age and tenure distributions) are accurately captured and scored against defined baselines.
Improve the overall accuracy, credibility, and utility of the workforce projection model.
Enable model users to confidently use a well-calibrated baseline for analyses and scenario modeling.
3. Target Users

Workforce Planners/Analysts: Utilize the calibrated model for accurate forecasting and strategic decision-making.
Model Maintainers/Developers: Responsible for implementing features, calibrating the model, ensuring its accuracy, and running tuning campaigns.
4. Scope (Focusing on "What Remains")

In Scope for Completion:
Implementation and verification of age sensitivity in the promotion engine.
Full implementation and bug-fixing of the auto-tuning script (tune_configs.py), including:
Accurate extraction of age and tenure distributions from simulation snapshot data.
Correction of the KL Divergence formula.
Comprehensive scoring function incorporating all target metrics (age/tenure distributions, headcount growth, compensation growth) with appropriate weighting.
Loading of actual baseline distributions and target values.
Review and expansion of the SEARCH_SPACE to include detailed hazard configuration parameters (e.g., specific age/tenure multipliers for termination and promotion).
Ensuring the tuning script can effectively override all parameters in the SEARCH_SPACE.
Targeted verification of the directional impact of age sensitivity features (termination already partially verified, promotion once implemented).
End-to-end testing of the auto-tuning script with a small number of iterations.
Execution of at least one initial full-scale auto-tuning campaign to produce an optimized baseline configuration.
Out of Scope (for this immediate finalization effort):
Major refactoring of simulation engines beyond what's necessary for parameter overriding by the tuner.
Introduction of new tuning algorithms beyond the current random search (unless a quick integration, like Optuna, was already planned and partially implemented).
Development of a graphical user interface (GUI) for the tuning script.
5. Requirements

FR1: Promotion Engine Age Sensitivity:
FR1.1: The promotion engine must be updated to incorporate age-based multipliers that adjust base promotion rates.
FR1.2: These promotion age multipliers must be configurable, ideally through the main scenario YAML file, to be accessible by the auto-tuner.
FR2: Auto-Tuner - Complete Data Parsing (parse_summary):
FR2.1: The parse_summary function in tune_configs.py must accurately read detailed snapshot data (e.g., consolidated_snapshots.parquet) from each simulation run.
FR2.2: It must derive and populate normalized age distributions (age_hist) from this snapshot data.
FR2.3: It must derive and populate normalized tenure distributions (tenure_hist) from this snapshot data.
FR3: Auto-Tuner - Correct Scoring Logic (kl_divergence, score):
FR3.1: The kl_divergence function must implement the correct mathematical formula for Kullback-Leibler divergence.
FR3.2: The score function must calculate a composite score based on the difference/divergence between simulation outputs and baseline targets for:
Age distribution (age_hist).
Tenure distribution (tenure_hist).
Headcount growth (hc_growth).
Average compensation growth (pay_growth).
FR3.3: The score function should allow for configurable weights for each component of the total score.
FR4: Auto-Tuner - Baseline Data (load_baseline_distributions):
FR4.1: The load_baseline_distributions function must load actual, externally defined baseline age and tenure distributions.
FR4.2: The system must define and load actual target values for headcount growth and compensation growth.
FR5: Auto-Tuner - Comprehensive & Effective SEARCH_SPACE:
FR5.1: The SEARCH_SPACE in tune_configs.py must be reviewed and expanded to include fine-grained, tunable parameters crucial for calibration, especially specific age and tenure multipliers for both termination and promotion from the hazard configuration.
FR5.2: The simulation engines (term.py, promotion engine) must be configurable such that parameters defined in hazard_defaults.yaml (or equivalent) can be effectively overridden by the main scenario YAML modified by the tuner.
NFR1: Verification: The directional impact of age sensitivity (for both termination and promotion) must be clearly verifiable through targeted test runs.
NFR2: Usability (Tuner): The auto-tuning script must be runnable via CLI, provide informative logs, and clearly indicate the best configuration found.
NFR3: Reliability (Tuner): The auto-tuning script should robustly handle individual simulation run failures and continue with the tuning process where possible.
6. Success Metrics

Age sensitivity for promotions is implemented and verified to directionally impact promotion outcomes as expected.
The tune_configs.py script successfully completes tuning runs, accurately parsing all required metrics (age/tenure distributions, hc/comp growth) and applying a meaningful, multi-component score.
An initial auto-tuning campaign (e.g., 100+ iterations) identifies a best_config.yaml that, when run, produces simulation results measurably closer to all defined baseline targets (distributions and growth rates) than a default/starting configuration.
The system is documented sufficiently for a model maintainer to run new tuning campaigns.
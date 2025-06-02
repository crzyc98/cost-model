Epic 2: Fully Operationalize the Auto-Tuning Script (tune_configs.py)

User Story 2.1 (Implement Age/Tenure Distribution Parsing):
As a Model Developer, I want to update the parse_summary() function in tune_configs.py to accurately extract normalized age distributions (age_hist) and tenure distributions (tenure_hist) from the consolidated_snapshots.parquet (or equivalent detailed output) of each simulation run.
User Story 2.2 (Correct KL Divergence Formula):
As a Model Developer, I want to correct the mathematical formula in the kl_divergence() function in tune_configs.py so that the dissimilarity between probability distributions is calculated accurately.
User Story 2.3 (Enhance Scoring Function):
As a Model Developer, I want to enhance the score() function in tune_configs.py to calculate a composite score that includes errors for headcount growth and compensation growth (against defined targets), in addition to the (corrected) KL divergence for age and tenure distributions, using a clear and potentially configurable weighting system.
User Story 2.4 (Implement Real Baseline Loading):
As a Model Developer, I want to implement the load_baseline_distributions() function in tune_configs.py to load actual target age/tenure distributions and target values for headcount/compensation growth from a reliable external source (e.g., a configuration file or data file).
User Story 2.5 (Expand and Refine Tuner Search Space):
As a Model Developer, I want to review and expand the SEARCH_SPACE in tune_configs.py to include all critical parameters for calibration, especially specific age/tenure multipliers for both termination and promotion as defined in hazard configurations.
User Story 2.6 (Ensure Comprehensive Parameter Overriding for Tuner):
As a Model Developer, I want to ensure the simulation engines (term.py, promotion engine) source all tunable hazard parameters (like detailed age/tenure multipliers) from the main scenario configuration object that tune_configs.py modifies, rather than loading them from hardcoded file paths, so the tuner can effectively control these parameters.
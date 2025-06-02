Epic 3: Validate System and Conduct Initial Tuning Campaign

User Story 3.1 (End-to-End Auto-Tuner Test):
As a Model Developer, I want to run the tune_configs.py script for a small number of iterations (e.g., 3-5) after all parsing and scoring enhancements are complete, so that I can verify the entire end-to-end workflow (config generation, simulation execution, full metrics parsing, correct scoring, and results saving) is functioning correctly.
User Story 3.2 (Execute Initial Full Auto-Tuning Campaign):
As a Model Maintainer, I want to execute a comprehensive auto-tuning campaign (e.g., 100-1000 iterations) using the validated tune_configs.py script so that I can generate an empirically optimized baseline configuration for the workforce model.
User Story 3.3 (Analyze and Select Best Tuned Configuration):
As a Model Maintainer, I want to analyze the tuning_results.json and best_config.yaml from the auto-tuning campaign so that I can understand the results, select the most suitable configuration, and establish it as the new production baseline.
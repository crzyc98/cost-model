Objective
Enhance the retirement plan simulation model to incorporate more realistic employee behavioral responses regarding their deferral rate choices when specific plan design features are introduced or changed. This goes beyond the mechanical application of Auto-Enrollment (AE) and Auto-Increase (AI).
Core Requirements
Model Auto-Enrollment Outcomes: Simulate the different paths employees might take when auto-enrolled, rather than assuming all non-opt-outs stay at the default rate.
Model Response to Plan Changes: Simulate how existing participants might adjust their deferral rates in response to significant plan design changes introduced in a scenario (e.g., a change in the employer match formula).
Detailed Design Specifications
1. Enhanced Auto-Enrollment Logic
Concept: When AE is active (scenario_config['plan_rules']['auto_enrollment']['enabled'] == True), newly eligible employees who don't actively choose a rate or opt-out should be assigned an initial deferral rate based on a probabilistic distribution, not just the single default rate.
Configuration Parameter Updates (Add to scenario_config['plan_rules']['auto_enrollment']):
ae_outcome_distribution (dict): A dictionary defining the probability of different outcomes for an auto-enrolled employee. Example:
"ae_outcome_distribution": {
    "opt_out": 0.08, // Probability of opting out entirely (deferral = 0)
    "stay_default": 0.80, // Probability of staying at ae_default_rate
    "increase_to_match_cap": 0.12 // Probability of increasing to get full match
    // Ensure probabilities sum to 1.0
}

(Note: This replaces the previous simple ae_opt_out_rate).
ae_default_rate: (Existing) The rate applied for the stay_default outcome.
Module Update (apply_auto_enrollment function):
Input: Needs access to the scenario's match formula parameters to determine the rate needed for the maximum match (employer_match_cap_deferral_perc).
Logic:
Identify the target group: Newly eligible employees with deferral_rate == 0 (or no active election).
For each employee in the target group, randomly assign an outcome based on the probabilities defined in ae_outcome_distribution.
Apply the outcome:
opt_out: Set deferral_rate = 0, potentially set ae_opted_out = True.
stay_default: Set deferral_rate = config['ae_default_rate']. Set is_participating = True. Mark as enrolled_via_ae.
increase_to_match_cap: Calculate the deferral rate needed to maximize the employer match (usually config['employer_match_cap_deferral_perc'] * 100). Set the employee's deferral_rate to this calculated rate (or potentially the ae_default_rate if it's already higher). Set is_participating = True. Mark as enrolled_via_ae.
2. Plan Change Response Logic (Focus: Match Formula Change)
Concept: Model the behavior of existing participants potentially increasing their deferral rate if the employer match formula becomes more generous in a proposed scenario compared to the baseline.
Configuration Parameter Updates (Add to scenario_config['plan_rules'] or a new behavioral_response section):
match_change_response (dict): Parameters governing the response. Example:
"match_change_response": {
    "enabled": true, // Activate this behavioral model
    "increase_probability": 0.30, // Probability an eligible participant below the new optimal rate increases
    "increase_target": "optimal" // Target rate ('optimal' or specific percentage)
}


New Module (apply_plan_change_deferral_response function):
Purpose: To adjust deferral rates for existing participants based on significant plan rule changes between the current scenario and a baseline scenario (if provided).
Inputs: df, current_scenario_config, baseline_scenario_config, year.
Trigger Condition: This logic should ideally run only once per simulation run for a given scenario, perhaps in the first projection year (year == config['start_year']), or by comparing the relevant rule (e.g., match formula) between current_scenario_config and baseline_scenario_config.
Logic (Match Change Example):
Compare Match Formulas: Extract match parameters (rate, cap percentage) from current_scenario_config and baseline_scenario_config. Determine if the current scenario's match is "more generous" (e.g., higher cap percentage or higher match rate at the same cap). A utility function will be needed to parse the formula string (e.g., "50% up to 6%") into numeric rate and cap values.
Identify Target Group: If the match is more generous in the current scenario:
Calculate the optimal_deferral_rate needed to maximize the match under the current scenario's rules (usually employer_match_cap_deferral_perc * 100).
Identify existing participants (is_participating == True) whose current deferral_rate is greater than 0 but less than this optimal_deferral_rate.
Simulate Response: For each employee in the target group, apply the increase_probability.
Apply Change: If an employee is selected to increase:
If increase_target is 'optimal', set their deferral_rate to the calculated optimal_deferral_rate.
(Future enhancement: allow targeting a specific percentage increase or absolute rate).
Integration into Simulation Flow:
The new apply_plan_change_deferral_response function should be called within the yearly simulation loop (e.g., in the modified project_census function).
Timing: Place the call after apply_auto_increase but before calculate_contributions. Ensure its trigger condition (e.g., first year only, or detected change vs. baseline) is implemented correctly to prevent it from running unnecessarily every year.
Technical Considerations
Match Formula Parsing: Implement a robust utility function to parse the employer_match_formula string (e.g., "X% up to Y%") into numeric rate and cap percentage values. This is needed for both contribution calculations and the behavioral response logic.
Probability Sums: Add validation to ensure probabilities in ae_outcome_distribution sum to 1.0.
Baseline Comparison: The apply_plan_change_deferral_response function requires access to the corresponding baseline_scenario_config to detect changes. The main simulation runner needs to manage passing this appropriately.
Configuration Clarity: Clearly document all new configuration parameters.
Extensibility: Design the apply_plan_change_deferral_response function so it could potentially handle responses to other plan changes (e.g., NEC changes) in the future, although the initial focus is the match.

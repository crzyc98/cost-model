1. Objective
To create a flexible simulation model using an Agent-Based Modeling (ABM) approach in Python to analyze the cost-benefit impact of various 401(k) plan design changes over a 5-year projection period. The model will simulate individual employee decisions and track aggregate outcomes like participation, savings rates, and employer costs.
2. Core Framework: Agent-Based Modeling (e.g., Mesa)
The simulation will be built using an ABM framework like Mesa. This involves defining:
Agents: Representing individual employees.
Model: Representing the overall environment, including the company, the retirement plan rules for a given scenario, and managing the simulation time and agent population.
3. Model Components
3.1. EmployeeAgent Class
This class represents a single employee.
Attributes (State):
unique_id: Unique identifier (e.g., ssn).
model: Reference to the parent RetirementPlanModel instance.
birth_date, hire_date, termination_date: Core demographic dates.
status: ('Active', 'Terminated').
role: (e.g., 'Staff', 'Manager', 'Executive').
gross_compensation: Current annual salary.
is_eligible: Boolean flag for plan eligibility.
is_participating: Boolean flag (deferral > 0).
deferral_rate: Current pre-tax deferral percentage (e.g., 0.06 for 6%).
enrollment_method: ('Manual', 'AE', 'None').
ae_opted_out: Boolean flag.
ai_opted_out: Boolean flag.
behavioral_profile: (Optional) Category influencing decisions (e.g., 'Proactive Saver', 'Inert', 'Cautious').
contributions_current_year: Dictionary to store calculated contributions for the current step (e.g., {'employee': $, 'nec': $, 'match': $}).
Methods:
__init__(...): Initializes the agent with attributes based on input data (e.g., from a census row).
step(): Contains the agent's logic executed each simulation step (typically representing one year). This is the core decision-making engine:
Check Status: If not 'Active', return.
Update Compensation: Apply annual increase based on model.scenario_config.
Check Eligibility: Call eligibility logic based on current age/tenure and model.scenario_config. Update is_eligible.
Make Deferral Decision: This is the key behavioral part:
If Newly Eligible:
If AE enabled in model.scenario_config: Apply AE outcome logic (use ae_outcome_distribution probabilities, potentially influenced by behavioral_profile). Update deferral_rate, is_participating, ae_opted_out, enrollment_method.
If AE not enabled: Model voluntary enrollment probability (potentially based on behavioral_profile, age, income).
If Already Eligible & Participating:
If AI enabled: Apply AI logic (check cap, apply increase based on ai_increase_rate, simulate opt-out based on ai_opt_out_rate / behavioral_profile). Update deferral_rate.
Apply Plan Change Response: Check if significant plan rules (e.g., match) changed compared to a baseline (requires model to track this). If so, apply logic based on match_change_response config and behavioral_profile to potentially adjust deferral_rate.
If Eligible but Not Participating (e.g., opted out previously): Model a small probability of opting back in voluntarily each year.
Calculate Contributions: Determine the dollar contributions for the year based on the final deferral_rate, prorated compensation (calculated here or by the model), IRS limits (from model.scenario_config), and employer contribution formulas (from model.scenario_config). Store results in contributions_current_year.
(Helper methods may be added for clarity, e.g., _calculate_age, _calculate_tenure, _check_eligibility).
3.2. RetirementPlanModel Class
This class manages the overall simulation environment and agents.
Attributes (State):
scenario_config: Dictionary holding all parameters for the current simulation run.
current_year: The simulation year being processed.
schedule: Mesa scheduler (e.g., mesa.time.RandomActivation) to control agent activation order.
population: List or dictionary holding all EmployeeAgent instances.
datacollector: Mesa DataCollector instance configured to record agent-level and model-level variables.
Methods:
__init__(initial_census_df, scenario_config):
Stores scenario_config.
Initializes current_year.
Creates EmployeeAgent instances from initial_census_df rows and adds them to the schedule and population.
Sets up the DataCollector to track desired outputs (e.g., agent deferral_rate, is_participating, contributions_current_year; model-level aggregates).
step(): Executes one full simulation step (one year):
Increment current_year.
Handle Terminations:
Calculate termination probability/score for each active agent (using ML model loaded based on config, or rule-based logic).
Determine which agents terminate based on scores and the target termination_rate from config.
Update the status and termination_date for terminated agents.
Remove terminated agents from the schedule so their step() method isn't called in subsequent years.
Handle New Hires:
Calculate the number of new hires needed based on hire_rate and terminations.
Generate attributes for new hires (using role-based logic from config/generate_census.py concepts).
Create new EmployeeAgent instances for hires.
Add new agents to the population and the schedule.
Execute Agent Steps: Call self.schedule.step(). This triggers the step() method for all active agents currently in the schedule.
Collect Data: Call self.datacollector.collect(self) to record data for this step.
(Optional) Calculate Model-Level Aggregates: Compute summary statistics (total costs, participation rates, etc.) directly here or rely on post-simulation analysis of the DataCollector output.
(Helper methods for termination scoring, new hire generation can be included).
4. Simulation Execution and Scenarios
A main script (similar to run_projection.py) will be responsible for:
Loading the initial census data.
Defining the list of scenario configuration dictionaries (Baseline, Proposed_1, Proposed_2, etc.).
For each scenario config:
Instantiating RetirementPlanModel.
Running the simulation loop: for i in range(model.scenario_config['projection_years']): model.step().
Extracting results from the model's DataCollector.
Aggregating and comparing results across scenarios.
5. Behavioral Modeling Details (Key Enhancement)
Configuration: The scenario_config dictionary is central. It must contain the detailed parameters for:
ae_outcome_distribution (probabilities for opt-out, stay default, increase to match cap).
match_change_response (probability and target for increasing deferral if match improves).
(Optional) Parameters defining different behavioral_profile types and their associated probabilities or decision weights.
Implementation: The logic defined in the EmployeeAgent.step() method will use these configuration parameters and potentially the agent's behavioral_profile to make probabilistic decisions about enrollment, deferral rates, and opt-outs.
6. Outputs
Primary Output: DataFrame(s) generated from the DataCollector containing:
Model-level variables per year (Total Employer Cost, Participation Rate, Avg Deferral Rate, Headcount, etc.).
(Optional) Agent-level variables per year (allowing detailed distribution analysis).
These DataFrames will be used for comparing scenarios.
7. Technical Considerations
Framework: Requires installing and learning the Mesa library (pip install mesa).
Refactoring: Significant refactoring of the existing Pandas-based functions (determine_eligibility, calculate_contributions, etc.) will be needed to fit the agent-centric step() method logic.
State: Agent attributes hold the state. The model manages global state (year, rules).
Performance: Consider performance implications for very large agent populations. Mesa offers batch processing capabilities if needed.
Modularity: Keep agent decision logic (step) and model processes (termination, hiring, scheduling) distinct.

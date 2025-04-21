import mesa
import pandas as pd
from agents.employee_agent import EmployeeAgent
# Need random for termination probability
import random 
# Need Decimal for accurate financial calculations
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation 
import math # For math.isclose
import logging # For logging
# Need relativedelta for adding months to dates
from dateutil.relativedelta import relativedelta 
import re # Import regular expression module
import yaml  # For loading hazard model parameters
from utils.status_enums import EmploymentStatus

# Configure logging
# Set up basic logging for the module if needed, or rely on root config
# logging.basicConfig(level=logging.DEBUG)

class RetirementPlanModel(mesa.Model):
    """The main model running the retirement plan simulation."""
    def __init__(self, initial_census_df, scenario_config):
        """Create a new RetirementPlanModel.

        Args:
            initial_census_df: DataFrame containing the initial employee census.
            scenario_config: Dictionary with parameters for this simulation scenario.
        """
        super().__init__()
        self.logger = logging.getLogger("RetirementPlanModel")
        self.scenario_config = scenario_config
        # Explicitly set start_year attribute
        self.start_year = self.scenario_config.get('start_year', pd.Timestamp.now().year) 
        # Use current_year consistently
        self.current_year = self.start_year # Initialize current_year from start_year
        self.projection_years = self.scenario_config.get('projection_years', 5)
        # Keep track of the simulation year (distinct from agent's current_year)
        self.year = self.current_year # Initialize simulation year tracker
        self.schedule = mesa.time.RandomActivation(self) # Activate agents randomly each step
        self.population = {} # Using dict for easier agent lookup by id
        self.last_step_hires_list = [] # Add instance variable to store the list

        # --- Initialize Mesa DataCollector ---
        from mesa.datacollection import DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "NumAgents": lambda m: len(m.schedule.agents),
                "Continuous Active": lambda m: sum(1 for a in m.population.values() if a.employment_status == "Continuous Active"),
                "New Hire Active": lambda m: sum(1 for a in m.population.values() if a.employment_status == "New Hire Active"),
                "Experienced Terminated": lambda m: sum(1 for a in m.population.values() if a.employment_status == "Experienced Terminated"),
                "New Hire Terminated": lambda m: sum(1 for a in m.population.values() if a.employment_status == "New Hire Terminated"),
                "Previously Terminated": lambda m: sum(1 for a in m.population.values() if a.employment_status == "Previously Terminated"),
                "PlanCost": lambda m: float(sum(a.contributions_current_year.get('total_employer', Decimal('0.0')) for a in m.population.values())),
                "AvgDeferralPct": lambda m: float(sum(a.deferral_rate for a in m.population.values()) / (len(m.population) if len(m.population) > 0 else 1)),
            },
            agent_reporters={
                "DeferralRate": lambda a: getattr(a, "deferral_rate", None),
                "IsParticipating": lambda a: getattr(a, "is_participating", None),
                "EmploymentStatus": lambda a: getattr(a, "employment_status", None),
                "TenureMonths": lambda a: a._calculate_tenure_months(pd.Timestamp(f"{a.model.year}-12-31")),
                "Cohort": lambda a: (
                    "0-1yr" if a._calculate_tenure_months(pd.Timestamp(f"{a.model.year}-12-31")) <= 12
                    else ("1-3yr" if a._calculate_tenure_months(pd.Timestamp(f"{a.model.year}-12-31")) <= 36 else "3+yr")
                ),
                "EmployerMatch": lambda a: float(a.contributions_current_year.get('employer_match', Decimal('0.0'))),
                "EmployerNEC": lambda a: float(a.contributions_current_year.get('employer_nec', Decimal('0.0'))),
                "HireDate": lambda a: a.hire_date.date() if pd.notnull(a.hire_date) else None,
                "BirthDate": lambda a: a.birth_date.date() if pd.notnull(a.birth_date) else None,
                "TerminationDate": lambda a: a.termination_date.date() if pd.notnull(a.termination_date) else None,
                "ParticipationDate": lambda a: a.participation_date.date() if pd.notnull(a.participation_date) else None
            }
        )

        # --- Process and Validate Configuration ---
        # Ensure plan_rules exists in scenario_config for consistent access
        if 'plan_rules' not in self.scenario_config:
            self.scenario_config['plan_rules'] = {}
            
        # Move top-level config elements into plan_rules if they're not already there
        if 'auto_enrollment' in self.scenario_config and 'auto_enrollment' not in self.scenario_config['plan_rules']:
            self.scenario_config['plan_rules']['auto_enrollment'] = self.scenario_config['auto_enrollment']
            
        self.plan_rules = self.scenario_config['plan_rules']
        self.ae_config = self.plan_rules.get('auto_enrollment', {})
        self.ae_enabled = str(self.ae_config.get('enabled', 'False')).lower() in ('true', '1', 't', 'y', 'yes')
        self.ae_default_rate = Decimal(str(self.ae_config.get('default_rate', '0.0')))
        self.ae_outcome_dist = self.ae_config.get('outcome_distribution', {})
        self.ae_opt_down_target = Decimal(str(self.ae_config.get('opt_down_target_rate', '0.0')))
        self.ae_increase_high_target = Decimal(str(self.ae_config.get('increase_high_target_rate', '0.0')))
        self.ae_prob_opt_out = Decimal('0.0')
        self.ae_prob_stay_default = Decimal('0.0')
        self.ae_prob_opt_down = Decimal('0.0')
        self.ae_prob_increase_match = Decimal('0.0')
        self.ae_prob_increase_high = Decimal('0.0')
        if self.ae_enabled and self.ae_outcome_dist:
            try:
                decimal_probs = {k: Decimal(str(v)) for k, v in self.ae_outcome_dist.items()}
                prob_sum = sum(decimal_probs.values())
                if not math.isclose(prob_sum, Decimal('1.0'), abs_tol=Decimal('1e-9')):
                    raise ValueError(f"AE outcome probabilities in config must sum to 1.0, but sum to {prob_sum}")
                self.ae_prob_opt_out = decimal_probs.get('prob_opt_out', Decimal('0.0'))
                self.ae_prob_stay_default = decimal_probs.get('prob_stay_default', Decimal('0.0'))
                self.ae_prob_opt_down = decimal_probs.get('prob_opt_down', Decimal('0.0'))
                self.ae_prob_increase_match = decimal_probs.get('prob_increase_to_match', Decimal('0.0'))
                self.ae_prob_increase_high = decimal_probs.get('prob_increase_high', Decimal('0.0'))
            except (TypeError, InvalidOperation) as e:
                raise ValueError(f"Error converting AE outcome probabilities to Decimal: {e}. Check config values.")
        self.employer_match_formula = self.scenario_config.get('employer_match_formula', '')
        self.rate_for_max_match = self._parse_max_match_rate(self.employer_match_formula)
        self.logger.info(f"Parsed max match rate from '{self.employer_match_formula}': {self.rate_for_max_match}")
        self.ai_config = self.plan_rules.get('auto_increase', {})
        self.ai_enabled = self.ai_config.get('enabled', False)
        self.ai_increase_rate = Decimal(str(self.ai_config.get('increase_rate', '0.0')))
        self.ai_cap_rate = Decimal(str(self.ai_config.get('cap_rate', '0.0')))
        self.irs_limits = self.scenario_config.get('irs_limits', {})
        self.eligibility_config = self.plan_rules.get('eligibility', {})
        self.start_year = self.scenario_config.get('start_year', pd.Timestamp.now().year)
        self.current_year = self.start_year
        self.projection_years = self.scenario_config.get('projection_years', 5)
        self.year = self.current_year
        self.schedule = mesa.time.RandomActivation(self)
        self.population = {}
        self.last_step_hires_list = []
        self._calculate_initial_comp_stats(initial_census_df)
        self._create_agents_from_census(initial_census_df)
        # DO NOT call self._create_new_hires here! Only call it during simulation steps.

        # Load hazard model parameters if monthly_transition enabled
        if self.scenario_config.get('monthly_transition', False):
            hazard_config = self.scenario_config.get('hazard_model_params', {})
            hazard_file = hazard_config.get('file')
            if hazard_file:
                try:
                    with open(hazard_file, 'r') as f:
                        params = yaml.safe_load(f)
                    self.hazard_params = params
                    coeffs = params.get('cox_coefficients', {})
                    self.cox_coeffs = {k: float(v) for k, v in coeffs.items()}
                    median = params.get('kaplan_meier_median_years', None)
                    self.km_median_years = float(median) if median not in (None, 'inf', float('inf')) else math.inf
                    self.baseline_hazard = (math.log(2) / self.km_median_years) if self.km_median_years != math.inf else 0.0
                    self.logger.info(f"Loaded hazard parameters from {hazard_file}")
                except Exception as e:
                    self.logger.error(f"Error loading hazard model params from {hazard_file}: {e}")
                    self.hazard_params = None
                    self.baseline_hazard = 0.0
        else:
            self.hazard_params = None
            self.baseline_hazard = 0.0

        # Phase 4 Onboarding config: early-tenure hazard & productivity ramp
        onboarding_cfg = self.scenario_config.get('onboarding', {})
        self.onboarding_enabled = onboarding_cfg.get('enabled', False)
        self.onboarding_early_months = int(onboarding_cfg.get('early_tenure_months', 0))
        self.onboarding_multiplier = float(onboarding_cfg.get('hazard_multiplier', 1.0))
        self.productivity_curve = onboarding_cfg.get('productivity_curve', [])

    def _normalize_config(self) -> None:
        """Ensure plan_rules and key config sections are present and normalized."""
        if 'plan_rules' not in self.scenario_config:
            self.scenario_config['plan_rules'] = {}
        if 'auto_enrollment' in self.scenario_config and 'auto_enrollment' not in self.scenario_config['plan_rules']:
            self.scenario_config['plan_rules']['auto_enrollment'] = self.scenario_config['auto_enrollment']
        if 'auto_increase' in self.scenario_config and 'auto_increase' not in self.scenario_config['plan_rules']:
            self.scenario_config['plan_rules']['auto_increase'] = self.scenario_config['auto_increase']
        if 'eligibility' in self.scenario_config and 'eligibility' not in self.scenario_config['plan_rules']:
            self.scenario_config['plan_rules']['eligibility'] = self.scenario_config['eligibility']

    def _calculate_initial_comp_stats(self, initial_census_df: pd.DataFrame) -> None:
        """Calculate initial compensation statistics for the census."""
        column_map = {
            "IsEligible": "is_eligible",        # Boolean flag for plan eligibility
            "IsParticipating": "is_participating", # Boolean flag for plan participation 
            "EnrollmentMethod": "enrollment_method",
            "AEOptOut": "ae_opted_out",
            "AIOptOut": "ai_opted_out",
            "DeferralRate": lambda a: a.deferral_rate,
            # Contributions for the year
            "Contrib_EmpPreTax": lambda a: a.contributions_current_year.get('employee_pretax', Decimal('0.0')),
            "Contrib_EmpCatchUp": lambda a: a.contributions_current_year.get('employee_catchup', Decimal('0.0')),
            "Contrib_ErMatch": lambda a: a.contributions_current_year.get('employer_match', Decimal('0.0')),
            "Contrib_ErNEC": lambda a: a.contributions_current_year.get('employer_nec', Decimal('0.0')),
            "Contrib_TotalEmp": lambda a: a.contributions_current_year.get('total_employee', Decimal('0.0')),
        }
        initial_active_mask = (pd.isna(initial_census_df['termination_date'])) | \
                              (pd.to_datetime(initial_census_df['termination_date']).dt.year >= self.start_year)
        initial_active_census = initial_census_df[initial_active_mask]
        if not initial_active_census.empty and 'gross_compensation' in initial_active_census.columns:
            numeric_comp = pd.to_numeric(initial_active_census['gross_compensation'], errors='coerce').dropna()
            if not numeric_comp.empty:
                self.initial_comp_mean = Decimal(str(numeric_comp.mean()))
                self.initial_comp_std_dev = Decimal(str(numeric_comp.std()))
                if self.initial_comp_std_dev.is_zero():
                    self.initial_comp_std_dev = (self.initial_comp_mean * Decimal('0.05')).quantize(Decimal('0.01'))
                    self.logger.warning(f"Initial census compensation std dev was zero. Using 5% of mean ({self.initial_comp_std_dev}) instead.")
                self.logger.info(f"Calculated initial comp stats: Mean={self.initial_comp_mean:.2f}, StdDev={self.initial_comp_std_dev:.2f}")
            else:
                self.logger.warning("Could not calculate initial compensation stats: No valid numeric compensation data found after filtering.")
                self.initial_comp_mean = Decimal(str(self.scenario_config.get('new_hire_start_salary', 50000)))
                self.initial_comp_std_dev = self.initial_comp_mean * Decimal('0.1')
        else:
            self.logger.warning("Could not calculate initial compensation stats: No active employees or 'gross_compensation' column found.")
            self.initial_comp_mean = Decimal(str(self.scenario_config.get('new_hire_start_salary', 50000)))
            self.initial_comp_std_dev = self.initial_comp_mean * Decimal('0.1')

    def _create_agents_from_census(self, initial_census_df: pd.DataFrame) -> None:
        """Create EmployeeAgent instances from the census and add them to the population and schedule."""
        # Initialize next agent ID based on max ID in census
        max_existing_id = 0
        if 'ssn' in initial_census_df.columns and not initial_census_df['ssn'].empty:
            numeric_ids = pd.to_numeric(initial_census_df['ssn'], errors='coerce').dropna()
            if not numeric_ids.empty:
                max_existing_id = int(numeric_ids.max())
        self.next_agent_id = max_existing_id + 1
        
        # Process each row in the initial census to create agents
        for _, row in initial_census_df.iterrows():
            # Convert row to dictionary for agent initialization
            agent_data = row.to_dict()
            
            # Extract or generate unique ID
            if 'ssn' in agent_data and pd.notna(agent_data['ssn']):
                unique_id = str(agent_data['ssn'])
            else:
                unique_id = str(self.next_agent_id)
                self.next_agent_id += 1
                
            # Create the agent
            agent = EmployeeAgent(unique_id, self, agent_data)
            
            # Add to population dictionary
            self.population[unique_id] = agent
            
            # Add to scheduler if active
            if agent.is_active:
                self.schedule.add(agent)
        
        # Sample a few agents to check their termination dates and status
        sample_agents = list(self.population.values())[:5] # Take first 5 created
        logging.debug("\nSampling first 5 agents created:")
        for i, agent in enumerate(sample_agents):
            logging.debug(f"  Sample agent {i} (ID: {agent.unique_id})")
        
        # Initialize empty list for tracking new hires in steps
        self.last_step_hires_list = []
        
    def step(self):
        """Executes one step of the model, representing one year in the simulation."""
        # --- Reset flags at the start of the year --- 
        reset_count = 0
        for agent in self.population.values():
            if agent.is_new_hire: # Only log if it was True
                logging.debug(f"  [{self.year}] Resetting agent {agent.unique_id} is_new_hire from True to False.")
            agent.is_new_hire = False 
            reset_count += 1
        logging.debug(f"[{self.year}] Finished resetting is_new_hire flag for {reset_count} agents.")
        # --- End Reset flags --- 

        # Clear the list of hires from the *previous* step before creating new ones
        self.last_step_hires_list = []

        # Track start-of-year active headcount for next hires
        active_at_start = sum(1 for a in self.population.values() if a.is_active)
        # Reset hire record for this year (we'll spawn at end)
        self.last_step_hires_list = []

        # 1. Create New Hires for *this* year
        # 2. Identify Terminations for this year
        agents_terminating_this_year = self._identify_terminations()
        agents_terminating_this_year_ids = {a.unique_id for a in agents_terminating_this_year}
        
        # 3. Execute Agent Steps for agents active at the START of the year
        agents_active_at_start = list(self.schedule.agents)
        for agent in agents_active_at_start:
            agent.step() # Perform yearly updates, decisions, calculations

        # 4. Advance the scheduler step
        logging.debug(f"Executing self.schedule.step() for year {self.year}")
        self.schedule.step()
        logging.debug(f"Finished self.schedule.step() for year {self.year}")
        
        # 5. Update employment status based on terminations
        for agent in self.population.values():
            is_new_hire = agent.unique_id in [a.unique_id for a in self.last_step_hires_list]
            terminates_this_year = agent.unique_id in agents_terminating_this_year_ids
            
            # Status logic based on this year's events
            if terminates_this_year:
                # Agent was terminated during this step
                agent.is_active = False # Update active status
                if is_new_hire:
                    agent.employment_status = "New Hire Terminated"
                else:
                    agent.employment_status = "Experienced Terminated"
                    # self.schedule.remove(agent)  # commented out so termination_date is still collected
                # Optional: Add debug print for assigned date
                logging.debug(f"  Termination Processed: Agent {agent.unique_id} terminated on {agent.termination_date.date()} (Year {self.year}) Status: {agent.employment_status}")

            elif is_new_hire: # Not terminated this year, but was hired this year
                agent.is_active = True
                agent.employment_status = EmploymentStatus.NEW_HIRE.value # Ensure this is set for new hires
                # Removed redundant print statement from here, it's in _create_new_hires

            elif agent.is_active: # Not terminated, not hired this year, AND was active at start
                # This relies on agent.is_active reflecting start-of-step status correctly
                agent.employment_status = "Continuous Active"

            else: # Not terminated, not hired, was not active at start (e.g., already terminated)
                # Ensure these are marked inactive and have appropriate status
                agent.is_active = False
                # Should differentiate between 'Terminated' this year and 'Previously Terminated'
                # Let's assign 'Previously Terminated' if they have a termination date before this year
                if agent.termination_date and agent.termination_date.year < self.year:
                    agent.employment_status = "Previously Terminated"
                else:
                    # Fallback if logic missed something, or if term date is exactly this year but they weren't in term list
                    agent.employment_status = "Inactive" # A generic inactive status?

        # 6. Calibrate hires using expected or realized veteran terminations and apply head-on attrition
        growth_rate = self.scenario_config.get('annual_growth_rate', 0.02)
        h_ex = self.scenario_config.get('annual_termination_rate', 0.0)
        h_nh = self.scenario_config.get('new_hire_termination_rate', h_ex)
        delta = round(active_at_start * growth_rate)
        T_ex_real = len(agents_terminating_this_year)
        use_expected = self.scenario_config.get('use_expected_attrition', False)
        T_ex_used = h_ex * active_at_start if use_expected else T_ex_real
        # H_t = (delta + T_ex_used) / (1 - h_nh)
        if (1 - h_nh) > 0:
            H_t = int((delta + T_ex_used) / (1 - h_nh))
        else:
            H_t = int(delta + T_ex_used)
        method = "expected" if use_expected else "realized"
        logging.debug(f"Creating {H_t} new agents (using {method} T_ex={T_ex_used:.2f}, target_g={growth_rate:.2%}) for year {self.year}...")
        # Spawn new hires
        new_hires = self._create_new_hires(num_new_hires=H_t)
        # Determine and apply new-hire attrition head-on
        T_nh = int(round(H_t * h_nh))
        if T_nh > 0 and new_hires:
            term_batch = self.random.sample(new_hires, min(T_nh, len(new_hires)))
            year_end = pd.Timestamp(f"{self.year}-12-31")
            for agent in term_batch:
                hire_ts = pd.to_datetime(agent.hire_date)
                days_range = (year_end - hire_ts).days if pd.notna(hire_ts) else 0
                offset = self.random.randint(0, days_range) if days_range > 0 else 0
                term_date = hire_ts + pd.Timedelta(days=offset) if pd.notna(hire_ts) else year_end
                agent.termination_date = term_date
                agent.is_active = False
                agent.employment_status = "New Hire Terminated"
                # self.schedule.remove(agent)  # commented out so termination_date is still collected
        # Keep survivors
        survivors = [a for a in new_hires if a.is_active]
        self.last_step_hires_list = survivors
        logging.info(f"Δ={delta}, T_ex_real={T_ex_real}, T_ex_used={T_ex_used:.2f}, T_nh={T_nh}, H_t={H_t}")

        # --- Calculate Prorated Compensation for Reporting --- 
        # Do this *after* final termination dates are set for the year, for *all* agents present
        logging.debug(f"[{self.year}] Calculating prorated compensation for reporting...")
        proration_count = 0
        for agent in self.population.values():
            current_year = self.year # Use the model's current year
            year_start = pd.Timestamp(f"{current_year}-01-01")
            year_end = pd.Timestamp(f"{current_year}-12-31")
            total_days_in_year = Decimal((year_end - year_start).days + 1)

            hire_date = pd.to_datetime(agent.hire_date)
            term_date = pd.to_datetime(agent.termination_date) # Use final assigned date

            effective_start_date = max(year_start, hire_date) if pd.notna(hire_date) else year_start
            
            # Use the potentially updated termination date
            if pd.notna(term_date) and term_date.year == current_year:
                effective_end_date = min(year_end, term_date)
            else:
                effective_end_date = year_end # If not terminated this year, they worked till year end

            # Ensure start is not after end (can happen if hired and termed same day)
            effective_end_date = max(effective_start_date, effective_end_date)
            
            days_worked = Decimal((effective_end_date - effective_start_date).days + 1)

            # Calculate prorated compensation
            if total_days_in_year > 0:
                proration_factor = (days_worked / total_days_in_year).min(Decimal('1.0')) # Cap at 1.0
                # Base prorated compensation
                base_comp = agent.gross_compensation * proration_factor
                # Apply onboarding productivity ramp if enabled
                if self.onboarding_enabled:
                    tenure_months = agent._calculate_tenure_months(effective_start_date)
                    if tenure_months <= len(self.productivity_curve):
                        factor = Decimal(str(self.productivity_curve[tenure_months-1]))
                    else:
                        factor = Decimal('1.0')
                else:
                    factor = Decimal('1.0')
                prorated_comp = (base_comp * factor).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            else:
                prorated_comp = Decimal('0.00') # Avoid division by zero

            # Store it on the agent for reporting
            agent.prorated_compensation_for_reporting = prorated_comp
            proration_count += 1
        logging.debug(f"[{self.year}] Finished calculating reporting proration for {proration_count} agents.")
        # --- End Prorated Compensation Calculation ---

        # --- Collect Data --- 
        # Now collects data including agents added/removed in this step
        # Debug print the number of agents with each employment status and new hire status
        status_counts = {}
        new_hire_count = 0
        for agent in self.population.values():
            status = agent.employment_status
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1
            if agent.is_new_hire:
                new_hire_count += 1
                
        logging.debug(f"Employment status counts for year {self.year}:")
        for status, count in status_counts.items():
            logging.debug(f"  {status}: {count}")
        logging.debug(f"  New hires (is_new_hire=True): {new_hire_count}")
            
        logging.debug(f"[{self.year}] Pre-Data Collection Check for specific agent flags:")
        agents_to_check = ['97', '47'] # Agents from user example
        for agent_id in agents_to_check:
            agent = self.population.get(agent_id)
            if agent:
                logging.debug(f"  CHECK Agent {agent_id}: is_new_hire = {getattr(agent, 'is_new_hire', 'ERROR_NO_ATTR')}")
            else:
                logging.debug(f"  CHECK Agent {agent_id}: Not found in population this year.")
        
        logging.debug(f"Collecting data for year {self.year}...")
        self.datacollector.collect(self)
        logging.debug(f"Data collected for year {self.year}.")

        # 7. Increment Year for the next step
        self.year += 1 # Use self.year consistently

        logging.debug(f"--- Finished model step for year {self.year - 1} ---")
        # Optional: Print population/schedule size difference for debugging
        # print(f"End of Year {self.year - 1}: Population={len(self.population)}, Schedule={self.schedule.get_agent_count()}")

    def _identify_terminations(self):
        """Identifies agents terminating this year based on probability, adjusted for tenure, age, and compensation."""
        # Route to monthly transitions if enabled
        if self.scenario_config.get('monthly_transition', False) and getattr(self, 'hazard_params', None):
            return self._identify_monthly_terminations()
        
        terminating_agents = []
        # Iterate ONLY agents currently active in the schedule
        agents_eligible_for_term = [a for a in self.schedule.agents if a.is_active]

        # Use the correct key from config.yaml
        base_term_rate = self.scenario_config.get('annual_termination_rate', 0.05)
        
        # Get the current simulation date for age/tenure calculations
        current_sim_date = pd.Timestamp(year=self.year, month=12, day=31)
        
        logging.debug(f"Identifying terminations for year {self.year} using base rate: {base_term_rate:.2%}")
        logging.debug(f"Adjusting for tenure, age, and compensation factors")
        
        # Track termination counts by category for reporting
        term_counts = {
            'new_hire': 0,
            'short_tenure': 0,
            'mid_tenure': 0,
            'long_tenure': 0,
            'young': 0,
            'mid_age': 0,
            'older': 0,
            'low_comp': 0,
            'mid_comp': 0,
            'high_comp': 0
        }
        
        for agent in agents_eligible_for_term:
            # Start with base termination rate
            adjusted_term_rate = base_term_rate
            
            # 1. Adjust for tenure
            tenure_months = agent._calculate_tenure_months(current_sim_date)
            is_new_hire = agent.is_new_hire
            
            if is_new_hire:
                # New hires have significantly higher turnover (1.5x)
                adjusted_term_rate *= 1.5
                term_category = 'new_hire'
            elif tenure_months < 24:  # Less than 2 years
                # Short tenure has higher turnover (1.3x)
                adjusted_term_rate *= 1.3
                term_category = 'short_tenure'
            elif tenure_months < 60:  # 2-5 years
                # Mid tenure has slightly higher turnover (1.1x)
                adjusted_term_rate *= 1.1
                term_category = 'mid_tenure'
            else:  # 5+ years
                # Long tenure has lower turnover (0.8x)
                adjusted_term_rate *= 0.8
                term_category = 'long_tenure'
            
            # 2. Adjust for age
            age = agent._calculate_age(current_sim_date)
            
            if age < 30:  # Younger employees
                # Higher turnover for younger employees (1.2x)
                adjusted_term_rate *= 1.2
                age_category = 'young'
            elif age < 50:  # Mid-career
                # Baseline turnover for mid-career (1.0x)
                # No adjustment needed
                age_category = 'mid_age'
            else:  # Older employees
                # Lower turnover for older employees (0.9x)
                adjusted_term_rate *= 0.9
                age_category = 'older'
            
            # 3. Adjust for compensation
            comp = agent.gross_compensation
            
            if comp < Decimal('60000'):  # Lower compensation
                # Higher turnover for lower-paid employees (1.2x)
                adjusted_term_rate *= 1.2
                comp_category = 'low_comp'
            elif comp < Decimal('120000'):  # Mid compensation
                # Baseline turnover for mid-compensation (1.0x)
                # No adjustment needed
                comp_category = 'mid_comp'
            else:  # Higher compensation
                # Lower turnover for higher-paid employees (0.8x)
                adjusted_term_rate *= 0.8
                comp_category = 'high_comp'
            
            # Apply termination based on adjusted rate
            if self.random.random() < adjusted_term_rate:
                agent.termination_date = pd.Timestamp(year=self.year, month=12, day=31)
                agent.status = 'Terminated'
                terminating_agents.append(agent)
                
                # Track termination counts by category
                term_counts[term_category] += 1
                term_counts[age_category] += 1
                term_counts[comp_category] += 1
        
        # Report termination statistics
        total_terms = len(terminating_agents)
        if total_terms > 0:
            logging.debug(f"Termination summary for year {self.year}:")
            logging.debug(f"  Total terminations: {total_terms} of {len(agents_eligible_for_term)} eligible agents ({total_terms/len(agents_eligible_for_term):.2%})")
            logging.debug(f"  By tenure: New hires: {term_counts['new_hire']}, Short: {term_counts['short_tenure']}, Mid: {term_counts['mid_tenure']}, Long: {term_counts['long_tenure']}")
            logging.debug(f"  By age: Young: {term_counts['young']}, Mid-age: {term_counts['mid_age']}, Older: {term_counts['older']}")
            logging.debug(f"  By compensation: Low: {term_counts['low_comp']}, Mid: {term_counts['mid_comp']}, High: {term_counts['high_comp']}")

        return terminating_agents

    def _identify_monthly_terminations(self):
        """Identifies terminations using monthly transition probabilities from hazard model."""
        terminating_agents = []
        agents_active = [a for a in self.schedule.agents if a.is_active]
        for agent in agents_active:
            age_at_hire = (agent.hire_date.year - agent.birth_date.year) if agent.hire_date and agent.birth_date else 0
            comp = float(agent.gross_compensation)
            xb = self.cox_coeffs.get('age_at_hire', 0.0) * age_at_hire + self.cox_coeffs.get('gross_compensation', 0.0) * comp
            hazard_annual = self.baseline_hazard * math.exp(xb)
            hazard_monthly = hazard_annual / 12.0
            for month in range(1, 13):
                # adjust monthly hazard for onboarding period
                month_start = pd.Timestamp(year=self.year, month=month, day=1)
                tenure_months = agent._calculate_tenure_months(month_start)
                hazard_m = hazard_monthly * self.onboarding_multiplier if self.onboarding_enabled and tenure_months <= self.onboarding_early_months else hazard_monthly
                try:
                    p = 1.0 - math.exp(-hazard_m)
                except OverflowError:
                    p = 0.0
                if random.random() < p:
                    month_end = month_start + relativedelta(months=1) - pd.Timedelta(days=1)
                    agent.termination_date = month_end
                    terminating_agents.append(agent)
                    break
        self.logger.debug(f"Monthly terminations for year {self.year}: {len(terminating_agents)} agents terminated.")
        return terminating_agents

    def _create_new_hires(self, num_new_hires=None):
        """Creates new agents based on growth rate using census-derived compensation stats."""
        # Override hire count if provided, else compute via calibration
        growth_rate = self.scenario_config.get('annual_growth_rate', 0.02)
        term_rate = self.scenario_config.get('annual_termination_rate', 0.0)
        new_hire_term_rate = self.scenario_config.get('new_hire_termination_rate', term_rate)
        denom_new = 1 - new_hire_term_rate
        # Solve N*(1-t) + H*(1-h_t) = N*(1+g) => H = N*(t+g)/(1-h_t)
        hire_multiplier = (term_rate + growth_rate) / denom_new if denom_new > 0 else growth_rate
        avg_start_age = self.scenario_config.get('new_hire_average_age', 30)
        # Get year start/end for random date generation
        year_start = pd.Timestamp(f"{self.year}-01-01")
        year_end = pd.Timestamp(f"{self.year}-12-31")
        days_in_year = (year_end - year_start).days

        # Use the stored mean and std dev calculated in __init__
        mean_comp = float(self.initial_comp_mean) # Convert Decimal to float for random.normalvariate
        std_dev_comp = float(self.initial_comp_std_dev)

        # Set a minimum floor for generated salary (e.g., 30% of mean, or a fixed amount)
        min_salary_floor = self.initial_comp_mean * Decimal('0.3')

        # Determine how many to spawn
        if num_new_hires is None:
            if self.schedule.get_agent_count() > 0:
                count_to_create = round(self.schedule.get_agent_count() * hire_multiplier)
            else:
                count_to_create = self.scenario_config.get('initial_hires_if_empty', 5)
        else:
            count_to_create = num_new_hires

        new_agents_list = []
        if count_to_create > 0:
            logging.debug(f"Creating {count_to_create} new agents for year {self.year} using census stats (Mean: {self.initial_comp_mean:.2f}, StdDev: {self.initial_comp_std_dev:.2f})...")
            for _ in range(count_to_create):
                unique_id = self.next_agent_id
                self.next_agent_id += 1

                # --- Generate Random Hire Date --- 
                random_day_offset = random.randint(0, days_in_year)
                random_hire_date = year_start + pd.Timedelta(days=random_day_offset)
                # --- End Generate Random Hire Date ---

                birth_year = self.year - avg_start_age
                birth_date = pd.Timestamp(f"{birth_year}-07-01")

                # --- Generate Random Compensation ---
                generated_salary_float = random.normalvariate(mean_comp, std_dev_comp)
                generated_salary = Decimal(str(generated_salary_float)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                generated_salary = max(generated_salary, min_salary_floor)
                # --- End Generate Random Compensation ---

                initial_state = {
                    'ssn': unique_id,
                    'birth_date': birth_date,
                    'hire_date': random_hire_date,
                    'termination_date': pd.NaT,
                    'status': 'Active',
                    'gender': random.choice(['M', 'F']),
                    'gross_compensation': generated_salary,
                    'pre_tax_deferral_percentage': 0,
                    'ae_opted_out': False,
                    'ai_opted_out': False,
                    'is_new_hire': True,
                }
                agent = EmployeeAgent(unique_id, self, initial_state)
                agent.is_new_hire = True
                agent.employment_status = EmploymentStatus.NEW_HIRE.value
                # Only add to scheduler and population here; not during model init
                self.schedule.add(agent)
                self.population[unique_id] = agent
                new_agents_list.append(agent)
        return new_agents_list


    def _parse_max_match_rate(self, formula_string):
        """
        Parses the employer match formula string to find the deferral percentage
        up to which the match applies (rate needed for max match).

        Example formulas:
        - "1.0_of_1.0_up_to_6.0_pct" -> returns Decimal('0.06')
        - "50% up to 6%" -> returns Decimal('0.06')
        - "0.5 of 1.0 up to 0.05" -> returns Decimal('0.05')
        - "match 3%" -> returns Decimal('0.03') # Simple case
        - "100% up to 4% of compensation" -> returns Decimal('0.04')

        Returns:
            Decimal: The deferral rate required for the maximum match, or
                     Decimal('0.0') if parsing fails or no match exists.
        """
        if not formula_string or not isinstance(formula_string, str):
            logging.warning("Employer match formula is missing or not a string. Cannot parse max match rate.")
            return Decimal('0.0')

        # Regex to find patterns like "up to X%", "up to X pct", "up to X.Y", or just "match X%"
        # Handles variations in spacing and percentage symbols/words
        # Prioritize "up to" patterns
        # Updated regex to allow underscores OR spaces: [\s_]*
        match_up_to = re.search(r'up[\s_]*to[\s_]*([0-9.]+)[\s_]*(%|pct)?', formula_string, re.IGNORECASE)
        match_simple = re.search(r'match[\s_]*([0-9.]+)[\s_]*(%|pct)?', formula_string, re.IGNORECASE) # Added simple case

        target_match = match_up_to or match_simple # Prefer "up to" if found

        if target_match:
            try:
                rate_str = target_match.group(1)
                rate_decimal = Decimal(rate_str)
                # If '%' or 'pct' is present, or if the number is >= 1 (and not like 0.06), assume it's a percentage
                # Refined check: Explicit %/pct OR number >= 1.0
                is_percentage = target_match.group(2) is not None or rate_decimal >= Decimal('1.0')
                if is_percentage:
                    # Ensure it's not something like "up to 1.0" meaning 100% rate on a portion
                    # If the number is exactly 1.0 without %/pct, it could be ambiguous
                    # Let's assume >= 1.0 always means percentage unless explicitly < 1.0
                    return (rate_decimal / Decimal('100.0')).quantize(Decimal('0.0001')) # Standardize precision
                else: # Assume it's already a decimal rate (e.g., "up to 0.06")
                    return rate_decimal.quantize(Decimal('0.0001'))
            except (InvalidOperation, IndexError) as e:
                logging.warning(f"Could not parse rate from employer match formula: '{formula_string}'. Found number: '{target_match.group(1) if target_match else 'None'}'. Error: {e}. Defaulting max match rate to 0.")
                return Decimal('0.0')
        else:
            # Fallback: Try extracting any number followed by % or pct if other patterns fail
            # Updated regex to allow underscores OR spaces: [\s_]*
            fallback_match = re.search(r'([0-9.]+)[\s_]*(%|pct)', formula_string, re.IGNORECASE)
            if fallback_match:
                try:
                    rate_str = fallback_match.group(1)
                    rate_decimal = Decimal(rate_str)
                    logging.warning(f"Used fallback pattern for match formula '{formula_string}'. Found rate {rate_str}%.")
                    return (rate_decimal / Decimal('100.0')).quantize(Decimal('0.0001'))
                except (InvalidOperation, IndexError):
                     logging.warning(f"Fallback parsing failed for '{formula_string}'. Defaulting max match rate to 0.")
                     return Decimal('0.0')

            logging.warning(f"Could not find a recognizable rate pattern in employer match formula: '{formula_string}'. Defaulting max match rate to 0.")
            return Decimal('0.0')

import mesa
import pandas as pd
from dateutil.relativedelta import relativedelta
from decimal import Decimal, ROUND_HALF_UP # For precision
import logging # Ensure logging is imported

ZERO_DECIMAL = Decimal('0.00')

class EmployeeAgent(mesa.Agent):
    """An agent representing an employee in the retirement plan simulation."""

    # Enrollment method constants
    ENROLL_METHOD_AE: str = 'AE'
    ENROLL_METHOD_MANUAL: str = 'Manual'
    ENROLL_METHOD_NONE: str = 'None'

    # Employment status constants
    STATUS_PREV_TERMINATED: str = "Previously Terminated"
    STATUS_NEW_HIRE: str = "New Hire Active"
    STATUS_ACTIVE_INITIAL: str = "Active Initial"
    STATUS_ACTIVE_CONTINUOUS: str = "Active Continuous"
    STATUS_NOT_HIRED: str = "Not Hired"
    STATUS_UNKNOWN: str = "Unknown"

    _debug_print_count = 0
    _debug_print_limit = 5

    def __init__(self, unique_id: int, model: "RetirementPlanModel", initial_state: dict) -> None:
        """
        Create a new employee agent.
        Args:
            unique_id: Unique identifier for the agent.
            model: The model instance the agent belongs to.
            initial_state: Dictionary containing initial attributes from the census.
        """
        super().__init__(unique_id, model)
        self.logger = logging.getLogger(f"EmployeeAgent.{unique_id}")

        # Core Demographics & Status
        self.birth_date = initial_state.get('birth_date')
        self.hire_date = initial_state.get('hire_date')
        self.termination_date = initial_state.get('termination_date', None)
        self.status = initial_state.get('status', 'Active')
        self.role = initial_state.get('role')
        self.gross_compensation = initial_state.get('gross_compensation', 0)
        self.employment_status = None
        self.hire_year = pd.to_datetime(self.hire_date).year if self.hire_date else None
        self.is_new_hire = False  # Flag to track new hires separately

        # Set employment status and active flag
        self._initialize_employment_status()

        # Plan-Specific State
        self.is_eligible: bool = False
        initial_deferral_percentage = initial_state.get('pre_tax_deferral_percentage', 0.0)
        self.deferral_rate: Decimal = Decimal(str(initial_deferral_percentage)) / Decimal('100.0')
        self.is_participating: bool = self.deferral_rate > ZERO_DECIMAL
        self.enrollment_method: str = initial_state.get('enrollment_method', self.ENROLL_METHOD_NONE)
        self.ae_opted_out: bool = str(initial_state.get('ae_opted_out', 'False')).lower() in ('true', '1', 't', 'y', 'yes')
        self.ai_opted_out: bool = str(initial_state.get('ai_opted_out', 'False')).lower() in ('true', '1', 't', 'y', 'yes')
        self.behavioral_profile: str = initial_state.get('behavioral_profile', 'Default')
        self.contributions_current_year: dict = {}
        self.prorated_compensation_for_reporting: Decimal = ZERO_DECIMAL

    def _initialize_employment_status(self) -> None:
        """
        Determine initial employment status and is_active flag based on hire/termination dates and model start year.
        """
        term_date = self.termination_date
        if pd.isna(term_date):
            self.is_active = True
        else:
            self.is_active = (term_date.year >= self.model.start_year)
        if not self.is_active:
            self.employment_status = self.STATUS_PREV_TERMINATED
        else:
            if self.hire_date and self.hire_date.year == self.model.start_year:
                self.employment_status = self.STATUS_NEW_HIRE
            elif self.hire_date and self.hire_year < self.model.start_year:
                self.employment_status = self.STATUS_ACTIVE_CONTINUOUS
            else:
                self.employment_status = self.STATUS_ACTIVE_INITIAL
        if self.hire_date and self.model.start_year:
            if self.hire_year < self.model.start_year:
                self.employment_status = self.STATUS_ACTIVE_CONTINUOUS
            elif self.hire_year == self.model.start_year:
                self.employment_status = self.STATUS_NEW_HIRE
            else:
                self.employment_status = self.STATUS_NOT_HIRED
        else:
            self.employment_status = self.STATUS_UNKNOWN
        # Logging for debug
        if EmployeeAgent._debug_print_count < EmployeeAgent._debug_print_limit:
            self.logger.debug(f"Init: term_date={repr(self.termination_date)}, is_active={self.is_active}, status={self.status}, employment_status={self.employment_status}")
            EmployeeAgent._debug_print_count += 1

    def _calculate_age(self, current_date: pd.Timestamp) -> int:
        """Calculates the agent's age as of a given date."""
        if pd.isnull(self.birth_date):
            return 0
        return relativedelta(current_date, self.birth_date).years

    def _calculate_tenure_months(self, current_date):
        """Calculates the agent's tenure in months as of a given date."""
        if pd.isna(self.hire_date):
            return 0
        # Ensure hire_date is Timestamp
        hire_date_ts = pd.Timestamp(self.hire_date)
        # Calculate the difference in months using period arithmetic
        # Use year and month components for robustness across date ranges
        delta_years = current_date.year - hire_date_ts.year
        delta_months = current_date.month - hire_date_ts.month
        total_months = delta_years * 12 + delta_months

        # Adjust if current_date's day is before hire_date's day in the same month/year or subsequent months
        if current_date.day < hire_date_ts.day:
            total_months -= 1

        return max(0, total_months) # Ensure tenure is non-negative

    def _update_compensation(self):
        """Updates the agent's gross compensation based on annual increase rules."""
        # Only apply increase if agent is active
        if self.status == 'Active':
            # Use the correct config key 'annual_compensation_increase_rate' instead of 'annual_comp_increase_rate'
            increase_rate = self.model.scenario_config.get('annual_compensation_increase_rate', 0.03)
            # Convert increase_rate to Decimal for calculation
            decimal_increase_rate = Decimal(str(increase_rate))
            self.gross_compensation *= (Decimal('1.0') + decimal_increase_rate)
            # Ensure result is rounded appropriately (e.g., to 2 decimal places)
            self.gross_compensation = self.gross_compensation.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            # print(f"DEBUG Agent {self.unique_id} Year {self.model.year}: Comp updated to {self.gross_compensation}") # Optional debug

    def _update_eligibility(self):
        """Updates the agent's eligibility status based on plan rules. 
           Called during the agent's step()."""
        # If already eligible from immediate check at hire (for min_service=0), keep it.
        if self.is_eligible:
             # Optional: Could add logic here if eligibility can be lost later
             # print(f"WARN: Agent {self.unique_id} participating but technically ineligible. Grandfathered.")
             self.is_eligible = True # Ensure it stays true
             return

        # Access eligibility rules directly from model's pre-processed plan_rules
        eligibility_rules = self.model.plan_rules.get('eligibility', {})
        min_age_req = eligibility_rules.get('min_age', 0)
        min_service_req = eligibility_rules.get('min_service_months', 0)

        # Check Service Requirement - Only calculate tenure if min_service > 0
        meets_service_req = False
        if min_service_req == 0:
            meets_service_req = True # Handled at hire time, but re-confirm here for safety
        else:
            # Calculate tenure based on year-end date for service req > 0
            current_sim_date = pd.Timestamp(f"{self.model.year}-12-31")
            tenure_months = self._calculate_tenure_months(current_sim_date)
            meets_service_req = (tenure_months >= min_service_req)

        # Check Age Requirement (using year-end date)
        current_sim_date_age = pd.Timestamp(f"{self.model.year}-12-31")
        age = self._calculate_age(current_sim_date_age)
        meets_age_req = (age >= min_age_req)

        # Update eligibility based on *this step's* checks
        self.is_eligible = meets_age_req and meets_service_req

        # Optional: Add final eligibility debug print
        is_new_hire_flag = getattr(self, 'is_new_hire', False)
        if is_new_hire_flag:
            print(f"DEBUG NH ELIG CHECK: {self.unique_id} Year {self.model.year}: Eligibility check in step(). Meets Age: {meets_age_req} (Age {age}), Meets Service: {meets_service_req} (Tenure {tenure_months if min_service_req > 0 else 'N/A'}). Final Eligible: {self.is_eligible}")

    def _make_deferral_decision(self):
        """Handles the agent's decision regarding participation and deferral rate.
           Incorporates AE/AI features and voluntary behavioral changes."""

        was_participating = self.is_participating # Capture status at the START of the decision process
        is_new_hire_flag = getattr(self, 'is_new_hire', False) # Check if it's a new hire

        if is_new_hire_flag:
            print(f"DEBUG NH {self.unique_id} Year {self.model.year}: Entering _make_deferral_decision. Eligible: {self.is_eligible}, Participating: {self.is_participating}, AE Opt-Out: {self.ae_opted_out}, Initial Rate: {self.deferral_rate}")


        # --- Eligibility Check --- 
        if not self.is_eligible: # Check eligibility status updated earlier in step()
            # Agent does not meet formal eligibility rules for THIS year
            if was_participating: # Check INITIAL status
                # Grandfathering: Agent was already participating, let them continue.
                # Maintain current status and rate. They are not subject to AE/AI this step.
                # print(f"WARN: Agent {self.unique_id} participating but technically ineligible. Grandfathered.")
                self.is_participating = True # Ensure it stays true
            else:
                # Ineligible and wasn't participating - definitely stays non-participating
                self.is_participating = False
                # Ensure rate is zero if ineligible and not grandfathered
                self.deferral_rate = ZERO_DECIMAL
                self.enrollment_method = self.ENROLL_METHOD_NONE
            # Skip the rest of the AE/AI/Voluntary logic for ineligible agents this year
            if is_new_hire_flag: print(f"DEBUG NH {self.unique_id} Year {self.model.year}: Ineligible branch. Final Rate: {self.deferral_rate}")
            return 
        
        # --- Agent is Eligible --- 
        # Proceed with AE/AI/Voluntary logic only for eligible agents
        if is_new_hire_flag: print(f"DEBUG NH {self.unique_id} Year {self.model.year}: Eligible branch.")

        # behavior_config = self.model.scenario_config.get('behavioral_params', {}) # Still needed for behavior
        # Use pre-processed flags directly from the model
        ae_enabled = self.model.ae_enabled 
        ai_enabled = self.model.ai_enabled 
        current_rate = self.deferral_rate # Store initial rate for the step

        # --- Decision Logic for Eligible Agents --- 
        if not was_participating: # Use INITIAL status
            # Agent was not participating at the start of the year
            # Apply Auto-Enrollment only if eligible, enabled, and agent hasn't previously opted out
            if is_new_hire_flag: print(f"DEBUG NH {self.unique_id} Year {self.model.year}: Entering 'not was_participating' block.") # ADDED THIS
            if is_new_hire_flag: print(f"DEBUG NH {self.unique_id} Year {self.model.year}: Was not participating branch.")
            
            # Explicitly re-check ae_enabled from model to ensure it's the correct value
            ae_enabled = self.model.ae_enabled
            
            if is_new_hire_flag: 
                print(f"DEBUG NH PRE-AE CHECK {self.unique_id} Year {self.model.year}: ae_enabled={ae_enabled}, self.ae_opted_out={self.ae_opted_out}")
            if ae_enabled and not self.ae_opted_out:
                 # Apply Auto-Enrollment logic for newly eligible or previously non-participating
                 if is_new_hire_flag: print(f"DEBUG NH {self.unique_id} Year {self.model.year}: AE enabled & not opted out branch.")

                 # Fetch AE config details directly from the model where they were processed
                 prob_opt_out = self.model.ae_prob_opt_out
                 prob_stay_default = self.model.ae_prob_stay_default
                 prob_opt_down = self.model.ae_prob_opt_down
                 prob_increase_match = self.model.ae_prob_increase_match
                 prob_increase_high = self.model.ae_prob_increase_high # This should sum to 1 with others

                 default_rate = self.model.ae_default_rate
                 opt_down_target = self.model.ae_opt_down_target
                 rate_for_match = self.model.rate_for_max_match # Rate employee needs to contribute for full match
                 increase_high_target = self.model.ae_increase_high_target

                 # --- ADD DEBUG PRINTS FOR TARGET RATES ---
                 if is_new_hire_flag: # Only print for new hires to reduce noise
                     print(f"DEBUG NH RATES {self.unique_id} Year {self.model.year}: Default Rate = {default_rate}")
                     print(f"DEBUG NH RATES {self.unique_id} Year {self.model.year}: Opt Down Target = {opt_down_target}")
                     print(f"DEBUG NH RATES {self.unique_id} Year {self.model.year}: Rate for Match = {rate_for_match}")
                     print(f"DEBUG NH RATES {self.unique_id} Year {self.model.year}: Increase High Target = {increase_high_target}")
                 # --- END DEBUG PRINTS ---

                 # Generate random number to determine outcome
                 rand_choice = Decimal(str(self.random.random())) # Use Decimal for comparison
                 new_rate = ZERO_DECIMAL # Default to opt-out
                 self.is_participating = False
                 self.enrollment_method = self.ENROLL_METHOD_NONE
                 self.ae_opted_out = True # Assume opt-out unless overridden
                 outcome_desc = "Opt Out"

                 # Determine outcome based on cumulative probability
                 cumulative_prob = ZERO_DECIMAL
                 if rand_choice <= (cumulative_prob + prob_opt_out):
                     # Outcome is Opt Out (handled by defaults above)
                     pass
                 else:
                     cumulative_prob += prob_opt_out
                     if rand_choice <= (cumulative_prob + prob_stay_default):
                         # Outcome is Stay Default
                         new_rate = default_rate
                         self.is_participating = True
                         self.enrollment_method = self.ENROLL_METHOD_AE
                         self.ae_opted_out = False
                         outcome_desc = "Stay Default"
                     else:
                         cumulative_prob += prob_stay_default
                         if rand_choice <= (cumulative_prob + prob_opt_down):
                             # Outcome is Opt Down
                             new_rate = opt_down_target
                             self.is_participating = True
                             self.enrollment_method = self.ENROLL_METHOD_AE
                             self.ae_opted_out = False
                             outcome_desc = "Opt Down"
                         else:
                             cumulative_prob += prob_opt_down
                             if rand_choice <= (cumulative_prob + prob_increase_match):
                                 # Outcome is Increase to Match
                                 # Use the actual rate needed for max match from model
                                 new_rate = rate_for_match 
                                 self.is_participating = True
                                 self.enrollment_method = self.ENROLL_METHOD_AE
                                 self.ae_opted_out = False
                                 outcome_desc = "Increase to Match"
                             else:
                                 # Outcome is Increase High (covers the rest)
                                 new_rate = increase_high_target
                                 self.is_participating = True
                                 self.enrollment_method = self.ENROLL_METHOD_AE
                                 self.ae_opted_out = False
                                 outcome_desc = "Increase High"
                 
                 # Set the final determined rate
                 self.deferral_rate = new_rate
                 if is_new_hire_flag: print(f"DEBUG NH {self.unique_id} Year {self.model.year}: AE Outcome '{outcome_desc}' (Rand: {rand_choice:.4f}). Rate set to {self.deferral_rate}")

            else:
                # Remain non-participating unless they make a voluntary choice (if implemented)
                self.is_participating = False
                self.deferral_rate = ZERO_DECIMAL
                self.enrollment_method = self.ENROLL_METHOD_NONE
                if is_new_hire_flag: print(f"DEBUG NH NO AE {self.unique_id} Year {self.model.year}: AE skipped. Enabled={ae_enabled}, OptedOut={self.ae_opted_out}")
                # Add debug print for new hires in this branch
                if is_new_hire_flag: print(f"DEBUG NH {self.unique_id} Year {self.model.year}: AE disabled or opted out branch. Final Rate: {self.deferral_rate}")

    
        elif was_participating: # Use INITIAL status
            # ... existing AI / Voluntary logic ...
            # No extra debug needed here for now
            pass
        
        # If AI didn't apply or was opted out, consider voluntary changes
        if is_new_hire_flag: print(f"DEBUG NH {self.unique_id} Year {self.model.year}: Exiting _make_deferral_decision. Final Rate: {self.deferral_rate}")
        
        # Final check: Ensure is_participating reflects the final deferral rate after all decisions
        self.is_participating = self.deferral_rate > ZERO_DECIMAL

    def _calculate_contributions(self):
        """Calculates employee and employer contributions for the year using Decimal."""
        # Reset contributions for the year
        self.contributions_current_year = {
            'employee_pretax': ZERO_DECIMAL,
            'employee_catchup': ZERO_DECIMAL,
            'employer_match': ZERO_DECIMAL,
            'employer_nec': ZERO_DECIMAL,
            'total_employee': ZERO_DECIMAL,
            'total_employer': ZERO_DECIMAL,
        }

        # --- Check for Activity/Participation for Actual Contributions --- 
        # Check self.is_active which reflects status *after* termination processing in model.step
        if not self.is_active or not self.is_participating:
            # No contributions if inactive or not participating
            return 
    
        # --- Use *Annual* Compensation for Calculations --- 
        # Proration is handled externally by the model before data collection for reporting.
        # Contribution calculation should still use the appropriate base (here, annual, but limits apply)
        gross_comp = Decimal(str(self.gross_compensation)) # Use annual comp as base for limits

        # --- Get IRS Limits (Existing Logic) --- 
        config = self.model.scenario_config
        # Ensure IRS limits are nested by year in the config
        irs_limits = config.get('irs_limits', {}).get(self.model.year, {}) # Use integer year directly as key
        if not irs_limits:
             print(f"Warning: IRS limits not found for year {self.model.year} in config.")
             # Provide default fallbacks or handle error appropriately
             irs_limits = {'compensation_limit': Decimal('345000'), 'deferral_limit': Decimal('23000'), 'catchup_limit': Decimal('7500'), 'catchup_eligibility_age': 50}
        else:
            # Ensure limits from config are Decimal
            irs_limits = {k: Decimal(str(v)) if k != 'catchup_eligibility_age' else v for k, v in irs_limits.items()}

        comp_limit = irs_limits.get('compensation_limit', Decimal('345000'))
        deferral_limit = irs_limits.get('deferral_limit', Decimal('23000'))
        catchup_limit = irs_limits.get('catchup_limit', Decimal('7500'))
        catchup_age = irs_limits.get('catchup_eligibility_age', 50)

        # Use Decimal for compensation and deferral rate
        deferral_rate = Decimal(str(self.deferral_rate))

        # Use compensation up to the limit for calculations
        # NOTE: This eligible_comp might need adjustment if rules depend on prorated comp,
        # but for standard limits applied to deferrals/match, using annual comp is often correct.
        # If match *formula* itself depends on *actual* pay received, this needs more thought.
        # Assuming limit applies to potential annual earnings for now.
        eligible_comp = min(gross_comp, comp_limit)
        if eligible_comp <= ZERO_DECIMAL: # Avoid division by zero later
            return
        # 1. Calculate Employee Pre-tax Deferral
        potential_deferral = (eligible_comp * deferral_rate).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        actual_deferral = min(potential_deferral, deferral_limit)
        # Ensure non-negative
        self.contributions_current_year['employee_pretax'] = max(ZERO_DECIMAL, actual_deferral)

        # 2. Calculate Employee Catch-up Contribution
        # Make sure birth_date is available and calculation is robust
        catchup_contribution = ZERO_DECIMAL # Initialize with ZERO_DECIMAL
        try:
             current_sim_date = pd.Timestamp(f"{self.model.year}-12-31")
             age = self._calculate_age(current_sim_date)
             if age >= catchup_age:
                 # Calculate how much deferral room is left under the regular limit
                 remaining_potential = max(ZERO_DECIMAL, potential_deferral - actual_deferral)
                 # Catch-up is the minimum of the remaining potential deferral and the catch-up limit
                 catchup_contribution = min(remaining_potential, catchup_limit)
                 # Ensure non-negative and round
                 self.contributions_current_year['employee_catchup'] = max(ZERO_DECIMAL, catchup_contribution).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except Exception as e:
             print(f"Error calculating age or catch-up for agent {self.unique_id}: {e}")
             # Handle error, maybe default catch-up to 0

        total_employee_contribution = self.contributions_current_year['employee_pretax'] + self.contributions_current_year['employee_catchup']
        self.contributions_current_year['total_employee'] = total_employee_contribution # Already rounded individually

        # 3. Calculate Employer Match
        match_formula = config.get('employer_match_formula', "")
        employer_match = ZERO_DECIMAL # Initialize with ZERO_DECIMAL
        if match_formula and total_employee_contribution > ZERO_DECIMAL:
             try:
                 # Example parsing for "1.0_of_1.0_up_to_6.0_pct"
                 parts = match_formula.split('_')
                 match_rate = Decimal(str(parts[0]))           # e.g., 1.0 (100%)
                 match_on_rate = Decimal(str(parts[2]))        # e.g., 1.0 (applied to 100% of deferral)
                 match_cap_pct = Decimal(str(parts[5])) / Decimal('100.0') # e.g., 0.06 (up to 6% of comp)

                 # Calculate the maximum compensation base for the match cap
                 match_comp_base = (eligible_comp * match_cap_pct).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

                 # Determine the amount of employee contribution eligible for matching
                 # It's the minimum of their total contribution and the compensation base for the match cap
                 deferral_subject_to_match = min(total_employee_contribution, match_comp_base)

                 # Apply the match rate to the eligible deferral amount
                 employer_match = (deferral_subject_to_match * match_rate * match_on_rate).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

             except (IndexError, ValueError, TypeError) as e:
                 print(f"Warning: Could not parse match formula '{match_formula}' for agent {self.unique_id}: {e}")
                 # employer_match remains ZERO_DECIMAL

        # Ensure non-negative
        self.contributions_current_year['employer_match'] = max(ZERO_DECIMAL, employer_match)

        # 4. Calculate Employer Non-Elective Contribution (NEC)
        nec_formula = config.get('employer_nec_formula', "")
        employer_nec = ZERO_DECIMAL # Initialize with ZERO_DECIMAL
        if nec_formula:
            try:
                # Example parsing for "3.0_pct"
                parts = nec_formula.split('_')
                nec_rate = Decimal(str(parts[0])) / Decimal('100.0') # e.g., 0.03
                # NEC is based on eligible_comp (which is already prorated and capped)
                employer_nec = (eligible_comp * nec_rate).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            except (IndexError, ValueError, TypeError) as e:
                print(f"Warning: Could not parse NEC formula '{nec_formula}' for agent {self.unique_id}: {e}")
                # employer_nec remains ZERO_DECIMAL
        
        # Ensure non-negative
        self.contributions_current_year['employer_nec'] = max(ZERO_DECIMAL, employer_nec)

        # Sum total employer contributions
        self.contributions_current_year['total_employer'] = self.contributions_current_year['employer_match'] + self.contributions_current_year['employer_nec']

        # Optional: Final debug print of contributions
        # print(f"DEBUG CONTR Agent {self.unique_id} Year {self.model.year}: Comp={gross_comp:.2f}, EligComp={eligible_comp:.2f}, Deferral={self.contributions_current_year['employee_pretax']:.2f}, Catchup={self.contributions_current_year['employee_catchup']:.2f}, Match={self.contributions_current_year['employer_match']:.2f}, NEC={self.contributions_current_year['employer_nec']:.2f}")
    
    def step(self):
        """Execute one step of the agent's behavior for the current year."""
        # 0. Check Status: If not 'Active', do nothing further in the step
        if self.status != 'Active':
            return

        # --- Agent Logic --- 
        # 1. Update Compensation (Using helper method)
        self._update_compensation() # Consider using Decimal here too if precision matters

        # 2. Check/Update Eligibility (Using helper method)
        self._update_eligibility()
        
        # Make decisions based on updated state
        self._make_deferral_decision()

        # 4. Calculate Contributions
        self._calculate_contributions()

        # ... (rest of step method if any) ...

    def _determine_status_for_year(self):
        # Check if they were a new hire flagged earlier in the model step
        if self.is_new_hire:
            # Keep the 'New Hire Active' status set by the model for this year's reporting
            # Don't change it here.
            # **** Correction: Explicitly set status if is_new_hire is True ****
            self.employment_status = "New Hire Active"
            # print(f"DEBUG Agent {self.unique_id} Year {self.model.year}: Status explicitly set to 'New Hire Active'.")
        else:
            # Existing active employee, not a new hire this year
            self.employment_status = "Active Continuous"

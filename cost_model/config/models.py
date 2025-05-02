# cost_model/config/models.py
"""
Pydantic models for validating the structure and types of the configuration
loaded from YAML files (e.g., config.yaml).
"""

import logging
import numpy as np
from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)

# --- Low-level Reusable Models ---

class IRSYearLimits(BaseModel):
    """IRS limits for a specific year."""
    compensation_limit: int = Field(..., description="Annual compensation limit (e.g., 401(a)(17))")
    deferral_limit: int = Field(..., description="Elective deferral limit (e.g., 402(g))")
    catchup_limit: int = Field(..., description="Catch-up contribution limit for age 50+")
    catchup_eligibility_age: int = Field(50, description="Age at which catch-up contributions are allowed")

class MatchTier(BaseModel):
    """Defines a single tier for employer matching contributions."""
    match_rate: float = Field(..., ge=0.0, description="Employer match rate for this tier (e.g., 0.5 for 50%)")
    cap_deferral_pct: float = Field(..., ge=0.0, description="Maximum employee deferral percentage this tier applies to (e.g., 0.06 for 6%)")
    # Optional: Add tenure requirements if needed later
    # min_tenure_months: Optional[int] = Field(None, ge=0)

class AutoEnrollOutcomeDistribution(BaseModel):
    """Probability distribution for auto-enrollment outcomes."""
    prob_opt_out: float = Field(..., ge=0.0, le=1.0)
    prob_stay_default: float = Field(..., ge=0.0, le=1.0)
    prob_opt_down: float = Field(..., ge=0.0, le=1.0)
    prob_increase_to_match: float = Field(..., ge=0.0, le=1.0)
    prob_increase_high: float = Field(..., ge=0.0, le=1.0)

    @root_validator
    def check_probabilities_sum_to_one(cls, values):
        """Validate that the probabilities sum approximately to 1.0."""
        prob_sum = sum(
            values.get(k, 0.0) for k in [
                'prob_opt_out', 'prob_stay_default', 'prob_opt_down',
                'prob_increase_to_match', 'prob_increase_high'
            ]
        )
        if not np.isclose(prob_sum, 1.0):
            logger.warning(f"Auto-enrollment outcome probabilities sum to {prob_sum:.4f}, not 1.0. Check config.")
            # Depending on strictness, you might raise ValueError here instead of just warning
            # raise ValueError(f"Outcome probabilities must sum to 1.0, got {prob_sum}")
        return values

# --- Plan Rules Models ---

class EligibilityRules(BaseModel):
    min_age: Optional[int] = Field(None, ge=0, description="Minimum age requirement (years)")
    min_service_months: Optional[int] = Field(None, ge=0, description="Minimum service requirement (months)")
    min_hours_worked: Optional[int] = Field(None, ge=0, description="Minimum hours worked requirement")
    # Add other eligibility criteria like employment status, hours, etc. if needed

class OnboardingBumpRules(BaseModel):
    enabled: bool = False
    method: Optional[str] = Field(None, description="Method: 'flat_rate' or 'sample_plus_rate'")
    rate: Optional[float] = Field(None, ge=0.0, description="Rate for bump (used by both methods)")

    @root_validator
    def check_method_and_rate(cls, values):
        enabled = values.get('enabled')
        method = values.get('method')
        rate = values.get('rate')
        if enabled and not method:
            raise ValueError("Onboarding bump method must be specified if enabled.")
        if enabled and rate is None:
             raise ValueError("Onboarding bump rate must be specified if enabled.")
        if enabled and method not in ['flat_rate', 'sample_plus_rate']:
            raise ValueError(f"Invalid onboarding bump method: {method}")
        return values

class AutoEnrollmentRules(BaseModel):
    enabled: bool = False
    window_days: Optional[int] = Field(90, description="Enrollment window in days after eligibility", ge=0)
    proactive_enrollment_probability: float = Field(0.0, ge=0.0, le=1.0)
    proactive_rate_range: Optional[Tuple[float, float]] = Field(None, description="Optional [min, max] rate range for proactive enrollment")
    default_rate: float = Field(0.0, ge=0.0, description="Default deferral rate as percentage (e.g., 0.03 for 3%)")
    opt_down_target_rate: Optional[float] = Field(None, ge=0.0)
    increase_to_match_rate: Optional[float] = Field(None, ge=0.0)
    increase_high_rate: Optional[float] = Field(None, ge=0.0)
    outcome_distribution: Optional[AutoEnrollOutcomeDistribution] = None
    re_enroll_existing: Optional[bool] = Field(False, description="Whether to re-enroll existing participants with 0% or previous opt-outs")

    @validator('proactive_rate_range')
    def check_proactive_rate_range(cls, v):
        if v is not None:
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise ValueError("proactive_rate_range must be a list/tuple of two numbers")
            min_r, max_r = v
            if not (isinstance(min_r, (int, float)) and isinstance(max_r, (int, float))):
                raise ValueError("proactive_rate_range values must be numbers")
            if min_r < 0 or max_r < 0:
                raise ValueError("proactive_rate_range values cannot be negative")
            if min_r > max_r:
                raise ValueError("proactive_rate_range min cannot be greater than max")
        return v

    @root_validator
    def check_distribution_if_enabled(cls, values):
        enabled = values.get('enabled')
        distribution = values.get('outcome_distribution')
        if enabled and not distribution:
            raise ValueError("outcome_distribution must be defined if auto_enrollment is enabled.")
        # Add checks for optional rates if needed based on distribution usage
        return values

class AutoIncreaseRules(BaseModel):
    enabled: bool = False
    increase_rate: float = Field(0.0, ge=0.0)
    cap_rate: float = Field(0.0, ge=0.0)
    apply_to_new_hires_only: bool = False
    re_enroll_existing_below_cap: bool = False

    @root_validator
    def check_flags(cls, values):
        """Ensure mutually exclusive flags are not both true."""
        new_hires_only = values.get('apply_to_new_hires_only')
        re_enroll = values.get('re_enroll_existing_below_cap')
        if new_hires_only and re_enroll:
            raise ValueError("Cannot set both 'apply_to_new_hires_only' and 're_enroll_existing_below_cap' to true.")
        return values

class EmployerMatchRules(BaseModel):
    tiers: List[MatchTier] = Field(default_factory=list)
    dollar_cap: Optional[float] = Field(None, ge=0.0, description="Optional annual dollar cap on total match")

class EmployerNecRules(BaseModel):
    rate: float = Field(0.0, ge=0.0, description="Non-elective contribution rate as percentage (e.g., 0.03 for 3%)")

class BehavioralParams(BaseModel):
    voluntary_enrollment_rate: float = Field(0.0, ge=0.0, le=1.0)
    voluntary_default_deferral: float = Field(0.0, ge=0.0)
    voluntary_window_days: int = Field(180, ge=0)
    voluntary_change_probability: float = Field(0.0, ge=0.0, le=1.0)
    prob_increase_given_change: float = Field(0.0, ge=0.0, le=1.0)
    prob_decrease_given_change: float = Field(0.0, ge=0.0, le=1.0)
    prob_stop_given_change: float = Field(0.0, ge=0.0, le=1.0)
    voluntary_increase_amount: float = Field(0.0, ge=0.0)
    voluntary_decrease_amount: float = Field(0.0, ge=0.0)

    @root_validator
    def check_change_probs(cls, values):
        """Validate that the conditional change probabilities sum approximately to 1.0 or less."""
        prob_sum = sum(
            values.get(k, 0.0) for k in [
                'prob_increase_given_change', 'prob_decrease_given_change', 'prob_stop_given_change'
            ]
        )
        # Allow sum to be less than 1 (implies possibility of no change even if change event occurs)
        if prob_sum > 1.0001: # Allow for slight float inaccuracies
            logger.warning(f"Behavioral change outcome probabilities sum to {prob_sum:.4f}, > 1.0. Check config.")
            # raise ValueError(f"Change outcome probabilities must sum to <= 1.0, got {prob_sum}")
        return values


class ContributionRules(BaseModel):
    """Placeholder if specific contribution rules beyond match/NEC are needed."""
    enabled: bool = True # Often controls whether any contributions are calculated

class PlanRules(BaseModel):
    """Container for all plan rule configurations."""
    eligibility: Optional[EligibilityRules] = None
    onboarding_bump: Optional[OnboardingBumpRules] = None
    auto_enrollment: Optional[AutoEnrollmentRules] = None
    auto_increase: Optional[AutoIncreaseRules] = None
    employer_match: Optional[EmployerMatchRules] = None
    employer_nec: Optional[EmployerNecRules] = None
    irs_limits: Dict[int, IRSYearLimits] = Field(default_factory=dict)
    behavioral_params: Optional[BehavioralParams] = None
    contributions: Optional[ContributionRules] = Field(default_factory=ContributionRules) # Ensure default exists


# --- Top-Level Configuration Models ---

class CompensationParams(BaseModel):
    """Generic compensation parameters."""
    comp_base_salary: float
    comp_std: Optional[float] = None # Optional if using lognormal sigma
    comp_increase_per_age_year: float
    comp_increase_per_tenure_year: float
    comp_log_mean_factor: float = 1.0
    comp_spread_sigma: float = 0.3
    comp_min_salary: float

class OnboardingConfig(BaseModel):
    """Configuration for early tenure dynamics."""
    enabled: bool = False
    early_tenure_months: int = Field(6, ge=0)
    hazard_multiplier: float = Field(1.0, ge=0.0)
    productivity_curve: Optional[List[float]] = None

class HazardModelParams(BaseModel):
    """Reference to external hazard model parameters."""
    file: str # Path to the hazard model params file

class GlobalParameters(BaseModel):
    start_year: int
    projection_years: int
    random_seed: Optional[int] = None

    # HR Dynamics
    annual_compensation_increase_rate: float = Field(0.0, ge=0.0)
    annual_termination_rate: float = Field(0.0, ge=0.0, le=1.0)
    new_hire_termination_rate: float = Field(0.0, ge=0.0, le=1.0)
    use_expected_attrition: Optional[bool] = False # Optional flag
    new_hire_start_salary: Optional[float] = Field(None, ge=0.0) # Can be overridden by params
    new_hire_average_age: Optional[float] = Field(None, ge=0.0) # Can be overridden by params
    annual_growth_rate: float = Field(0.0) # Can be negative
    monthly_transition: Optional[bool] = False # Optional flag
    maintain_headcount: bool = False # Default to false if growth rate used

    # Hazard Model (Optional)
    hazard_model_params: Optional[HazardModelParams] = None

    # Role & Compensation Details
    role_distribution: Optional[Dict[str, float]] = None
    new_hire_compensation_params: Optional[CompensationParams] = None
    role_compensation_params: Optional[Dict[str, CompensationParams]] = None

    # Onboarding (Optional)
    onboarding: Optional[OnboardingConfig] = None

    # Plan Rules (Nested Model)
    plan_rules: PlanRules = Field(default_factory=PlanRules) # Use default if not specified

    @root_validator(pre=True) # Run before standard validation
    def check_role_dist_sums_to_one(cls, values):
        """Validate role distribution probabilities sum to 1.0 if provided."""
        role_dist = values.get('role_distribution')
        if role_dist and isinstance(role_dist, dict):
            prob_sum = sum(role_dist.values())
            if not np.isclose(prob_sum, 1.0):
                logger.warning(f"Global role_distribution probabilities sum to {prob_sum:.4f}, not 1.0. Check config.")
                # Normalization could happen here or in accessors.py
                # raise ValueError(f"Role distribution must sum to 1.0, got {prob_sum}")
        return values

    @root_validator
    def check_maintain_headcount_vs_growth(cls, values):
        """Ensure maintain_headcount and annual_growth_rate are used logically."""
        maintain = values.get('maintain_headcount')
        growth = values.get('annual_growth_rate')
        if maintain and growth != 0.0:
             logger.warning(f"maintain_headcount is True, but annual_growth_rate is {growth:.2%}. Growth rate will be ignored.")
             # Optionally force growth rate to 0 if maintain_headcount is True
             # values['annual_growth_rate'] = 0.0
        return values


class ScenarioDefinition(BaseModel):
    """Defines a single scenario, potentially overriding global parameters."""
    name: str # Scenario name is required
    # Allow any other keys from GlobalParameters to be overridden
    # We use Optional for all fields that *can* be overridden
    start_year: Optional[int] = None
    projection_years: Optional[int] = None
    random_seed: Optional[int] = None
    annual_compensation_increase_rate: Optional[float] = None
    annual_termination_rate: Optional[float] = None
    new_hire_termination_rate: Optional[float] = None
    use_expected_attrition: Optional[bool] = None
    new_hire_start_salary: Optional[float] = None
    new_hire_average_age: Optional[float] = None
    annual_growth_rate: Optional[float] = None
    monthly_transition: Optional[bool] = None
    maintain_headcount: Optional[bool] = None
    hazard_model_params: Optional[HazardModelParams] = None
    role_distribution: Optional[Dict[str, float]] = None
    new_hire_compensation_params: Optional[CompensationParams] = None
    role_compensation_params: Optional[Dict[str, CompensationParams]] = None
    onboarding: Optional[OnboardingConfig] = None
    plan_rules: Optional[PlanRules] = None # Allow overriding plan rules

    # Use Extra.allow to permit fields not explicitly defined (like custom scenario params)
    # Or use Extra.ignore to drop them silently
    # Or use Extra.forbid to raise errors on unknown fields
    class Config:
        extra = 'allow' # Or 'ignore' or 'forbid'


class MainConfig(BaseModel):
    """The root model for the entire configuration file."""
    global_parameters: GlobalParameters
    scenarios: Dict[str, ScenarioDefinition]

    @validator('scenarios')
    def check_baseline_scenario_exists(cls, scenarios):
        """Ensure at least a 'baseline' scenario is defined."""
        if 'baseline' not in scenarios:
            # Depending on requirements, could default or raise error
            logger.warning("No 'baseline' scenario found in configuration.")
            # raise ValueError("A 'baseline' scenario must be defined.")
        return scenarios

# Example of how to use these models after loading YAML data:
if __name__ == '__main__':
    # This block is for demonstration/testing purposes only
    import yaml
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s")

    # --- Example 1: Loading and validating config.yaml ---
    print("\n--- Validating config.yaml ---")
    try:
        # Construct path relative to this file's location
        config_path_main = Path(__file__).parent.parent.parent / 'configs' / 'config.yaml'
        print(f"Loading: {config_path_main}")
        with open(config_path_main, 'r') as f:
            raw_config_main = yaml.safe_load(f)

        validated_config_main = MainConfig(**raw_config_main)
        print("Validation Successful!")
        # Access validated data
        print(f"Global Start Year: {validated_config_main.global_parameters.start_year}")
        print(f"Baseline Eligibility Min Age: {validated_config_main.scenarios['baseline'].plan_rules.eligibility.min_age if validated_config_main.scenarios.get('baseline') and validated_config_main.scenarios['baseline'].plan_rules and validated_config_main.scenarios['baseline'].plan_rules.eligibility else 'N/A (Override?)'}")
        # Note: Accessing overridden rules requires merging logic (likely in accessors.py)
        # print(validated_config_main.json(indent=2)) # Print validated JSON

    except Exception as e:
        print(f"Validation FAILED for config.yaml: {e}")


    # --- Example 2: Loading and validating dev_tiny.yaml ---
    print("\n--- Validating dev_tiny.yaml ---")
    try:
        config_path_tiny = Path(__file__).parent.parent.parent / 'configs' / 'dev_tiny.yaml'
        print(f"Loading: {config_path_tiny}")
        with open(config_path_tiny, 'r') as f:
            raw_config_tiny = yaml.safe_load(f)

        validated_config_tiny = MainConfig(**raw_config_tiny)
        print("Validation Successful!")
        print(f"Global Start Year: {validated_config_tiny.global_parameters.start_year}")
        print(f"Baseline Scenario Name: {validated_config_tiny.scenarios['baseline'].name}")
        print(f"Baseline NEC Rate: {validated_config_tiny.global_parameters.plan_rules.employer_nec.rate}")


    except Exception as e:
        print(f"Validation FAILED for dev_tiny.yaml: {e}")

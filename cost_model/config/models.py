# cost_model/config/models.py
"""
Pydantic models for validating the structure and types of the configuration
loaded from YAML files (e.g., config.yaml).
"""

import logging
import numpy as np
from pydantic import BaseModel, Field, model_validator, validator
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# --- Low-level Reusable Models ---


class IRSYearLimits(BaseModel):
    """IRS limits for a specific year."""

    compensation_limit: int = Field(
        ..., description="Annual compensation limit (e.g., 401(a)(17))"
    )
    deferral_limit: int = Field(
        ..., description="Elective deferral limit (e.g., 402(g))"
    )
    catchup_limit: int = Field(
        ..., description="Catch-up contribution limit for age 50+"
    )
    catchup_eligibility_age: int = Field(
        50, description="Age at which catch-up contributions are allowed"
    )


class MatchTier(BaseModel):
    """Defines a single tier for employer matching contributions."""

    match_rate: float = Field(
        ..., ge=0.0, description="Employer match rate for this tier (e.g., 0.5 for 50%)"
    )
    cap_deferral_pct: float = Field(
        ...,
        ge=0.0,
        description="Maximum employee deferral percentage this tier applies to (e.g., 0.06 for 6%)",
    )
    # Optional: Add tenure requirements if needed later
    # min_tenure_months: Optional[int] = Field(None, ge=0)


class AutoEnrollOutcomeDistribution(BaseModel):
    """Probability distribution for auto-enrollment outcomes."""

    prob_opt_out: float = Field(..., ge=0.0, le=1.0)
    prob_stay_default: float = Field(..., ge=0.0, le=1.0)
    prob_opt_down: float = Field(..., ge=0.0, le=1.0)
    prob_increase_to_match: float = Field(..., ge=0.0, le=1.0)
    prob_increase_high: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode='after')
    def check_probabilities_sum_to_one(self) -> 'AutoEnrollOutcomeDistribution':
        """Validate that the probabilities sum approximately to 1.0."""
        prob_sum = (
            self.prob_opt_out +
            self.prob_stay_default +
            self.prob_opt_down +
            self.prob_increase_to_match +
            self.prob_increase_high
        )
        if not np.isclose(prob_sum, 1.0):
            raise ValueError(
                f"Probabilities must sum to 1.0, got {prob_sum:.4f}"
            )
        return self


# --- Plan Rules Models ---


class EligibilityRules(BaseModel):
    min_age: Optional[int] = Field(
        None, ge=0, description="Minimum age requirement (years)"
    )
    min_service_months: Optional[int] = Field(
        None, ge=0, description="Minimum service requirement (months)"
    )
    min_hours_worked: Optional[int] = Field(
        None, ge=0, description="Minimum hours worked requirement"
    )
    # Add other eligibility criteria like employment status, hours, etc. if needed


class OnboardingBumpRules(BaseModel):
    enabled: bool = False
    method: Optional[str] = Field(
        None, description="Method: 'flat_rate' or 'sample_plus_rate'"
    )
    rate: Optional[float] = Field(
        None, ge=0.0, description="Rate for bump (used by both methods)"
    )

    @model_validator(mode='after')
    def check_method_and_rate(self) -> 'OnboardingBumpRules':
        if self.enabled:
            if not self.method:
                raise ValueError("Onboarding bump method must be specified if enabled.")
            if self.rate is None:
                raise ValueError("Onboarding bump rate must be specified if enabled.")
            if self.method not in ["flat_rate", "sample_plus_rate"]:
                raise ValueError(f"Invalid onboarding bump method: {self.method}")
        return self


class AutoEnrollmentRules(BaseModel):
    enabled: bool = False
    window_days: Optional[int] = Field(
        90, description="Enrollment window in days after eligibility", ge=0
    )
    proactive_enrollment_probability: float = Field(0.0, ge=0.0, le=1.0)
    proactive_rate_range: Optional[Tuple[float, float]] = Field(
        None, description="Optional [min, max] rate range for proactive enrollment"
    )
    default_rate: float = Field(
        0.0,
        ge=0.0,
        description="Default deferral rate as percentage (e.g., 0.03 for 3%)",
    )
    opt_down_target_rate: Optional[float] = Field(None, ge=0.0)
    increase_to_match_rate: Optional[float] = Field(None, ge=0.0)
    increase_high_rate: Optional[float] = Field(None, ge=0.0)
    outcome_distribution: Optional[AutoEnrollOutcomeDistribution] = None
    re_enroll_existing: Optional[bool] = Field(
        False,
        description="Whether to re-enroll existing participants with 0% or previous opt-outs",
    )

    @model_validator(mode='after')
    def validate_auto_enrollment_rules(self) -> 'AutoEnrollmentRules':
        # Check proactive_rate_range
        if self.proactive_rate_range is not None:
            if not (isinstance(self.proactive_rate_range, (list, tuple)) and len(self.proactive_rate_range) == 2):
                raise ValueError("proactive_rate_range must be a list/tuple of two numbers")

            min_r, max_r = self.proactive_rate_range
            if not (isinstance(min_r, (int, float)) and isinstance(max_r, (int, float))):
                raise ValueError("proactive_rate_range values must be numbers")

            if min_r < 0 or max_r < 0:
                raise ValueError("proactive_rate_range values cannot be negative")

            if min_r > max_r:
                raise ValueError("proactive_rate_range min cannot be greater than max")

        # Check if outcome_distribution is provided when enabled
        if self.enabled and not self.outcome_distribution:
            raise ValueError("outcome_distribution must be defined if auto_enrollment is enabled.")

        return self


class AutoIncreaseRules(BaseModel):
    enabled: bool = False
    increase_rate: float = Field(0.0, ge=0.0)
    cap_rate: float = Field(0.0, ge=0.0)
    apply_to_new_hires_only: bool = False
    re_enroll_existing_below_cap: bool = False

    @model_validator(mode='after')
    def check_flags(self) -> 'AutoIncreaseRules':
        """Ensure mutually exclusive flags are not both true."""
        if self.apply_to_new_hires_only and self.re_enroll_existing_below_cap:
            raise ValueError(
                "Cannot set both 'apply_to_new_hires_only' and 're_enroll_existing_below_cap' to true."
            )
        return self


class EmployerMatchRules(BaseModel):
    tiers: List[MatchTier] = Field(default_factory=list)
    dollar_cap: Optional[float] = Field(
        None, ge=0.0, description="Optional annual dollar cap on total match"
    )


class EmployerNecRules(BaseModel):
    rate: float = Field(
        0.0,
        ge=0.0,
        description="Non-elective contribution rate as percentage (e.g., 0.03 for 3%)",
    )


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

    @model_validator(mode='after')
    def check_change_probs(self) -> 'BehavioralParams':
        """Validate that the conditional change probabilities sum approximately to 1.0 or less."""
        prob_sum = (
            self.prob_increase_given_change +
            self.prob_decrease_given_change +
            self.prob_stop_given_change
        )
        # Allow sum to be less than 1 (implies possibility of no change even if change event occurs)
        if prob_sum > 1.0001:  # Allow for slight float inaccuracies
            logger.warning(
                f"Behavioral change outcome probabilities sum to {prob_sum:.4f}, > 1.0. Check config."
            )
            # raise ValueError(f"Change outcome probabilities must sum to <= 1.0, got {prob_sum}")
        return self


class ContributionRules(BaseModel):
    """Placeholder if specific contribution rules beyond match/NEC are needed."""

    enabled: bool = True  # Often controls whether any contributions are calculated


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
    contributions: Optional[ContributionRules] = Field(
        default_factory=ContributionRules
    )  # Ensure default exists
    eligibility_events: Optional[Any] = None
    proactive_decrease: Optional[Any] = None  # Added for proactive decrease rules
    contribution_increase: Optional[Any] = None # Added for contribution increase rules



# --- Top-Level Configuration Models ---


class CompensationParams(BaseModel):
    """Generic compensation parameters."""

    comp_base_salary: float
    comp_std: Optional[float] = None  # Optional if using lognormal sigma
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

    file: str  # Path to the hazard model params file


class GlobalParameters(BaseModel):
    start_year: int
    projection_years: int
    random_seed: Optional[int] = None

    # HR Dynamics
    annual_compensation_increase_rate: float = Field(0.0, ge=0.0)
    annual_termination_rate: float = Field(0.0, ge=0.0, le=1.0)
    new_hire_termination_rate: float = Field(0.0, ge=0.0, le=1.0)
    use_expected_attrition: Optional[bool] = False  # Optional flag
    new_hire_start_salary: Optional[float] = Field(
        None, ge=0.0
    )  # Can be overridden by params
    new_hire_average_age: Optional[float] = Field(
        None, ge=0.0
    )  # Can be overridden by params
    annual_growth_rate: float = Field(0.0)  # Can be negative
    monthly_transition: Optional[bool] = False  # Optional flag
    maintain_headcount: bool = False  # Default to false if growth rate used
    new_hire_termination_rate_safety_margin: float = Field(
        0.0,
        ge=0.0,
        description="Additive safety margin for calculating number of new hires needed.",
    )

    # Hazard Model (Optional)
    hazard_model_params: Optional[HazardModelParams] = None

    # Role & Compensation Details
    role_distribution: Optional[Dict[str, float]] = None
    new_hire_compensation_params: Optional[CompensationParams] = None
    role_compensation_params: Optional[Dict[str, CompensationParams]] = None

    # Onboarding (Optional)
    onboarding: Optional[OnboardingConfig] = None

    # Promotion configuration
    dev_mode: bool = Field(False, description="Enable dev mode features like default matrices")
    promotion_matrix_path: Optional[str] = Field(None, description="Path to promotion matrix YAML file")

    # Plan Rules (Nested Model)
    plan_rules: PlanRules = Field(
        default_factory=PlanRules
    )  # Use default if not specified

    @model_validator(mode='after')
    def validate_global_parameters(self) -> 'GlobalParameters':
        """Run all validations for GlobalParameters."""
        self._check_role_dist_sums_to_one()
        self._check_maintain_headcount_vs_growth()
        return self

    def _check_role_dist_sums_to_one(self) -> None:
        """Validate role distribution probabilities sum to 1.0 if provided."""
        if self.role_distribution and isinstance(self.role_distribution, dict):
            prob_sum = sum(self.role_distribution.values())
            if not np.isclose(prob_sum, 1.0):
                logger.warning(
                    f"Global role_distribution probabilities sum to {prob_sum:.4f}, not 1.0. Check config."
                )
                # Normalization could happen here or in accessors.py
                # raise ValueError(f"Role distribution must sum to 1.0, got {prob_sum}")

    def _check_maintain_headcount_vs_growth(self) -> None:
        """Ensure maintain_headcount and annual_growth_rate are used logically."""
        if self.maintain_headcount and self.annual_growth_rate != 0.0:
            logger.warning(
                f"maintain_headcount is True, but annual_growth_rate is {self.annual_growth_rate:.2%}. Growth rate will be ignored."
            )
            # Optionally force growth rate to 0 if maintain_headcount is True
            # self.annual_growth_rate = 0.0


class ScenarioDefinition(BaseModel):
    """Defines a single scenario, potentially overriding global parameters."""

    name: str  # Scenario name is required
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
    dev_mode: Optional[bool] = None
    promotion_matrix_path: Optional[str] = None
    plan_rules: Optional[PlanRules] = None  # Allow overriding plan rules

    # Use Extra.allow to permit fields not explicitly defined (like custom scenario params)
    # Or use Extra.ignore to drop them silently
    # Or use Extra.forbid to raise errors on unknown fields
    class Config:
        extra = "allow"  # Or 'ignore' or 'forbid'


class MainConfig(BaseModel):
    """The root model for the entire configuration file."""

    global_parameters: GlobalParameters
    scenarios: Dict[str, ScenarioDefinition]

    @model_validator(mode='after')
    def check_baseline_scenario_exists(self) -> 'MainConfig':
        """Ensure at least a 'baseline' scenario is defined."""
        if "baseline" not in self.scenarios:
            # Depending on requirements, could default or raise error
            logger.warning("No 'baseline' scenario found in configuration.")
            # raise ValueError("A 'baseline' scenario must be defined.")
        return self


# Example of how to use these models after loading YAML data:
if __name__ == "__main__":
    # This block is for demonstration/testing purposes only
    import yaml
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s",
    )

    # --- Example 1: Loading and validating config.yaml ---
    print("\n--- Validating config.yaml ---")
    try:
        # Construct path relative to this file's location
        config_path_main = (
            Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        )
        print(f"Loading: {config_path_main}")
        with open(config_path_main, "r") as f:
            raw_config_main = yaml.safe_load(f)

        validated_config_main = MainConfig(**raw_config_main)
        print("Validation Successful!")
        # Access validated data
        print(
            f"Global Start Year: {validated_config_main.global_parameters.start_year}"
        )
        print(
            f"Baseline Eligibility Min Age: {validated_config_main.scenarios['baseline'].plan_rules.eligibility.min_age if validated_config_main.scenarios.get('baseline') and validated_config_main.scenarios['baseline'].plan_rules and validated_config_main.scenarios['baseline'].plan_rules.eligibility else 'N/A (Override?)'}"
        )
        # Note: Accessing overridden rules requires merging logic (likely in accessors.py)
        # print(validated_config_main.json(indent=2)) # Print validated JSON

    except Exception as e:
        print(f"Validation FAILED for config.yaml: {e}")

    # --- Example 2: Loading and validating dev_tiny.yaml ---
    print("\n--- Validating dev_tiny.yaml ---")
    try:
        config_path_tiny = (
            Path(__file__).parent.parent.parent / "configs" / "dev_tiny.yaml"
        )
        print(f"Loading: {config_path_tiny}")
        with open(config_path_tiny, "r") as f:
            raw_config_tiny = yaml.safe_load(f)

        validated_config_tiny = MainConfig(**raw_config_tiny)
        print("Validation Successful!")
        print(
            f"Global Start Year: {validated_config_tiny.global_parameters.start_year}"
        )
        print(
            f"Baseline Scenario Name: {validated_config_tiny.scenarios['baseline'].name}"
        )
        print(
            f"Baseline NEC Rate: {validated_config_tiny.global_parameters.plan_rules.employer_nec.rate}"
        )

    except Exception as e:
        print(f"Validation FAILED for dev_tiny.yaml: {e}")

"""
Configuration parameter parsing and validation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parse and validate the configuration dictionary into global parameters
    and plan rules.

    Args:
        config: Dictionary containing configuration parameters

    Returns:
        Tuple of (global_params, plan_rules)
    """
    # Extract global parameters with defaults
    global_section = config.get("global_parameters", {})

    # Initialize global parameters with values from global_parameters section
    global_params = SimpleNamespace(
        seed=global_section.get("random_seed", 42),
        start_year=global_section.get("start_year", datetime.now().year),
        num_years=global_section.get("projection_years", 5),
        target_growth=global_section.get("target_growth", 0.0),
        annual_growth_rate=global_section.get(
            "target_growth", 0.0
        ),  # Alias for backward compatibility
        new_hire_rate=global_section.get("attrition", {}).get("new_hire_termination_rate", 0.17),
        term_rate=global_section.get("attrition", {}).get("experienced_attrition_rate", 0.15),
        comp_raise_pct=global_section.get("annual_compensation_increase_rate", 0.03),
        cola_pct=global_section.get("compensation", {}).get("COLA_rate", 0.02),
        new_hire_termination_rate=global_section.get("attrition", {}).get(
            "new_hire_termination_rate", 0.25
        ),
        compensation=SimpleNamespace(
            cola_pct=global_section.get("compensation", {}).get("COLA_rate", 0.02),
            comp_raise_pct=global_section.get("annual_compensation_increase_rate", 0.03),
        ),
    )

    # Log the loaded parameters for debugging
    logger.debug(f"Loaded global parameters: {vars(global_params)}")

    # Extract plan rules configuration
    plan_rules = SimpleNamespace(
        eligibility=SimpleNamespace(min_age=21, min_service_months=0),
        onboarding_bump=SimpleNamespace(enabled=True, method="sample_plus_rate", rate=0.05),
        auto_enrollment=SimpleNamespace(
            enabled=False,
            window_days=90,
            proactive_enrollment_probability=0.0,
            proactive_rate_range=[0.0, 0.0],
            default_rate=0.0,
            re_enroll_existing=False,
            opt_down_target_rate=0.0,
            increase_to_match_rate=0.0,
            increase_high_rate=0.0,
            outcome_distribution=SimpleNamespace(
                prob_opt_out=0.0,
                prob_stay_default=1.0,
                prob_opt_down=0.0,
                prob_increase_to_match=0.0,
                prob_increase_high=0.0,
            ),
        ),
        auto_increase=SimpleNamespace(enabled=False, increase_rate=0.0, cap_rate=0.0),
        employer_match=SimpleNamespace(
            tiers=[SimpleNamespace(match_rate=0.0, cap_deferral_pct=0.0)], dollar_cap=0.0
        ),
        employer_nec=SimpleNamespace(rate=0.01),
        irs_limits=SimpleNamespace(
            _2024=SimpleNamespace(
                compensation_limit=345000,
                deferral_limit=23000,
                catchup_limit=7500,
                catchup_eligibility_age=50,
            )
        ),
        behavioral_params=SimpleNamespace(
            voluntary_enrollment_rate=0.2,
            voluntary_default_deferral=0.05,
            voluntary_window_days=180,
            voluntary_change_probability=0.1,
            prob_increase_given_change=0.4,
            prob_decrease_given_change=0.3,
            prob_stop_given_change=0.05,
            voluntary_increase_amount=0.01,
            voluntary_decrease_amount=0.01,
        ),
        contributions=SimpleNamespace(enabled=True),
        eligibility_events=SimpleNamespace(
            milestone_months=[], milestone_years=[], event_type_map={}
        ),
    )

    logger.info(f"Parsed global parameters: {global_params}")
    logger.info(f"Parsed plan rules: {plan_rules}")

    return global_params, plan_rules

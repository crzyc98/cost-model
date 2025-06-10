# This file contains the legacy full projection logic

import logging
import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from cost_model.dynamics.hiring import generate_new_hires

# Import constants
from cost_model.state.schema import (  # Imported from schema.py for single source of truth
    EMP_TERM_DATE,
    STATUS_COL,
)

from .constants import (  # Assuming these are needed implicitly
    ACTIVE_STATUS,
    INACTIVE_STATUS,
)

# Import necessary utilities using relative paths
from .date_utils import calculate_tenure

# Import shared helper functions from hr_projection
from .hr_projection import _apply_comp_bump, _apply_turnover, apply_onboarding_bump
from .ml.ml_utils import predict_turnover, try_load_ml_model
from .rules.auto_enrollment import apply as apply_auto_enrollment
from .rules.auto_increase import apply as apply_auto_increase
from .rules.contributions import apply as apply_contributions

# Import rule application functions (needed by legacy logic)
from .rules.eligibility import apply as apply_eligibility
from .rules.validators import (
    AutoEnrollmentRule,
    ContributionsRule,
    EligibilityRule,
    MatchRule,
    NonElectiveRule,
)
from .sampling.new_hires import sample_new_hire_compensation
from .sampling.salary import DefaultSalarySampler, SalarySampler

logger = logging.getLogger(__name__)


# Legacy: Full census projection (all steps in one pass, unchanged from original)
def project_census(
    start_df: pd.DataFrame,
    scenario_config: Mapping[str, Any],
    random_seed: Optional[int] = None,
) -> Dict[int, pd.DataFrame]:
    """Performs a full projection including HR changes and plan rule applications year by year."""
    projection_years = scenario_config["projection_years"]
    start_year = scenario_config["start_year"]
    # HR parameter fallbacks (top-level or under plan_rules)
    pr_cfg = scenario_config.get("plan_rules", {})
    comp_increase_rate = scenario_config.get("annual_compensation_increase_rate") or pr_cfg.get(
        "annual_compensation_increase_rate", 0.0
    )
    termination_rate = scenario_config.get("annual_termination_rate") or pr_cfg.get(
        "annual_termination_rate", 0.0
    )
    # headcount growth
    growth_rate = scenario_config.get("annual_growth_rate") or pr_cfg.get("annual_growth_rate", 0.0)
    maintain_hc = scenario_config.get("maintain_headcount", None)
    maintain_headcount = (
        maintain_hc if maintain_hc is not None else pr_cfg.get("maintain_headcount", True)
    )
    # new-hire termination rate
    nh_term_rate = scenario_config.get("new_hire_termination_rate") or pr_cfg.get(
        "new_hire_termination_rate", 0.0
    )
    # Initialize pluggable salary sampler
    sampler_cfg = scenario_config.get(
        "salary_sampler", DefaultSalarySampler()
    )  # Default if not specified
    if isinstance(sampler_cfg, dict):  # If config dict is provided for sampler
        # Assuming a way to instantiate from dict, placeholder for now
        # sampler_type = sampler_cfg.get('type', 'DefaultSalarySampler')
        # sampler_params = sampler_cfg.get('params', {})
        # sampler = dynamically_load_sampler(sampler_type)(**sampler_params)
        sampler = DefaultSalarySampler()  # Fallback for now
    elif isinstance(sampler_cfg, SalarySampler):
        sampler = sampler_cfg  # If an instance is passed
    else:
        sampler = DefaultSalarySampler()  # Default instance

    seed = random_seed if random_seed is not None else scenario_config.get("random_seed")
    # Master SeedSequence for dedicated streams
    master_ss = np.random.SeedSequence(seed)
    # Need all original 5 streams for legacy function
    bump_ss, term_ss, nh_ss, ml_ss, contrib_ss = master_ss.spawn(5)
    rng_bump = np.random.default_rng(bump_ss)
    rng_term = np.random.default_rng(term_ss)
    rng_nh = np.random.default_rng(nh_ss)
    rng_ml = np.random.default_rng(ml_ss)
    np.random.default_rng(contrib_ss)  # Used implicitly by rule applications

    # Load ML model and features if configured
    model_path = scenario_config.get("ml_model_path", "")
    features_path = scenario_config.get("model_features_path", "")
    if model_path and features_path:
        ml_pair = try_load_ml_model(model_path, features_path)
        projection_model, feature_cols = (None, []) if ml_pair is None else ml_pair
    else:
        projection_model, feature_cols = None, []
    use_ml = scenario_config.get("use_ml_turnover", False) and projection_model is not None

    # Prepare snapshots
    projected_data: Dict[int, pd.DataFrame] = {}
    # Compute and freeze baseline hire salary distribution
    baseline_hire_salaries = start_df.loc[
        start_df["employee_hire_date"].dt.year == start_year - 1,
        "employee_gross_compensation",
    ].dropna()
    if baseline_hire_salaries.empty:
        logger.warning(
            "No hires found in year %d to baseline salary. Using all employees.",
            start_year - 1,
        )
        baseline_hire_salaries = start_df["employee_gross_compensation"].dropna()
    if baseline_hire_salaries.empty:
        logger.error("Cannot create baseline hire salary distribution - empty compensation data.")
        # Decide how to handle this - raise error or use default? For now, log and maybe use default
        baseline_hire_salaries = np.array([50000])  # Arbitrary default

    current_df = start_df.copy()
    # Initialize prev_term_salaries based on the correct compensation column name
    comp_col = "employee_gross_compensation"
    prev_term_salaries = current_df[comp_col].dropna()
    base_count = len(start_df)

    logger.info(
        "Starting legacy projection for '%s' from %s",
        scenario_config["scenario_name"],
        start_year,
    )
    logger.info("Initial headcount: %d", len(current_df))

    for year_num in range(1, projection_years + 1):
        sim_year = start_year + year_num - 1
        start_date = pd.Timestamp(f"{sim_year}-01-01")
        end_date = pd.Timestamp(f"{sim_year}-12-31")

        logger.debug(f"--- Projecting Year {year_num} ({sim_year}) ---")

        # Year 1 Start: log before any changes
        if year_num == 1:
            init_hc = len(current_df)
            init_comp = current_df[comp_col].sum()
            logger.info(f"[Year 1 Start] headcount={init_hc}, total_gross_comp={init_comp:.2f}")

        # Reset Status based on term date from *previous* year's end state
        current_df[STATUS_COL] = np.where(
            current_df[EMP_TERM_DATE].isna() | (current_df[EMP_TERM_DATE] > start_date),
            ACTIVE_STATUS,
            INACTIVE_STATUS,
        )
        current_df = current_df[current_df[STATUS_COL] == ACTIVE_STATUS].copy()
        active_hc_start = len(current_df)
        logger.debug(f"Active headcount at start of {sim_year}: {active_hc_start}")

        # 1. Apply Compensation Bump
        current_df["tenure"] = calculate_tenure(current_df["employee_hire_date"], start_date)
        current_df = _apply_comp_bump(
            current_df,
            comp_col,
            scenario_config.get("second_year_compensation_dist", {}),
            comp_increase_rate,
            rng_bump,
            sampler,
        )
        logger.debug(f"After comp bump: headcount={len(current_df)}")

        # 2. Early Termination Sampling (Rule-based)
        current_df = _apply_turnover(
            current_df,
            "employee_hire_date",
            termination_rate,
            start_date,
            end_date,
            rng_term,
            sampler,
            prev_term_salaries.values,
        )
        # Update term salaries *after* this round
        prev_term_salaries = current_df.loc[
            current_df[EMP_TERM_DATE].between(start_date, end_date), comp_col
        ].dropna()
        logger.debug(f"After early term sampling: headcount={len(current_df)}")

        # 3. Filter out people terminated *before* the period starts (redundant if filtered above, but safe)
        current_df = current_df[
            current_df[EMP_TERM_DATE].isna() | (current_df[EMP_TERM_DATE] >= start_date)
        ].copy()
        survivors_pre_hire = len(
            current_df[current_df[EMP_TERM_DATE].isna()]
        )  # Active before hires
        logger.debug(f"Survivors before new hires: {survivors_pre_hire}")

        # 4. New Hire Generation & Compensation Sampling
        if maintain_headcount:
            needed = max(0, base_count - survivors_pre_hire)
        else:
            target = int(base_count * (1 + growth_rate) ** year_num)
            net_needed = max(0, target - survivors_pre_hire)
            if nh_term_rate < 1:
                needed = math.ceil(net_needed / (1 - nh_term_rate)) if net_needed > 0 else 0
            else:
                needed = net_needed

        hire_rate = scenario_config.get("hire_rate")
        if hire_rate is not None and not maintain_headcount:
            needed = int(base_count * hire_rate)

        if needed > 0:
            logger.debug(f"Generating {needed} new hires for {sim_year}")
            age_mean = (
                scenario_config.get("new_hire_average_age")
                or pr_cfg.get("new_hire_average_age")
                or 30
            )
            age_std = (
                scenario_config.get("new_hire_age_std_dev")
                or scenario_config.get("age_std_dev")
                or pr_cfg.get("age_std_dev", 5.0)
                or 5.0
            )
            min_age = scenario_config.get("min_working_age", 18)
            max_age = scenario_config.get("max_working_age", 65)

            nh_df = generate_new_hires(
                num_hires=needed,
                hire_year=sim_year,
                role_distribution=scenario_config.get("role_distribution"),
                role_compensation_params=scenario_config.get("role_compensation_params"),
                age_mean=age_mean,
                age_std_dev=age_std,
                min_working_age=min_age,
                max_working_age=max_age,
                scenario_config=scenario_config,
            )
            nh_df = sample_new_hire_compensation(
                nh_df, comp_col, baseline_hire_salaries.values, rng_nh
            )
            ob_cfg = pr_cfg.get("onboarding_bump", {})
            nh_df = apply_onboarding_bump(
                nh_df, comp_col, ob_cfg, baseline_hire_salaries.values, rng_nh
            )
            current_df = pd.concat([current_df, nh_df], ignore_index=True)
            logger.debug(f"After adding new hires: headcount={len(current_df)}")
        else:
            logger.debug(f"No new hires needed for {sim_year}")

        # 5. ML-Based or Rule-Based Turnover (Applied after new hires join)
        if use_ml:
            probs = predict_turnover(
                current_df, projection_model, feature_cols, random_state=rng_ml
            )
            term_rate_for_turnover = probs
            logger.debug("Applying ML-based turnover")
        else:
            term_rate_for_turnover = termination_rate
            logger.debug("Applying rule-based turnover rate: %.2f%%", termination_rate * 100)

        current_df = _apply_turnover(
            current_df,
            "employee_hire_date",
            term_rate_for_turnover,
            start_date,
            end_date,
            rng_term,
            sampler,
            prev_term_salaries.values,
        )
        # Update term salaries *after* this second pass
        newly_termed_salaries_post_hire = current_df.loc[
            current_df[EMP_TERM_DATE].between(start_date, end_date)
            & ~current_df[comp_col].isin(prev_term_salaries),  # Avoid duplicates
            comp_col,
        ].dropna()
        prev_term_salaries = pd.concat(
            [prev_term_salaries, newly_termed_salaries_post_hire]
        ).dropna()
        logger.debug(f"After second turnover pass: headcount={len(current_df)}")

        # Filter for snapshot: Keep only those active at year-end
        snapshot_df = current_df[
            current_df[EMP_TERM_DATE].isna() | (current_df[EMP_TERM_DATE] > end_date)
        ].copy()

        active_hc_pre_rules = len(snapshot_df)
        logger.debug(f"Active headcount before plan rules: {active_hc_pre_rules}")

        # --- 6. Apply Plan Rule Facades --- Apply rules ONLY to the year-end snapshot

        # ensure deferral_rate exists
        if "deferral_rate" not in snapshot_df and "pre_tax_deferral_percentage" in snapshot_df:
            snapshot_df["deferral_rate"] = snapshot_df["pre_tax_deferral_percentage"]
        elif "deferral_rate" not in snapshot_df:
            snapshot_df["deferral_rate"] = 0.0  # Initialize if completely missing

        # normalize participation/opt-out flags
        for col in ["is_participating", "ae_opted_out", "ai_opted_out"]:
            if col not in snapshot_df:
                snapshot_df[col] = False  # Initialize if missing
            snapshot_df[col] = snapshot_df[col].astype("boolean").fillna(False)

        # Eligibility
        elig_config = pr_cfg.get("eligibility", {})
        if elig_config:  # Check if config exists
            elig_rules = EligibilityRule(**elig_config)
            snapshot_df = apply_eligibility(snapshot_df, elig_rules, end_date)

        # Auto-enrollment
        ae_config = pr_cfg.get("auto_enrollment", {})
        if ae_config and ae_config.get("enabled", False):
            ae_rules = AutoEnrollmentRule(**ae_config)
            snapshot_df = apply_auto_enrollment(snapshot_df, ae_rules, start_date, end_date)

        # Auto-increase
        ai_config = pr_cfg.get("auto_increase", {})
        if ai_config and ai_config.get("enabled", False):
            snapshot_df = apply_auto_increase(snapshot_df, pr_cfg, sim_year)

        # Contributions
        contrib_config = pr_cfg.get("contributions", {})
        if contrib_config and contrib_config.get("enabled", False):
            contrib_rules = ContributionsRule(**contrib_config)
            match_config = pr_cfg.get("employer_match", {})
            nec_config = pr_cfg.get("employer_nec", {})
            irs_limits = pr_cfg.get("irs_limits", {})
            if match_config and nec_config and irs_limits:
                match_rules = MatchRule(**match_config)
                nec_rules = NonElectiveRule(**nec_config)
                snapshot_df = apply_contributions(
                    snapshot_df,
                    contrib_rules,
                    match_rules,
                    nec_rules,
                    irs_limits,
                    sim_year,
                    start_date,
                    end_date,
                )
            else:
                logger.warning(
                    f"[{sim_year}] Contributions enabled but missing match/NEC/IRS limits config. Skipping."
                )

        final_hc = len(snapshot_df)
        final_comp = snapshot_df[comp_col].sum()
        logger.info(
            f"[Year {year_num} End] headcount={final_hc}, total_gross_comp={final_comp:.2f}"
        )

        # Store the final snapshot for the year
        projected_data[year_num] = snapshot_df.copy()

        # Prepare for the next year: Carry forward the state *before* plan rules were applied
        # but *after* all HR actions (comp bump, terms, hires, terms)
        # This was `current_df` before the final filtering step for the snapshot
        current_df = (
            current_df.copy()
        )  # Start next year with the full roster including those termed mid-year

    logger.info("Finished legacy projection for %d years.", projection_years)
    return projected_data

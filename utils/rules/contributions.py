import pandas as pd
import numpy as np
import re
import logging
from utils.date_utils import calculate_age
from utils.rules.formula_parsers import parse_match_formula, parse_match_tiers

logger = logging.getLogger(__name__)

def apply(df, scenario_config, simulation_year, year_start_date, year_end_date):
    """
    Calculates employee and employer contributions for the simulation year.
    Extracted from utils.plan_rules.calculate_contributions.
    """
    logger.info(f"Calculating contributions for {simulation_year}...")
    plan_rules = scenario_config.get('plan_rules', {})
    irs_limits = scenario_config.get('irs_limits', {})
    year_limits = irs_limits.get(simulation_year, {})
    logger.debug(f"IRS Limits for {simulation_year}: {year_limits}")

    statutory_comp_limit = year_limits.get('comp_limit', 345000)
    deferral_limit = year_limits.get('deferral_limit', 23000)
    catch_up_limit = year_limits.get('catch_up', 7500)
    overall_limit = year_limits.get('overall_limit', 69000)
    catch_up_age = plan_rules.get('catch_up_age', 50)

    employer_match_formula = plan_rules.get('employer_match_formula', "")
    employer_non_elective_formula = plan_rules.get('employer_non_elective_formula', "")

    required_input_cols = ['gross_compensation', 'deferral_rate', 'birth_date', 'status', 'hire_date']
    missing_cols = [col for col in required_input_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Required columns missing: {', '.join(missing_cols)}. Cannot calculate contributions.")
        return df

    output_cols = {
        'plan_year_compensation': 0.0,
        'capped_compensation': 0.0,
        'pre_tax_contributions': 0.0,
        'employer_match_contribution': 0.0,
        'employer_non_elective_contribution': 0.0,
        'total_contributions': 0.0,
        'is_catch_up_eligible': False,
        'effective_deferral_limit': 0.0
    }
    for col, default in output_cols.items():
        if col not in df.columns:
            df[col] = default
        if 'contribution' in col or 'compensation' in col or 'limit' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        elif col == 'is_catch_up_eligible':
            df[col] = df[col].fillna(False).astype(bool)

    df['deferral_rate'] = pd.to_numeric(df['deferral_rate'], errors='coerce').fillna(0.0)

    active_mask = df['status'].isin(['Active', 'Unknown'])
    if 'is_eligible' not in df.columns:
        df['is_eligible'] = False
        eligible_mask = active_mask
    else:
        df['is_eligible'] = df['is_eligible'].fillna(False).astype(bool)
        eligible_mask = df['is_eligible']

    if 'deferral_rate' in df.columns and 'is_participating' in df.columns:
        df.loc[eligible_mask & (df['deferral_rate'] > 0), 'is_participating'] = True

    calc_mask = active_mask & eligible_mask
    zero_cols = [
        'plan_year_compensation', 'capped_compensation', 'pre_tax_contributions',
        'employer_match_contribution', 'employer_non_elective_contribution', 'total_contributions'
    ]
    df.loc[~calc_mask, zero_cols] = 0.0

    if calc_mask.any():
        df.loc[calc_mask, 'plan_year_compensation'] = df.loc[calc_mask, 'gross_compensation'].fillna(0)
        df.loc[calc_mask, 'capped_compensation'] = np.minimum(
            df.loc[calc_mask, 'plan_year_compensation'], statutory_comp_limit)

        df.loc[calc_mask, 'current_age'] = calculate_age(
            df.loc[calc_mask, 'birth_date'], year_end_date)
        df['is_catch_up_eligible'] = False
        df.loc[calc_mask, 'is_catch_up_eligible'] = (
            df.loc[calc_mask, 'current_age'] >= catch_up_age)

        df['effective_deferral_limit'] = deferral_limit
        df.loc[calc_mask & df['is_catch_up_eligible'], 'effective_deferral_limit'] = (
            deferral_limit + catch_up_limit)

        potential_deferral = (
            df.loc[calc_mask, 'capped_compensation'] *
            df.loc[calc_mask, 'deferral_rate']
        )
        df.loc[calc_mask, 'pre_tax_contributions'] = np.minimum(
            potential_deferral,
            df.loc[calc_mask, 'effective_deferral_limit']
        )

        nec_rate = 0.0
        if employer_non_elective_formula:
            match = re.match(r"\s*(\d+\.?\d*)\s*%\s*", employer_non_elective_formula, re.IGNORECASE)
            if match:
                nec_rate = float(match.group(1)) / 100.0
            else:
                logger.warning(
                    f"Could not parse non-elective formula: '{employer_non_elective_formula}'. Assuming 0%."
                )
        df.loc[calc_mask, 'employer_non_elective_contribution'] = (
            df.loc[calc_mask, 'capped_compensation'] * nec_rate
        )

        if plan_rules.get('last_day_work_rule', False):
            term_mask = df['status'] == 'Terminated'
            df.loc[
                term_mask,
                ['employer_match_contribution', 'employer_non_elective_contribution']
            ] = 0.0
            logger.info(
                f"Last day work rule: zeroed match/NEC for {term_mask.sum()} terminated employees."
            )

        df.loc[calc_mask, 'employer_match_contribution'] = 0.0
        if employer_match_formula:
            tiers = parse_match_tiers(employer_match_formula)
            if tiers:
                eligible_deferral_rate = df.loc[calc_mask, 'deferral_rate']
                eligible_capped_comp = df.loc[calc_mask, 'capped_compensation']
                calculated_match = pd.Series(
                    0.0,
                    index=df[calc_mask].index
                )
                last_deferral_cap = 0.0
                for tier in tiers:
                    current_cap_pct = tier['deferral_cap_pct']
                    tier_match_rate = tier['match_pct']
                    def_in_range = np.minimum(
                        eligible_deferral_rate, current_cap_pct
                    ) - last_deferral_cap
                    def_in_range = np.maximum(def_in_range, 0)
                    match_for_tier = (
                        def_in_range * eligible_capped_comp * tier_match_rate
                    )
                    calculated_match += match_for_tier
                    last_deferral_cap = current_cap_pct
                df.loc[calc_mask, 'employer_match_contribution'] = calculated_match

        total_calc = (
            df.loc[calc_mask, 'pre_tax_contributions'] +
            df.loc[calc_mask, 'employer_match_contribution'] +
            df.loc[calc_mask, 'employer_non_elective_contribution']
        )

        exceeds = total_calc > overall_limit
        if exceeds.any():
            logger.warning(
                f"{exceeds.sum()} employees exceeded overall contribution limit of ${overall_limit}. Reducing employer contributions."
            )
            excess_amount = total_calc[exceeds] - overall_limit
            er_total = (
                df.loc[exceeds, 'employer_match_contribution'] +
                df.loc[exceeds, 'employer_non_elective_contribution']
            )
            reduction = pd.Series(1.0, index=excess_amount.index)
            nonzero = er_total > 0
            reduction[nonzero] = 1.0 - (
                excess_amount[nonzero] / er_total[nonzero]
            )
            reduction = np.maximum(reduction, 0)
            df.loc[exceeds, 'employer_match_contribution'] *= reduction
            df.loc[exceeds, 'employer_non_elective_contribution'] *= reduction
            total_calc[exceeds] = (
                df.loc[exceeds, 'pre_tax_contributions'] +
                df.loc[exceeds, 'employer_match_contribution'] +
                df.loc[exceeds, 'employer_non_elective_contribution']
            )

        df.loc[calc_mask, 'total_contributions'] = total_calc

    if 'current_age' in df.columns:
        df.drop(columns=['current_age'], inplace=True, errors='ignore')

    return df

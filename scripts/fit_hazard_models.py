#!/usr/bin/env python3
"""
Script to fit Kaplan–Meier and Cox hazard models on historical turnover data.

Usage:
    python scripts/fit_hazard_models.py --historical data/historical_turnover.csv \
        --output data/hazard_model_params.yaml
"""
import argparse
import pandas as pd
import yaml
from lifelines import KaplanMeierFitter, CoxPHFitter


def main(historical_path: str, output_path: str):
    # Load historical turnover data
    df = pd.read_csv(
        historical_path,
        parse_dates=["hire_date", "termination_date", "birth_date"]
    )
    # Event observed: separated (1) or censored (0)
    df['event_observed'] = df['termination_date'].notna().astype(int)

    # Use max separation date as censoring cutoff for active employees
    cutoff_date = df['termination_date'].max()
    df['termination_date_filled'] = df['termination_date'].fillna(cutoff_date)
    # Compute duration in years
    df['duration'] = (
        df['termination_date_filled'] - df['hire_date']
    ).dt.days / 365.25

    # Kaplan–Meier estimation
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df['duration'], event_observed=df['event_observed'])
    print("Kaplan–Meier median survival time (years):", kmf.median_survival_time_)
    kmf.survival_function_.to_csv('data/km_survival_function.csv')

    # Prepare covariates for Cox model
    df['age_at_hire'] = (
        df['hire_date'] - df['birth_date']
    ).dt.days / 365.25
    cox_df = df[['duration', 'event_observed', 'age_at_hire', 'gross_compensation']].dropna()

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='duration', event_col='event_observed')
    print("\nCoxPH model summary:")
    print(cph.summary)
    cph.baseline_hazard_.to_csv('data/cox_baseline_hazard.csv')

    # Export estimated parameters to YAML
    params = {
        'kaplan_meier_median_years': float(kmf.median_survival_time_),
        'cox_coefficients': cph.params_.to_dict()
    }
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.safe_dump(params, f)
    print(f"Hazard model parameters saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit survival/hazard models on turnover data"
    )
    parser.add_argument(
        '--historical', type=str, default='data/historical_turnover.csv',
        help='Path to historical turnover CSV'
    )
    parser.add_argument(
        '--output', type=str, default='data/hazard_model_params.yaml',
        help='Path to output YAML file for hazard params'
    )
    args = parser.parse_args()
    main(args.historical, args.output)

#!/usr/bin/env python3
"""
Fit Kaplan–Meier and Cox survival models on historical turnover data.
Usage:
    fit_hazard_models.py --historical PATH --output PATH [options]
"""
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
import yaml
from lifelines import KaplanMeierFitter, CoxPHFitter
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(
    historical: Path,
    output: Path,
    covariates: List[str],
    censor_date: Optional[pd.Timestamp],
    km_out: Path,
    km_ci_out: Path,
    baseline_out: Path
):
    """Run KM and Cox models, save CSVs and YAML parameters."""
    df = pd.read_csv(
        historical,
        parse_dates=["hire_date", "termination_date", "birth_date"]
    )
    # Drop bad rows
    df = df[df["hire_date"] <= df["termination_date"].fillna(pd.Timestamp.today())]
    # Event observed
    df['event_observed'] = df['termination_date'].notna().astype(int)
    # Censor date
    if censor_date is None:
        censor = df['termination_date'].max() or pd.Timestamp.today()
    else:
        censor = censor_date
    df['termination_date_filled'] = df['termination_date'].fillna(censor)
    # Duration in years, non-negative
    df['duration'] = (
        df['termination_date_filled'] - df['hire_date']
    ).dt.days.clip(lower=0) / 365.25

    # Kaplan–Meier
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df['duration'], event_observed=df['event_observed'])
    km_df = kmf.survival_function_.reset_index().rename(columns={'timeline':'duration'})
    km_out.parent.mkdir(parents=True, exist_ok=True)
    km_df.to_csv(km_out, index=False)
    ci_df = kmf.confidence_interval_.reset_index()
    km_ci_out.parent.mkdir(parents=True, exist_ok=True)
    ci_df.to_csv(km_ci_out, index=False)
    logger.info("KM median survival time: %.2f years", kmf.median_survival_time_)

    # Cox PH
    for cov in covariates:
        if cov not in df.columns:
            logger.error("Missing covariate: %s", cov)
            return
    cox_df = df[['duration','event_observed'] + covariates].dropna()
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='duration', event_col='event_observed')
    baseline_out.parent.mkdir(parents=True, exist_ok=True)
    cph.baseline_hazard_.to_csv(baseline_out)
    logger.info("Cox coefficients:\n%s", cph.params_)

    # Write parameters
    params = {
        'kaplan_meier': {
            'median_years': float(kmf.median_survival_time_),
            'ci_lower': float(ci_df.iloc[0,1]),
            'ci_upper': float(ci_df.iloc[0,2]) if ci_df.shape[1]>2 else None,
            'survival_function': km_df.to_dict(orient='records')
        },
        'cox': {
            'coefficients': cph.params_.to_dict(),
            'baseline_hazard': cph.baseline_hazard_.to_dict()
        }
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        yaml.safe_dump(params, f)
    logger.info("Saved hazard parameters to %s", output)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Fit survival/hazard models on turnover data")
    p.add_argument('--historical', type=Path, required=True,
                   help='CSV of historical turnover data')
    p.add_argument('--output', type=Path, default=Path('data/hazard_model_params.yaml'),
                   help='YAML output path')
    p.add_argument('--covariates', type=lambda s: s.split(','),
                   default=["age_at_hire","gross_compensation"],
                   help='Comma-separated covariate names')
    p.add_argument('--censor-date', type=lambda s: pd.to_datetime(s), default=None,
                   help='Override censor date (YYYY-MM-DD)')
    p.add_argument('--km-out', type=Path, default=Path('data/km_survival_function.csv'),
                   help='Path for KM survival CSV')
    p.add_argument('--km-ci-out', type=Path, default=Path('data/km_confidence_intervals.csv'),
                   help='Path for KM confidence intervals CSV')
    p.add_argument('--baseline-out', type=Path, default=Path('data/cox_baseline_hazard.csv'),
                   help='Path for Cox baseline hazard CSV')
    args = p.parse_args()
    main(
        args.historical,
        args.output,
        args.covariates,
        args.censor_date,
        args.km_out,
        args.km_ci_out,
        args.baseline_out
    )

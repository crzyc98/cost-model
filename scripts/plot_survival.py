#!/usr/bin/env python3
"""
Standalone script to plot Kaplan–Meier survival curves by tenure cohort.
"""
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import argparse

# ── Setup ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def assign_cohort(duration_years):
    if duration_years <= 1:    return '0-1yr'
    if duration_years <= 3:    return '1-3yr'
    return '3+yr'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',   type=Path, default=Path(__file__).resolve().parent.parent / 'data' / 'historical_turnover.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent.parent / 'output' / 'survival_curves.png')
    parser.add_argument('--as-of',  type=str, help="Censoring date (YYYY-MM-DD), default=today")
    parser.add_argument('--show',   action='store_true', help="Display plot interactively")
    args = parser.parse_args()

    data_path = args.data
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return

    censor_date = pd.to_datetime(args.as_of) if args.as_of else pd.Timestamp.today()

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path, parse_dates=['hire_date','termination_date'])

    # ── Compute durations ────────────────────────────────────────────────────────
    df['duration'] = (df['termination_date'].fillna(censor_date) - df['hire_date']).dt.days / 365.25
    df['event_observed'] = df['termination_date'].notna().astype(int)
    df['cohort'] = df['duration'].apply(assign_cohort)

    # ── Plot ────────────────────────────────────────────────────────────────────
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8,6))
    for cohort, group in df.groupby('cohort'):
        kmf.fit(group['duration'], group['event_observed'], label=cohort)
        kmf.plot_survival_function(ci_show=False)

    plt.title('Survival Curves by Tenure Cohort')
    plt.xlabel('Years Since Hire')
    plt.ylabel('Survival Probability')
    plt.tight_layout()

    out_fig = args.output
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig)
    logger.info("Saved survival curves to %s", out_fig)

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Visualize ABM workforce dynamics:
  - Net growth decomposition
  - Cohort counts by tenure
  - Hires vs. terminations
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    logger.info("Saved %s", path)
    plt.close(fig)


def plot_net_growth(df: pd.DataFrame, outdir: Path):
    df = df.copy()
    df["Hires"] = df["New Hire Active"] + df["New Hire Terminated"]
    df["TotalActive"] = df["Continuous Active"] + df["New Hire Active"]
    df["Delta"] = df["TotalActive"].diff().fillna(0).astype(int)
    decomp = df.set_index("Year")[
        ["Delta", "Experienced Terminated", "New Hire Terminated", "Hires"]
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    decomp.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Net Growth Decomposition")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, outdir / "net_growth_decomposition.png")


def plot_cohort_counts(agent_df: pd.DataFrame, outdir: Path):
    cohort_counts = agent_df.groupby(["Year", "Cohort"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    cohort_counts.plot(ax=ax)
    ax.set_title("Cohort Counts by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Agents")
    ax.legend(title="Cohort")
    _save(fig, outdir / "cohort_counts.png")


def plot_hire_termination(df: pd.DataFrame, outdir: Path):
    hires = df["New Hire Active"] + df["New Hire Terminated"]
    dynamics = pd.DataFrame(
        {
            "Hires": hires,
            "Experienced Terminations": df["Experienced Terminated"],
            "New Hire Terminations": df["New Hire Terminated"],
        },
        index=pd.to_datetime(df["Year"], format="%Y"),
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    dynamics.plot(marker="o", ax=ax)
    ax.set_title("Hire vs. Termination Dynamics")
    ax.set_ylabel("Count")
    _save(fig, outdir / "hire_termination_dynamics.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_csv", type=Path, required=True)
    p.add_argument("--agent_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("output"))
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_df = pd.read_csv(args.model_csv)
    agent_df = pd.read_csv(args.agent_csv)

    plot_net_growth(model_df, args.output_dir)
    plot_cohort_counts(agent_df, args.output_dir)
    plot_hire_termination(model_df, args.output_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

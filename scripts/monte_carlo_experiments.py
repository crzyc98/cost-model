#!/usr/bin/env python3
"""
Monte Carlo simulation experiments for the retirement plan ABM.
"""
import argparse
import sys
import yaml
import subprocess
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def sample_config(base_cfg, rng):
    cfg = base_cfg.copy()
    # experienced attrition ±10%
    ex = base_cfg.get("annual_termination_rate", 0.13)
    cfg["annual_termination_rate"] = float(rng.uniform(ex * 0.9, ex * 1.1))
    # new-hire attrition tied to experienced
    cfg["new_hire_termination_rate"] = cfg["annual_termination_rate"] * 1.6
    # growth ±10%
    g = base_cfg.get("annual_growth_rate", 0.02)
    cfg["annual_growth_rate"] = float(rng.uniform(g * 0.9, g * 1.1))
    # salary growth truncated normal
    sal = rng.normal(
        loc=base_cfg.get("annual_compensation_increase_rate", 0.03),
        scale=base_cfg.get("salary_growth_std", 0.03 * 0.02),
    )
    lower, upper = sal - 2 * base_cfg.get(
        "salary_growth_std", 0.0
    ), sal + 2 * base_cfg.get("salary_growth_std", 0.0)
    cfg["annual_compensation_increase_rate"] = float(
        np.clip(sal, lower, upper, out=np.empty_like(sal))
    )
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--census", required=True)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument(
        "--seed", type=int, default=None, help="RNG seed for reproducibility"
    )
    p.add_argument("--output_dir", default="output/monte_carlo")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    base_cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    python = sys.executable

    for i in range(1, args.runs + 1):
        cfg_i = sample_config(base_cfg, rng)
        cfg_path = out_dir / f"config_run_{i}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_i))

        prefix = out_dir / f"run_{i}"
        cmd = [
            python,
            "scripts/run_retirement_plan_abm.py",
            "--config",
            str(cfg_path),
            "--census",
            args.census,
            "--output",
            str(prefix),
        ]
        logger.info("Starting run %d: cmd=%s", i, " ".join(cmd))
        subprocess.run(cmd, check=True)

        model_file = out_dir / f"run_{i}_model_results.csv"
        if not model_file.exists():
            logger.error("Missing ABM output %s", model_file)
            continue

        df = pd.read_csv(model_file)
        totals = df["Continuous Active"] + df["New Hire Active"]
        final_total = int(totals.iloc[-1])
        growth = int(final_total - totals.iloc[-2]) if len(totals) > 1 else 0

        summary.append(
            {
                "run": i,
                "annual_term": cfg_i["annual_termination_rate"],
                "new_hire_term": cfg_i["new_hire_termination_rate"],
                "growth_rate": cfg_i["annual_growth_rate"],
                "sal_growth": cfg_i["annual_compensation_increase_rate"],
                "final_total_active": final_total,
                "growth": growth,
                "plan_cost": (
                    df.get("PlanCost", pd.Series()).iloc[-1]
                    if "PlanCost" in df
                    else None
                ),
                "avg_deferral_pct": (
                    df.get("AvgDeferralPct", pd.Series()).iloc[-1]
                    if "AvgDeferralPct" in df
                    else None
                ),
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_csv = out_dir / "monte_carlo_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("✔ Saved summary to %s", summary_csv)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Thin wrapper to run the two-phase pipeline:
 1. HR snapshots
 2. Plan rules application
"""
import subprocess
import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run two-phase pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--census", default="data/census_data.csv")
    parser.add_argument("--hr-dir", default="hr_snapshots/")
    parser.add_argument("--plan-dir", default="plan_outputs/")
    args = parser.parse_args()

    os.makedirs(args.hr_dir, exist_ok=True)
    os.makedirs(args.plan_dir, exist_ok=True)

    cmds = [
        [
            sys.executable,
            "scripts/run_hr_snapshots.py",
            "--config",
            args.config,
            "--census",
            args.census,
            "--output",
            args.hr_dir,
        ],
        [
            sys.executable,
            "scripts/run_plan_rules.py",
            "--config",
            args.config,
            "--snapshots",
            args.hr_dir,
            "--outdir",
            args.plan_dir,
        ],
    ]
    for cmd in cmds:
        print("Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Command failed (exit code {e.returncode}): {' '.join(cmd)}")
            if e.stderr:
                print(e.stderr.decode())
            sys.exit(e.returncode)
    print("All done.")
    sys.exit(0)


if __name__ == "__main__":
    main()

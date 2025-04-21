"""
Command-line utility to run Monte Carlo comparisons across multiple scenarios.
"""
import os
import argparse
import csv

from .scenario_loader import load as load_scenarios
from .sampler import ParameterSampler
from .mc import MonteCarloRunner

__all__ = ['main']

def main():
    parser = argparse.ArgumentParser(
        description='Run Monte Carlo comparison across scenarios'
    )
    parser.add_argument(
        '--scenario-dir', required=True,
        help='Directory containing scenario YAML files'
    )
    parser.add_argument(
        '--mc-package', required=True,
        help='MC package name or path for parameter sampling'
    )
    parser.add_argument(
        '--runs', type=int, default=1000,
        help='Number of Monte Carlo runs'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output directory for CSV results'
    )
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_dir)
    sampler = ParameterSampler(args.mc_package)
    runner = MonteCarloRunner(
        scenarios, sampler, args.runs, rng_seed=42
    )
    results = runner.run()

    os.makedirs(args.output, exist_ok=True)
    for scenario_name, kpis in results.items():
        output_path = os.path.join(
            args.output, f'{scenario_name}.csv'
        )
        if kpis:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(kpis[0].keys())
                )
                writer.writeheader()
                writer.writerows(kpis)
    print(f'Results saved to {args.output}')

if __name__ == '__main__':
    main()

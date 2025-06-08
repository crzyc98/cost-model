#!/usr/bin/env python3
"""
Fix the simulation_analysis.ipynb notebook to handle missing age_hist and tenure_hist data.
"""

import json
import re

def fix_notebook():
    # Read the notebook
    with open('notebooks/simulation_analysis.ipynb', 'r') as f:
        notebook = json.load(f)

    # Find the cell with the problematic code
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source_lines = cell['source']

            # Fix age distribution analysis
            for i, line in enumerate(source_lines):
                if 'col_name in comparison_df.columns:' in line and 'age_hist' in source_lines[i-1]:
                    print(f"Found age distribution analysis at line {i}")
                    # Add fallback for missing columns
                    source_lines[i+1] = source_lines[i+1].replace(
                        "            age_analysis[f'{band} (%)'] = (comparison_df[col_name] * 100).round(1)\n",
                        "            age_analysis[f'{band} (%)'] = (comparison_df[col_name] * 100).round(1)\n"
                    )
                    source_lines[i+2] = source_lines[i+2].replace(
                        "            age_variance_analysis[f'{band} Var'] = ((comparison_df[col_name] * 100) - target).round(1)\n",
                        "            age_variance_analysis[f'{band} Var'] = ((comparison_df[col_name] * 100) - target).round(1)\n"
                    )
                    # Add else clause for missing columns
                    source_lines.insert(i+3, "        else:\n")
                    source_lines.insert(i+4, "            age_analysis[f'{band} (%)'] = 'N/A'\n")
                    source_lines.insert(i+5, "            age_variance_analysis[f'{band} Var'] = 'N/A'\n")
                    break

            # Fix tenure distribution analysis
            for i, line in enumerate(source_lines):
                if 'col_name in comparison_df.columns:' in line and 'tenure_hist' in source_lines[i-1]:
                    print(f"Found tenure distribution analysis at line {i}")
                    # Add else clause for missing columns
                    source_lines.insert(i+3, "        else:\n")
                    source_lines.insert(i+4, "            tenure_analysis[f'{band} (%)'] = 'N/A'\n")
                    source_lines.insert(i+5, "            tenure_variance_analysis[f'{band} Var'] = 'N/A'\n")
                    break

    # Write the fixed notebook
    with open('notebooks/simulation_analysis.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)

    print("✅ Fixed the notebook!")

if __name__ == "__main__":
    fix_notebook()

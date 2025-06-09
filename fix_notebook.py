#!/usr/bin/env python3
"""
Fix the simulation_analysis.ipynb notebook to handle missing age_hist and tenure_hist data.
"""

import json

def fix_notebook():
    # Read the notebook
    with open('notebooks/simulation_analysis.ipynb', 'r') as f:
        notebook = json.load(f)

    # Find the cell with the problematic code
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source_lines = cell['source']

            # Fix the DataFrame initialization issue
            for i, line in enumerate(source_lines):
                if 'score_components = pd.DataFrame(index=' in line:
                    print(f"Found score_components DataFrame initialization at line {i}")
                    # Replace with proper column definition
                    source_lines[i] = "    score_components = pd.DataFrame(\n"
                    source_lines.insert(i+1, "        columns=['Age Error', 'Tenure Error', 'HC Growth Error', 'Pay Growth Error', 'Total Score'],\n")
                    source_lines.insert(i+2, "        index=[f\"Config {i+1}\" for i in range(len(comparison_df))]\n")
                    source_lines.insert(i+3, "    )\n")
                    break

            # Remove the redundant column assignment
            for i, line in enumerate(source_lines):
                if "score_components.columns = ['Age Error'" in line:
                    print(f"Removing redundant column assignment at line {i}")
                    # Remove this line
                    source_lines.pop(i)
                    break

            # Fix age distribution analysis
            for i, line in enumerate(source_lines):
                if 'col_name in comparison_df.columns:' in line and i > 0 and 'age_hist' in source_lines[i-1]:
                    print(f"Found age distribution analysis at line {i}")
                    # Add else clause for missing columns
                    if i+3 < len(source_lines) and "        else:" not in source_lines[i+3]:
                        source_lines.insert(i+3, "        else:\n")
                        source_lines.insert(i+4, "            age_analysis[f'{band} (%)'] = 'N/A'\n")
                        source_lines.insert(i+5, "            age_variance_analysis[f'{band} Var'] = 'N/A'\n")
                    break

            # Fix tenure distribution analysis
            for i, line in enumerate(source_lines):
                if 'col_name in comparison_df.columns:' in line and i > 0 and 'tenure_hist' in source_lines[i-1]:
                    print(f"Found tenure distribution analysis at line {i}")
                    # Add else clause for missing columns
                    if i+3 < len(source_lines) and "        else:" not in source_lines[i+3]:
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

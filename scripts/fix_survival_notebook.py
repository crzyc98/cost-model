#!/usr/bin/env python3
"""
Patch notebooks/survival_plots.ipynb to add a leading code cell for cwd and ensure each code cell has outputs and execution_count.
"""
import nbformat
from nbformat.v4 import new_code_cell
import os

project_root = os.path.dirname(os.path.dirname(__file__))
nb_path = os.path.join(project_root, 'notebooks', 'survival_plots.ipynb')

# Read notebook
nb = nbformat.read(nb_path, as_version=4)
# Ensure each code cell has required fields
for cell in nb.cells:
    cell.pop('id', None)
    if cell.cell_type == 'code':
        cell.setdefault('outputs', [])
        cell.setdefault('execution_count', None)
        # patch CSV path to point up one level
        if isinstance(cell.get('source'), list):
            cell['source'] = [line.replace('data/historical_turnover.csv', '../data/historical_turnover.csv') for line in cell['source']]
        else:
            cell['source'] = cell['source'].replace('data/historical_turnover.csv', '../data/historical_turnover.csv')
# Write back
nbformat.write(nb, nb_path)
print(f"Patched notebook: {nb_path}")

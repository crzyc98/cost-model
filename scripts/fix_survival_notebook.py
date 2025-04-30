#!/usr/bin/env python3
"""
Patch a notebook so that:
 1) A leading cell sets cwd to project root.
 2) Every code cell has 'outputs' and a non-None execution_count.
 3) Adjusts any CSV paths under data/ to ../data/.
"""
import argparse
import shutil
import re
import os
import nbformat
from nbformat.v4 import new_code_cell

def patch_notebook(nb_path:str, backup:bool=True):
    if not os.path.exists(nb_path):
        raise FileNotFoundError(f"Notebook not found: {nb_path}")
    # Backup
    if backup:
        bak = nb_path + ".bak"
        shutil.copy2(nb_path, bak)
        print(f"← backed up original to {bak}")
    # Load
    nb = nbformat.read(nb_path, as_version=4)
    cells = []
    # 1) Insert cwd cell
    cwd_cell = new_code_cell(source=(
        "import os\n"
        "from pathlib import Path\n"
        "# jump to project root (script lives in scripts/)\n"
        "os.chdir(Path(__file__).parents[1])\n"
    ))
    cells.append(cwd_cell)
    # 2) Patch existing cells
    exec_cnt = 1
    csv_pattern = re.compile(r"(['\"])data/(.*?\.csv)\1")
    for cell in nb.cells:
        cell.pop("id", None)
        if cell.cell_type == "code":
            cell.setdefault("outputs", [])
            cell["execution_count"] = exec_cnt
            exec_cnt += 1
            # rewrite any data/*.csv references to ../data/*.csv
            if isinstance(cell["source"], str):
                cell["source"] = csv_pattern.sub(r"\1../data/\2\1", cell["source"])
            else:
                cell["source"] = [
                    csv_pattern.sub(r"\1../data/\2\1", line)
                    for line in cell["source"]
                ]
        cells.append(cell)
    nb.cells = cells
    # Write back
    nbformat.write(nb, nb_path)
    print(f"✓ Patched notebook: {nb_path}")

def main():
    p = argparse.ArgumentParser(
        description="Patch paths and execution metadata in a Jupyter notebook."
    )
    p.add_argument("notebook", help="Path to .ipynb file")
    p.add_argument("--no-backup", action="store_true", 
                   help="Skip making a .bak backup")
    args = p.parse_args()
    patch_notebook(args.notebook, backup=not args.no_backup)

if __name__ == "__main__":
    main()
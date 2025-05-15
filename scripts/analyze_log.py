import re

with open("output_dev/projection_logs/projection_cli_run.log") as f:
    lines = f.readlines()

# Filter entries mentioning apply_comp_bump
bump_lines = [line for line in lines if "apply_comp_bump" in line]
print(f"Rows entering apply_comp_bump: {len(bump_lines)}")

# Sample a few entries to inspect compensation and bump amounts
# Example pattern (adjust regex based on your actual log format)
pattern = re.compile(r"pre_comp=([\d.]+), bump=([\d.]+)")

for line in bump_lines[:5]:
    match = pattern.search(line)
    if match:
        pre_comp, bump = match.groups()
        print(f"Pre-comp: {pre_comp}, Bump: {bump}")
    else:
        print("No comp data in line:", line)
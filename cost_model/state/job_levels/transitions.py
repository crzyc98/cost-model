# /cost_model/state/job_levels/transitions.py

import pandas as pd

# Promotion transition probabilities between states (rows=from, columns=to)
# States include levels 0-4 plus 'exit' as an absorbing state
STATES = [0, 1, 2, 3, 4, "exit"]

# Define the transition matrix with all rows summing to 1.0
# Each row represents the probability distribution of moving to other states
PROMOTION_MATRIX = pd.DataFrame(
    [
        # From level 0: 80% stay, 0% promote, 20% exit
        [0.80, 0.00, 0.00, 0.00, 0.00, 0.20],
        # From level 1: 15% demote, 83% stay, 2% exit
        [0.15, 0.83, 0.00, 0.00, 0.00, 0.02],
        # From level 2: 12% demote, 84% stay, 4% exit
        [0.00, 0.12, 0.84, 0.00, 0.00, 0.04],
        # From level 3: 10% demote, 85% stay, 5% exit
        [0.00, 0.00, 0.10, 0.85, 0.00, 0.05],
        # From level 4: 8% demote, 90% stay, 2% exit
        [0.00, 0.00, 0.00, 0.08, 0.90, 0.02],
        # From exit: 100% stay (absorbing state)
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    ],
    index=STATES,
    columns=STATES,
)

# Validate each row sums to 1.0 (within floating point tolerance)
assert (
    PROMOTION_MATRIX.sum(axis=1) - 1.0
).abs().max() < 1e-6, "Transition matrix rows must sum to 1.0"

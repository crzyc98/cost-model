import pandas as pd

# Promotion transition probabilities by current level (rows) to next state (columns)
# Next-state levels 0–4, plus 'exit' as terminal
PROMOTION_MATRIX = pd.DataFrame({
    0:     [0.80, None, None, None, None],
    1:     [0.15, 0.83, None, None, None],
    2:     [0.00, 0.12, 0.84, None, None],
    3:     [0.00, 0.00, 0.10, 0.85, None],
    4:     [0.00, 0.00, 0.00, 0.08, 0.92],
    "exit": [0.05, 0.05, 0.06, 0.07, 0.08],
}, index=[0, 1, 2, 3, 4])

# Fill NaNs with zero to make the matrix dense
PROMOTION_MATRIX.fillna(0.0, inplace=True)

# Validate each row sums to 1.0
assert (PROMOTION_MATRIX.sum(axis=1) - 1.0).abs().max() < 1e-6

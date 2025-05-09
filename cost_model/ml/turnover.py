# utils/ml/turnover.py

import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
from .ml_utils import predict_turnover


def apply_ml_turnover(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
    random_state: int | np.random.RandomState | None = None,
) -> pd.DataFrame:
    """
    1) Predict P(terminate) with the ML model.
    2) Make a stochastic decision for each row.
    3) Assign a uniform termination_date between hire_date and year_end.
    """
    df = df.copy()
    rng = check_random_state(random_state)

    # 1) get termination probabilities
    term_probs = predict_turnover(df, model, feature_cols, random_state=rng)

    # 2) make a Bernoulli draw for each employee
    to_terminate = rng.rand(len(df)) < term_probs.values

    if not to_terminate.any():
        return df

    # 3) uniformly choose a day between hire_date and year_end (inclusive)
    #    clip negative intervals to zero
    #    days_workable[i] = number of possible days (>= 1)
    days_workable = (year_end - df["hire_date"]).dt.days.clip(lower=0).add(1).to_numpy()

    # rand_offsets[i] in [0, days_workable[i))
    rand_offsets = (rng.rand(len(df)) * days_workable).astype(int)

    # compute full termination_date vector
    termination_dates = df["hire_date"] + pd.to_timedelta(rand_offsets, unit="D")

    # 4) assign back only for those stochastically terminated
    df.loc[to_terminate, "termination_date"] = termination_dates[to_terminate]
    df.loc[to_terminate, "status"] = "Terminated"

    return df

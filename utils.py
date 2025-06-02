## here we will make all the utils functions needed to set the data and the experimental study
## we start by importing the libraries
import pandas as pd
import numpy as np
from scipy import stats
import torch
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline



# -------------- updated simulator -------------------------------- #
def simulate_process(
    y0: float,
    coeffs: list[float],
    lag: float,
    x_bases: list[float],
    T: int = 500,               # KEEP 500 usable observations
    burnin: int = 100,          # <- NEW
    noise_x: tuple[float, float] = (0.01, 0.03),
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Simulate an AR-X process with a burn-in period.

    After discarding `burnin`, the returned DataFrame has length T.
    """
    if rng is None:
        rng = np.random.default_rng()

    m = len(coeffs)
    total_steps = T + burnin
    X = np.empty((total_steps + 1, m))            # +1 because we store X₀
    # ---- 1. generate X ---------------------------------------------------
    for j, base in enumerate(x_bases):
        X[0, j] = base
        sd = rng.uniform(*noise_x)
        if (j + 1) % 2:                           # odd → random walk + drift
            trend = rng.uniform(0.001, 0.1)
            for t in range(1, total_steps + 1):
                X[t, j] = X[t-1, j] + trend + rng.normal(0, sd)
        else:                                     # even → flat w/ noise
            for t in range(1, total_steps + 1):
                X[t, j] = base + rng.normal(0, sd)

    # ---- 2. generate y ---------------------------------------------------
    y = np.empty(total_steps + 1)
    y[0] = y0
    noise = rng.normal(0, 1)
    for t in range(1, total_steps + 1):
        y[t] = lag * y[t-1] + X[t-1].dot(coeffs) + noise 

    # ---- 3. discard burn-in & wrap up -----------------------------------
    keep_slice = slice(burnin, burnin + T + 1)      # +1 keeps y_T
    df = pd.DataFrame(X[keep_slice], columns=[f"v{j+1}" for j in range(m)])
    df["y"] = y[keep_slice]
    return df





## here we will define a function to run the model and evaluated it
## the data built using this will be used to train the model
def df_to_tensors(ddf, continuous, task):
    X = torch.from_numpy(ddf[continuous].values).float()      # shape (n,10)
    y = torch.from_numpy(ddf["y"].values).float()              # shape (n,)
    task = (ddf[task].astype(int) - 1).values             # 0,1,2
    t = torch.from_numpy(task).long()                          # shape (n,)
    return X, y, t

def df_to_tensors_soc(ddf, continuous, task):
    X = torch.from_numpy(ddf[continuous].values).float()      # shape (n,10)
    y = torch.from_numpy(ddf["SoC"].values).float()              # shape (n,)
    task = (ddf[task].astype(int) - 1).values             # 0,1,2
    t = torch.from_numpy(task).long()                          # shape (n,)
    return X, y, t

## this function is used to clean the dataset 
## and calculate the state of charge based on data provided by umd and information about the batteries

def battery_cleaning(df, battery_index, capacity_nominal, columns_drop):

    ## we read the csv
    df = df.drop(columns=columns_drop)
    df["SoC"] = ((df["Charge_Capacity(Ah)"]-df["Discharge_Capacity(Ah)"])/capacity_nominal)*100
    df = df.drop(columns=["Charge_Capacity(Ah)", "Discharge_Capacity(Ah)"])
    ## set the battery type
    df["battery"] = battery_index
    return df
    #df.to_csv(f"data_battery/battery_type{battery_index}.csv")
    

def reduce_time_series(df, n_points=500):
    # 1) original x: just [0,1,2,...,len(df)-1]
    x = np.arange(len(df), dtype=float)

    # 2) new, coarse grid of exactly n_points between 0 and len(df)-1
    x_new = np.linspace(0, len(df)-1, n_points)

    # 3) for each column, either carry constant or spline‐interpolate
    data = {}
    for col in df.columns:
        vals = df[col].values
        if df[col].nunique() == 1:
            # trivial column – just repeat the same value
            data[col] = np.full(n_points, vals[0])
        else:
            cs = CubicSpline(x, vals)      # exact cubic through your data
            data[col] = cs(x_new)          # <- this is the KEY: evaluate at x_new

    # 4) build the reduced DataFrame
    df_reduced = pd.DataFrame(data, index=np.round(x_new).astype(int))
    #    if you’d rather have a fresh 0..n_points-1 index, uncomment:
    # df_reduced = df_reduced.reset_index(drop=True)

    return df_reduced
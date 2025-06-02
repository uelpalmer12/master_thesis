import numpy as np
import pandas as pd

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


rng = np.random.default_rng(195)

x_bases = [                     # same 10 baselines
    rng.uniform(6, 7), rng.poisson(9, 1), rng.gamma(9, 3),
    rng.uniform(30, 31), rng.uniform(3, 4), rng.normal(1, 1),
    rng.uniform(6, 7), rng.beta(6, 7), rng.normal(6, 1),
    rng.uniform(20, 24),
]

specs = [
    dict(y0=rng.uniform(4, 5),
         coeffs=[0.5, 0.1, -0.2, 0.2, 1.2, 0.43, 0.4, 1.2, -0.6, -0.4],
         lag=-0.4),
    dict(y0=rng.uniform(7, 8),
         coeffs=[3.7, 5.1, -7.2, 0.8, 1.2, 3.3, 0.5, 1.2, -0.6, -3.9],
         lag=0.1),
    dict(y0=rng.uniform(7, 8),
         coeffs=[1.7, 0.1, -6.2, 3.4, 2.2, 5.3, 1.5, 3.2, -0.6, -1.9],
         lag=0.7),
]

frames = []
for i, s in enumerate(specs, 1):
    df = simulate_process(**s, x_bases=x_bases, T=500, burnin=200, rng=rng)
    df["process"] = i
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
combined.to_csv("simulated_processes.csv")



# import matplotlib.pyplot as plt

# for proc in combined["process"].unique():
#     sub = combined[combined["process"] == proc].reset_index(drop=True)
#     plt.figure()
#     plt.plot(sub.index, sub["y"])
#     plt.title(f"Process {proc} – y over time")
#     plt.xlabel("Time step")
#     plt.ylabel("y")
#     plt.tight_layout()
#     plt.show()
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 1. Discover all your metrics files
files = glob.glob("metrics_*.csv")

# 2. Read & concat
dfs = [pd.read_csv(f) for f in files]
all_df = pd.concat(dfs, ignore_index=True)

# 3. For each metric, make the appropriate boxplot
for metric in ["MSE", "Width", "Coverage%", "Robust W", "Robust Cov%"]:
    plt.figure()
    groups = [
        all_df[all_df["model"] == m][metric].values
        for m in sorted(all_df["model"].unique())
    ]
    plt.boxplot(groups, labels=sorted(all_df["model"].unique()))

    if metric != "Coverage%" and metric !="Robust Cov%":
        plt.yscale('log')
        plt.title(f"Comparison of {metric} (log scale)")
    else:
        plt.title(f"Comparison of {metric}")

    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.tight_layout()
    plt.show()


import pandas as pd
import numpy as np
import os

in_path = "kumamoto_challengedata.csv"

TARGETS = [(101, 102), (101, 103), (100, 102)]
LOOKBACK = 24

df = pd.read_csv(in_path)  # 欄位: uid, d, t, x, y

# t -> hour
t_max = int(df["t"].max())
if t_max >= 47:
    df["hour"] = (df["t"] // 2).astype(int)
else:
    df["hour"] = df["t"].astype(int)

# 只取你要的三個座標
targets_df = pd.DataFrame(TARGETS, columns=["x", "y"])
df = df.merge(targets_df, on=["x", "y"], how="inner")

# 聚合成每小時人流（unique uid）
hourly = (
    df.groupby(["x", "y", "d", "hour"], as_index=False)
      .agg(people_count=("uid", "nunique"))
)

# 補齊缺的時間點 (d, hour)，沒人就是 0
d_min, d_max = int(hourly["d"].min()), int(hourly["d"].max())
all_days = np.arange(d_min, d_max + 1)
all_hours = np.arange(0, 24)

def build_series_for_one_grid(sub):
    x0, y0 = int(sub["x"].iloc[0]), int(sub["y"].iloc[0])
    full = pd.MultiIndex.from_product([all_days, all_hours], names=["d", "hour"]).to_frame(index=False)
    full = full.merge(sub[["d", "hour", "people_count"]], on=["d","hour"], how="left").fillna({"people_count": 0})
    full["x"] = x0
    full["y"] = y0
    full["time_idx"] = full["d"] * 24 + full["hour"]
    full = full.sort_values("time_idx").reset_index(drop=True)
    return full

series_list = []
for (x, y) in TARGETS:
    sub = hourly[(hourly["x"] == x) & (hourly["y"] == y)].copy()
    series_list.append(build_series_for_one_grid(sub))

series_all = pd.concat(series_list, ignore_index=True)

# sliding window -> (X, y)
def make_windows(values, lookback):
    X, Y = [], []
    # 預測下一小時，所以最後一個可用的 input 結尾是 len(values)-2
    for end in range(lookback - 1, len(values) - 1):
        start = end - lookback + 1
        X.append(values[start:end+1])
        Y.append(values[end+1])
    return np.array(X), np.array(Y)

datasets = {}
for (x, y) in TARGETS:
    s = series_all[(series_all["x"] == x) & (series_all["y"] == y)].sort_values("time_idx")
    vals = s["people_count"].to_numpy(dtype=float)
    X, Y = make_windows(vals, LOOKBACK)

    N = len(Y)
    n_train = int(N * 0.8)
    n_val   = int(N * 0.1)
    # n_test = N - n_train - n_val

    X_train, y_train = X[:n_train], Y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:], Y[n_train+n_val:]

    datasets[(x, y)] = {
        "train": (X_train, y_train),
        "val":   (X_val, y_val),
        "test":  (X_test, y_test),
    }

    print(f"Grid ({x},{y}) -> N={N}, train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

os.makedirs("datasets", exist_ok=True)

for (x, y), data in datasets.items():
    save_path = f"datasets/grid_{x}_{y}.npz"

    np.savez(
        save_path,
        X_train=data["train"][0],
        y_train=data["train"][1],
        X_val=data["val"][0],
        y_val=data["val"][1],
        X_test=data["test"][0],
        y_test=data["test"][1],
    )

    print(f"saved: {save_path}")

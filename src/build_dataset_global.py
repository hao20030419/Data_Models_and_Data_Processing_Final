import pandas as pd
import numpy as np
import os

# ---------- 路徑設定 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

INPUT_CSV = os.path.join(PROJECT_ROOT, "datasets", "hourly_people_flow.csv")
OUTPUT_NPZ = os.path.join(PROJECT_ROOT, "datasets", "global_dataset.npz")

LOOKBACK = 24

df = pd.read_csv(INPUT_CSV)

# 建立完整時間範圍
all_days = np.arange(df["d"].min(), df["d"].max() + 1)
all_hours = np.arange(0, 24)

X_all, y_all, t_all = [], [], []

for (x, y), g in df.groupby(["x", "y"]):

    # 1️⃣ 建立完整時間軸
    full_time = pd.MultiIndex.from_product(
        [all_days, all_hours],
        names=["d", "hour"]
    ).to_frame(index=False)

    # 2️⃣ left join，補 0
    g_full = (
        full_time
        .merge(g[["d", "hour", "people_count"]], on=["d", "hour"], how="left")
        .fillna({"people_count": 0})
    )

    g_full["time_idx"] = g_full["d"] * 24 + g_full["hour"]
    g_full = g_full.sort_values("time_idx")

    values = g_full["people_count"].to_numpy()

    # 3️⃣ sliding window
    for end in range(LOOKBACK - 1, len(values) - 1):
        start = end - LOOKBACK + 1
        seq = values[start:end + 1]

        X_all.append(np.concatenate([seq, [x, y]]))  # 24 + x,y
        y_all.append(values[end + 1])
        t_all.append(g_full["time_idx"].iloc[end + 1])

# 轉成 array 並依時間排序
X_all = np.array(X_all)
y_all = np.array(y_all)
t_all = np.array(t_all)

idx = np.argsort(t_all)
X_all, y_all = X_all[idx], y_all[idx]

# ---------- 切資料集（時間切） ----------
N = len(y_all)
n_train = int(N * 0.8)
n_val = int(N * 0.1)

X_train, y_train = X_all[:n_train], y_all[:n_train]
X_val, y_val = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
X_test, y_test = X_all[n_train + n_val:], y_all[n_train + n_val:]

os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)

np.savez(
    OUTPUT_NPZ,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test
)

print("Saved:", OUTPUT_NPZ)
print("Train / Val / Test:", len(y_train), len(y_val), len(y_test))
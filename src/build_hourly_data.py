import pandas as pd
import os

# 取得目前檔案所在位置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 專案根目錄
PROJECT_ROOT = os.path.dirname(BASE_DIR)

INPUT_CSV = os.path.join(PROJECT_ROOT, "kumamoto_challengedata.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "datasets", "hourly_people_flow.csv")

# 確保輸出目錄存在
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# t → hour
t_max = int(df["t"].max())
if t_max >= 47:
    df["hour"] = (df["t"] // 2).astype(int)
else:
    df["hour"] = df["t"].astype(int)

# 每小時、每 grid 的人流（unique uid）
hourly = (
    df.groupby(["d", "hour", "x", "y"], as_index=False)
      .agg(people_count=("uid", "nunique"))
)

hourly.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
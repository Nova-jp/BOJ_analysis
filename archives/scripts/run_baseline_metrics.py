import sys
import pandas as pd
from src.processing import load_and_clean_data
from src.features import generate_features
from src.pooling import pool_boj_data
from src.modeling import walk_forward_validation, summarize_ic

EXCEL_PATH   = 'data/BOJ_data.xlsx'
MEETING_PATH = 'data/BOJ_meeting_history.csv'
START_DATE   = '2024-01-01'

print("Loading data...")
df_raw    = load_and_clean_data(EXCEL_PATH, MEETING_PATH)
print("Generating features...")
df_feat   = generate_features(df_raw)
print("Pooling data...")
df_pooled = pool_boj_data(df_feat)

print("Running 3d Walk-forward Validation...")
res_3d = walk_forward_validation(df_pooled, 'Target_3d_norm', START_DATE)
print("Running 5d Walk-forward Validation...")
res_5d = walk_forward_validation(df_pooled, 'Target_5d_norm', START_DATE)

metrics_3d = summarize_ic(res_3d)
metrics_5d = summarize_ic(res_5d)

print("\n=== OOS Metrics Summary (Baseline) ===")
print(f"3d Horizon:")
print(f"  Global IC: {metrics_3d['ic_all']:.4f}")
print(f"  CS IC:     {metrics_3d['cs_ic']:.4f}")
print(f"  TS IC:     {metrics_3d['ts_ic']:.4f}")
print(f"  Train IC:  {metrics_3d['train_ic']:.4f}")
print(f"  Gap:       {metrics_3d['gap']:.4f}")

print(f"\n5d Horizon:")
print(f"  Global IC: {metrics_5d['ic_all']:.4f}")
print(f"  CS IC:     {metrics_5d['cs_ic']:.4f}")
print(f"  TS IC:     {metrics_5d['ts_ic']:.4f}")
print(f"  Train IC:  {metrics_5d['train_ic']:.4f}")
print(f"  Gap:       {metrics_5d['gap']:.4f}")

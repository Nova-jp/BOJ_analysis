import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.processing import load_and_clean_data
from src.features import generate_features
from src.pooling import pool_boj_data, pool_boj_butterfly, butterfly_to_spread
from src.modeling import walk_forward_validation, calculate_metrics

EXCEL_PATH = "data/BOJ_data.xlsx"
MEETING_CSV_PATH = "data/BOJ_meeting_history.csv"
START_DATE = "2024-01-01"

# データの読み込み
df_raw = load_and_clean_data(EXCEL_PATH, MEETING_CSV_PATH)

def run_experiment(df_raw, label, **feat_kwargs):
    print(f"Running {label}...")
    df_feat = generate_features(df_raw, **feat_kwargs)
    df_pooled = pool_boj_data(df_feat)
    
    # 3d 予測
    res_3d = walk_forward_validation(df_pooled, "Target_3d", START_DATE)
    metrics_3d = calculate_metrics(res_3d["Actual"], res_3d["Pred"])
    
    # 5d 予測
    res_5d = walk_forward_validation(df_pooled, "Target_5d", START_DATE)
    metrics_5d = calculate_metrics(res_5d["Actual"], res_5d["Pred"])
    
    return metrics_3d["IC"], metrics_5d["IC"]

# --- Phase 1: Exp-A, B, C ---
results = []
# Baseline
ic3_base, ic5_base = run_experiment(df_raw, "Baseline")
results.append({"Experiment": "Baseline", "IC_3d": ic3_base, "IC_5d": ic5_base})

# Exp-A (Weekday)
ic3, ic5 = run_experiment(df_raw, "Exp-A (Weekday)", add_weekday_cyclic=True)
results.append({"Experiment": "Exp-A", "IC_3d": ic3, "IC_5d": ic5})

# Exp-B (DTM cyclic)
ic3, ic5 = run_experiment(df_raw, "Exp-B (DTM cyclic)", add_days_to_mpm_cyclic=True)
results.append({"Experiment": "Exp-B", "IC_3d": ic3, "IC_5d": ic5})

# Exp-C (Curve)
ic3, ic5 = run_experiment(df_raw, "Exp-C (Curve)", add_curve_features=True)
results.append({"Experiment": "Exp-C", "IC_3d": ic3, "IC_5d": ic5})

# Exp-ABC (All Phase 1)
ic3, ic5 = run_experiment(df_raw, "Exp-ABC (All P1)", add_weekday_cyclic=True, add_days_to_mpm_cyclic=True, add_curve_features=True)
results.append({"Experiment": "Exp-ABC", "IC_3d": ic3, "IC_5d": ic5})

# --- Phase 2: Exp-D, E, DE ---
def run_pooled_experiment(df_raw, label, feature_set):
    print(f"Running {label}...")
    df_feat = generate_features(df_raw)
    df_pooled = pool_boj_data(df_feat)
    all_optional = ["Absolute_Meeting_ID", "Days_since_first_seen"]
    drop_cols = [c for c in all_optional if c not in feature_set]
    df_input = df_pooled.drop(columns=drop_cols)
    res_3d = walk_forward_validation(df_input, "Target_3d", START_DATE)
    metrics_3d = calculate_metrics(res_3d["Actual"], res_3d["Pred"])
    res_5d = walk_forward_validation(df_input, "Target_5d", START_DATE)
    metrics_5d = calculate_metrics(res_5d["Actual"], res_5d["Pred"])
    return metrics_3d["IC"], metrics_5d["IC"]

ic3, ic5 = run_pooled_experiment(df_raw, "Exp-D (AbsID)", ["Absolute_Meeting_ID"])
results.append({"Experiment": "Exp-D", "IC_3d": ic3, "IC_5d": ic5})

ic3, ic5 = run_pooled_experiment(df_raw, "Exp-E (DaysSince)", ["Days_since_first_seen"])
results.append({"Experiment": "Exp-E", "IC_3d": ic3, "IC_5d": ic5})

ic3, ic5 = run_pooled_experiment(df_raw, "Exp-DE (Both)", ["Absolute_Meeting_ID", "Days_since_first_seen"])
results.append({"Experiment": "Exp-DE", "IC_3d": ic3, "IC_5d": ic5})

# --- Phase 2: Exp-F (Butterfly Target) ---
def run_butterfly_experiment(df_raw, horizon):
    print(f"Running Exp-F (h={horizon})...")
    df_feat = generate_features(df_raw)
    df_b = pool_boj_butterfly(df_feat)
    target_col = f"Target_{horizon}d_B_norm"
    df_b["Butterfly_Index"] = df_b["Butterfly_Index"].astype(int)
    df_input = df_b.rename(columns={"Butterfly_Index": "Meeting_Index"})
    res_b = walk_forward_validation(df_input, target_col, START_DATE)
    # 標準偏差をマージ
    res_b = pd.merge(res_b, df_input[["Date", "Meeting_Index", f"Target_{horizon}d_B_std"]], on=["Date", "Meeting_Index"], how="left")
    
    # M_n 空間への逆変換後の IC
    # 予測値を (Date, Index) でピボット
    pred_pivot = res_b.pivot(index="Date", columns="Meeting_Index", values="Pred")
    # 各日付の標準偏差でスケールを戻す
    std_pivot = res_b.pivot(index="Date", columns="Meeting_Index", values=f"Target_{horizon}d_B_std")
    pred_raw_b = pred_pivot * std_pivot
    reconstructed_preds = []
    for _, row in pred_raw_b.iterrows():
        reconstructed_preds.append(butterfly_to_spread(row.values))
    pred_mn = pd.DataFrame(reconstructed_preds, index=pred_raw_b.index, columns=[f"M{i}" for i in range(1, 9)])
    df_p = pool_boj_data(df_feat)
    actual_mn = df_p[df_p["Is_Tenor_OIS"] == 0].pivot(index="Date", columns="Meeting_Index", values=f"Target_{horizon}d")
    common_dates = pred_mn.index.intersection(actual_mn.index)
    y_true = actual_mn.loc[common_dates].values.flatten()
    y_pred = pred_mn.loc[common_dates].values.flatten()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    ic_mn = calculate_metrics(y_true[mask], y_pred[mask])["IC"]
    return ic_mn

ic3_f = run_butterfly_experiment(df_raw, 3)
ic5_f = run_butterfly_experiment(df_raw, 5)
results.append({"Experiment": "Exp-F", "IC_3d": ic3_f, "IC_5d": ic5_f})

# --- Summary ---
df_results = pd.DataFrame(results)
df_results["Delta_IC_3d"] = df_results["IC_3d"] - ic3_base
df_results["Delta_IC_5d"] = df_results["IC_5d"] - ic5_base

print("\n### 実験結果一覧 ###")
print(df_results.to_string(index=False))

# --- Additional Checks ---
print("\n### Exp-C Feature Importance (5d) ###")
df_feat = generate_features(df_raw, add_curve_features=True)
df_pooled = pool_boj_data(df_feat)
res, model, X_test, y_test, X_train = walk_forward_validation(df_pooled, "Target_5d", START_DATE, return_model=True)
importance = pd.DataFrame({"feature": X_test.columns, "importance": model.feature_importance(importance_type="gain")}).sort_values("importance", ascending=False)
print(importance.head(10).to_string(index=False))

print("\n### Exp-E Correlation with Days_to_MPM ###")
df_feat = generate_features(df_raw)
df_pooled = pool_boj_data(df_feat)
corr = df_pooled[["Days_since_first_seen", "Days_to_MPM"]].corr().iloc[0, 1]
print(f"Correlation: {corr:.4f}")

import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Add src to path
sys.path.insert(0, os.getcwd())

from src.processing import load_and_clean_data
from src.features import generate_features
from src.pooling import pool_boj_data, pool_boj_butterfly, butterfly_to_spread
from src.modeling import walk_forward_validation, calculate_metrics

# Constants
EXCEL_PATH = "data/BOJ_data.xlsx"
MEETING_CSV_PATH = "data/BOJ_meeting_history.csv"
START_DATE = "2024-01-01"

def run_experiment(df_raw, label, **feat_kwargs):
    print(f"Running {label}...")
    df_feat = generate_features(df_raw, **feat_kwargs)
    df_pooled = pool_boj_data(df_feat)
    
    # 3d
    res_3d = walk_forward_validation(df_pooled, "Target_3d", START_DATE)
    metrics_3d = calculate_metrics(res_3d["Actual"], res_3d["Pred"])
    
    # 5d
    res_5d = walk_forward_validation(df_pooled, "Target_5d", START_DATE)
    metrics_5d = calculate_metrics(res_5d["Actual"], res_5d["Pred"])
    
    print(f"  {label}: IC_3d={metrics_3d['IC']:.4f}, IC_5d={metrics_5d['IC']:.4f}")
    return metrics_3d["IC"], metrics_5d["IC"]

def run_pooled_experiment(df_raw, label, feature_set):
    print(f"Running {label}...")
    df_feat = generate_features(df_raw)
    df_pooled = pool_boj_data(df_feat)
    
    all_optional = ["Absolute_Meeting_ID", "Days_since_first_seen"]
    drop_cols = [c for c in all_optional if c not in feature_set]
    df_input = df_pooled.drop(columns=drop_cols)
    
    res_3d, model_3d, X_test_3d, y_test_3d, X_train_3d = walk_forward_validation(df_input, "Target_3d", START_DATE, return_model=True)
    metrics_3d = calculate_metrics(res_3d["Actual"], res_3d["Pred"])
    
    res_5d, model_5d, X_test_5d, y_test_5d, X_train_5d = walk_forward_validation(df_input, "Target_5d", START_DATE, return_model=True)
    metrics_5d = calculate_metrics(res_5d["Actual"], res_5d["Pred"])
    
    print(f"  {label}: IC_3d={metrics_3d['IC']:.4f}, IC_5d={metrics_5d['IC']:.4f}")
    return metrics_3d["IC"], metrics_5d["IC"], model_3d, X_test_3d, model_5d, X_test_5d

def run_butterfly_experiment(df_raw, horizon):
    print(f"Running Exp-F (h={horizon})...")
    df_feat = generate_features(df_raw)
    df_b = pool_boj_butterfly(df_feat)
    
    target_col = f"Target_{horizon}d_B_norm"
    df_b["Butterfly_Index"] = df_b["Butterfly_Index"].astype(int)
    # modeling.py の walk_forward_validation は Meeting_Index を期待しているので、
    # Butterfly_Index を Meeting_Index にリネームして渡す
    df_input = df_b.rename(columns={"Butterfly_Index": "Meeting_Index"})
    df_input["Is_Tenor_OIS"] = 0
    if "is_post_mpm" not in df_input.columns:
        df_input["is_post_mpm"] = 0
    
    # Butterfly 空間での IC
    res_b = walk_forward_validation(df_input, target_col, START_DATE)
    metrics_b = calculate_metrics(res_b["Actual"], res_b["Pred"])
    
    # Inverse Transform
    pred_pivot = res_b.pivot(index="Date", columns="Meeting_Index", values="Pred")
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
    
    return metrics_b["IC"], ic_mn

def main():
    # 0. Load Data
    df_raw = load_and_clean_data(EXCEL_PATH, MEETING_CSV_PATH)
    
    # 1. Phase 1 Experiments (A, B, C, ABC)
    results = []
    baseline_3d, baseline_5d = run_experiment(df_raw, "Baseline")
    results.append({"Experiment": "Baseline", "IC_3d": baseline_3d, "IC_5d": baseline_5d})
    
    ic_a_3d, ic_a_5d = run_experiment(df_raw, "Exp-A", add_weekday_cyclic=True)
    results.append({"Experiment": "Exp-A", "IC_3d": ic_a_3d, "IC_5d": ic_a_5d})
    
    ic_b_3d, ic_b_5d = run_experiment(df_raw, "Exp-B", add_days_to_mpm_cyclic=True)
    results.append({"Experiment": "Exp-B", "IC_3d": ic_b_3d, "IC_5d": ic_b_5d})
    
    # For Exp-C importance
    print("Running Exp-C for Importance...")
    df_feat_c = generate_features(df_raw, add_curve_features=True)
    df_pooled_c = pool_boj_data(df_feat_c)
    res_c_3d, model_c_3d, X_test_c_3d, _, _ = walk_forward_validation(df_pooled_c, "Target_3d", START_DATE, return_model=True)
    ic_c_3d = calculate_metrics(res_c_3d["Actual"], res_c_3d["Pred"])["IC"]
    
    res_c_5d, model_c_5d, X_test_c_5d, _, _ = walk_forward_validation(df_pooled_c, "Target_5d", START_DATE, return_model=True)
    ic_c_5d = calculate_metrics(res_c_5d["Actual"], res_c_5d["Pred"])["IC"]
    results.append({"Experiment": "Exp-C", "IC_3d": ic_c_3d, "IC_5d": ic_c_5d})
    
    ic_abc_3d, ic_abc_5d = run_experiment(df_raw, "Exp-ABC", add_weekday_cyclic=True, add_days_to_mpm_cyclic=True, add_curve_features=True)
    results.append({"Experiment": "Exp-ABC", "IC_3d": ic_abc_3d, "IC_5d": ic_abc_5d})
    
    # 2. Phase 2 Experiments (D, E, DE)
    ic_d_3d, ic_d_5d, _, _, _, _ = run_pooled_experiment(df_raw, "Exp-D", ["Absolute_Meeting_ID"])
    results.append({"Experiment": "Exp-D", "IC_3d": ic_d_3d, "IC_5d": ic_d_5d})
    
    ic_e_3d, ic_e_5d, _, _, _, _ = run_pooled_experiment(df_raw, "Exp-E", ["Days_since_first_seen"])
    results.append({"Experiment": "Exp-E", "IC_3d": ic_e_3d, "IC_5d": ic_e_5d})
    
    ic_de_3d, ic_de_5d, model_de_3d, X_test_de_3d, model_de_5d, X_test_de_5d = run_pooled_experiment(df_raw, "Exp-DE", ["Absolute_Meeting_ID", "Days_since_first_seen"])
    results.append({"Experiment": "Exp-DE", "IC_3d": ic_de_3d, "IC_5d": ic_de_5d})
    
    # 3. Phase 2 Experiment (F)
    ic_f_b_3d, ic_f_mn_3d = run_butterfly_experiment(df_raw, 3)
    ic_f_b_5d, ic_f_mn_5d = run_butterfly_experiment(df_raw, 5)
    results.append({"Experiment": "Exp-F", "IC_3d": ic_f_mn_3d, "IC_5d": ic_f_mn_5d})
    
    # --- Data Collection ---
    
    # Delta IC Table
    df_res = pd.DataFrame(results)
    df_res["Delta_IC_3d"] = df_res["IC_3d"] - baseline_3d
    df_res["Delta_IC_5d"] = df_res["IC_5d"] - baseline_5d
    
    # Exp-C Gain Importance Top 5 (Curve features)
    curve_features = ['Curve_Slope'] + [f'Butterfly_M{n}' for n in range(2, 8)] + \
                     ['Curve_Slope_frac_diff'] + [f'Butterfly_M{n}_frac_diff' for n in range(2, 8)]
    
    imp_c_3d = pd.DataFrame({"feat": X_test_c_3d.columns, "imp": model_c_3d.feature_importance(importance_type='gain')})
    imp_c_3d_curve = imp_c_3d[imp_c_3d["feat"].isin(curve_features)].sort_values("imp", ascending=False).head(5)
    
    imp_c_5d = pd.DataFrame({"feat": X_test_c_5d.columns, "imp": model_c_5d.feature_importance(importance_type='gain')})
    imp_c_5d_curve = imp_c_5d[imp_c_5d["feat"].isin(curve_features)].sort_values("imp", ascending=False).head(5)
    
    # Exp-E Independence (Correlation with Days_to_MPM)
    df_feat_e = generate_features(df_raw)
    df_pooled_e = pool_boj_data(df_feat_e)
    df_pooled_e = df_pooled_e.dropna(subset=["Days_since_first_seen", "Days_to_MPM"])
    corr_e_dtm, _ = pearsonr(df_pooled_e["Days_since_first_seen"], df_pooled_e["Days_to_MPM"])
    
    # Exp-DE SHAP
    explainer_3d = shap.TreeExplainer(model_de_3d)
    shap_3d = explainer_3d.shap_values(X_test_de_3d)
    # shap_3d is an array of shape (n_samples, n_features)
    shap_imp_3d = pd.DataFrame({"feat": X_test_de_3d.columns, "shap": np.abs(shap_3d).mean(axis=0)}).sort_values("shap", ascending=False)
    
    # Final Report Generation
    print("\n" + "="*50)
    print("EXPERIMENTAL REPORT")
    print("="*50)
    
    print("\n### 1. 各実験の OOS IC と Baseline との差（Delta IC）比較表 ###")
    print(df_res.to_string(index=False))
    
    print("\n### 2. Exp-B（Days_to_MPM 円環）の効果に関するコメント ###")
    gain_b_3d = ic_b_3d - baseline_3d
    gain_b_5d = ic_b_5d - baseline_5d
    if gain_b_3d > 0.005 or gain_b_5d > 0.005:
        print(f"改善が見られました (Delta 3d: {gain_b_3d:.4f}, 5d: {gain_b_5d:.4f})。円環エンコーディングにより会合サイクル内の相対位置が有効に機能している可能性があります。")
    else:
        print(f"改善は限定的でした (Delta 3d: {gain_b_3d:.4f}, 5d: {gain_b_5d:.4f})。GBDTのしきい値分割能力により、数値のままでも十分学習できていると考えられます。")
        
    print("\n### 3. Exp-C のカーブ特徴量の Gain importance 上位 5 つ ###")
    print("--- 3d 予測 ---")
    print(imp_c_3d_curve.to_string(index=False))
    print("--- 5d 予測 ---")
    print(imp_c_5d_curve.to_string(index=False))
    
    print("\n### 4. Exp-DE の SHAP 寄与に関するコメント ###")
    abs_id_shap = shap_imp_3d[shap_imp_3d["feat"] == "Absolute_Meeting_ID"]["shap"].values[0]
    days_since_shap = shap_imp_3d[shap_imp_3d["feat"] == "Days_since_first_seen"]["shap"].values[0]
    print(f"Absolute_Meeting_ID SHAP: {abs_id_shap:.4f}")
    print(f"Days_since_first_seen SHAP: {days_since_shap:.4f}")
    print("上位SHAP特徴量順位:")
    print(shap_imp_3d.head(10).to_string(index=False))
    
    print("\n### 5. Exp-E と Days_to_MPM の相関係数 ###")
    print(f"Pearson Correlation: {corr_e_dtm:.4f}")
    
    print("\n### 6. Exp-F（バタフライ変換）の逆変換後 Mn 空間での IC 比較 ###")
    print(f"3d: Mn Space IC = {ic_f_mn_3d:.4f} (Baseline IC = {baseline_3d:.4f}, Delta = {ic_f_mn_3d - baseline_3d:.4f})")
    print(f"5d: Mn Space IC = {ic_f_mn_5d:.4f} (Baseline IC = {baseline_5d:.4f}, Delta = {ic_f_mn_5d - baseline_5d:.4f})")
    print(f"Butterfly Space IC: 3d = {ic_f_b_3d:.4f}, 5d = {ic_f_b_5d:.4f}")
    
    print("\n### 7. 採用推奨の特徴量グループの提案 ###")
    recommendations = []
    if (ic_abc_3d - baseline_3d) > 0.01 or (ic_abc_5d - baseline_5d) > 0.01:
        recommendations.append("Exp-ABC (Weekday/DTM Cyclic + Curve features)")
    if (ic_de_3d - baseline_3d) > 0.01 or (ic_de_5d - baseline_5d) > 0.01:
        recommendations.append("Exp-DE (Meeting Identification)")
    if (ic_f_mn_3d - baseline_3d) > 0.01 or (ic_f_mn_5d - baseline_5d) > 0.01:
        recommendations.append("Exp-F (Butterfly Target Transformation)")
        
    if not recommendations:
        # Find best individual
        best_exp = df_res.loc[df_res["Delta_IC_5d"].idxmax()]
        print(f"全体的に劇的な改善は見られませんでしたが、最も効果的だったのは {best_exp['Experiment']} でした。")
    else:
        print("以下を採用推奨とします:")
        for rec in recommendations:
            print(f"- {rec}")

if __name__ == "__main__":
    main()

import json

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# 07 特徴量エンジニアリング実験\n',
                '\n',
                '本ノートブックでは、`designs/04_feature_engineering.md` に基づき、以下の特徴量グループの追加効果を検証します。\n',
                '- **Exp-A**: 曜日の円環エンコーディング\n',
                '- **Exp-B**: Days_to_MPM の円環エンコーディング\n',
                '- **Exp-C**: カーブ形状（Slope・Butterfly）の水準と変化速度\n',
                '- **Exp-ABC**: 全部のせ'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import sys\n',
                'sys.path.insert(0, "..")\n',
                'import pandas as pd\n',
                'import numpy as np\n',
                'import matplotlib.pyplot as plt\n',
                'import seaborn as sns\n',
                'import lightgbm as lgb\n',
                'from src.processing import load_and_clean_data\n',
                'from src.features import generate_features\n',
                'from src.pooling import pool_boj_data\n',
                'from src.modeling import walk_forward_validation, calculate_metrics\n',
                '\n',
                'EXCEL_PATH = "../data/BOJ_data.xlsx"\n',
                'MEETING_CSV_PATH = "../data/BOJ_meeting_history.csv"\n',
                'START_DATE = "2024-01-01"\n',
                '\n',
                '# データの読み込み\n',
                'df_raw = load_and_clean_data(EXCEL_PATH, MEETING_CSV_PATH)'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 実験の実行定義'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'def run_experiment(df_raw, label, **feat_kwargs):\n',
                '    print(f"Running {label}...")\n',
                '    df_feat = generate_features(df_raw, **feat_kwargs)\n',
                '    df_pooled = pool_boj_data(df_feat)\n',
                '    \n',
                '    # 3d 予測\n',
                '    res_3d = walk_forward_validation(df_pooled, "Target_3d", START_DATE)\n',
                '    metrics_3d = calculate_metrics(res_3d["Actual"], res_3d["Pred"])\n',
                '    \n',
                '    # 5d 予測\n',
                '    res_5d = walk_forward_validation(df_pooled, "Target_5d", START_DATE)\n',
                '    metrics_5d = calculate_metrics(res_5d["Actual"], res_5d["Pred"])\n',
                '    \n',
                '    print(f"  {label}: IC_3d={metrics_3d[\'IC\']:.4f}, IC_5d={metrics_5d[\'IC\']:.4f}")\n',
                '    return metrics_3d["IC"], metrics_5d["IC"]'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 各実験の実行'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'results = []\n',
                '\n',
                '# Baseline\n',
                'ic_3d, ic_5d = run_experiment(df_raw, "Baseline")\n',
                'results.append({"Experiment": "Baseline", "IC_3d": ic_3d, "IC_5d": ic_5d})\n',
                '\n',
                '# Exp-A (Weekday)\n',
                'ic_3d, ic_5d = run_experiment(df_raw, "Exp-A (Weekday)", add_weekday_cyclic=True)\n',
                'results.append({"Experiment": "Exp-A", "IC_3d": ic_3d, "IC_5d": ic_5d})\n',
                '\n',
                '# Exp-B (DTM cyclic)\n',
                'ic_3d, ic_5d = run_experiment(df_raw, "Exp-B (DTM cyclic)", add_days_to_mpm_cyclic=True)\n',
                'results.append({"Experiment": "Exp-B", "IC_3d": ic_3d, "IC_5d": ic_5d})\n',
                '\n',
                '# Exp-C (Curve)\n',
                'ic_3d, ic_5d = run_experiment(df_raw, "Exp-C (Curve)", add_curve_features=True)\n',
                'results.append({"Experiment": "Exp-C", "IC_3d": ic_3d, "IC_5d": ic_5d})\n',
                '\n',
                '# Exp-ABC (All)\n',
                'ic_3d, ic_5d = run_experiment(df_raw, "Exp-ABC (All)", \n',
                '                              add_weekday_cyclic=True, \n',
                '                              add_days_to_mpm_cyclic=True, \n',
                '                              add_curve_features=True)\n',
                'results.append({"Experiment": "Exp-ABC", "IC_3d": ic_3d, "IC_5d": ic_5d})'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 結果の比較'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'df_results = pd.DataFrame(results)\n',
                'baseline_ic_3d = df_results.loc[df_results["Experiment"] == "Baseline", "IC_3d"].values[0]\n',
                'baseline_ic_5d = df_results.loc[df_results["Experiment"] == "Baseline", "IC_5d"].values[0]\n',
                '\n',
                'df_results["Delta_IC_3d"] = df_results["IC_3d"] - baseline_ic_3d\n',
                'df_results["Delta_IC_5d"] = df_results["IC_5d"] - baseline_ic_5d\n',
                '\n',
                'print("### 実験結果一覧 ###")\n',
                'print(df_results.to_string(index=False))'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 特徴量重要度の確認 (Exp-C)'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'def plot_importance(df_raw, target_col, label, **feat_kwargs):\n',
                '    df_feat = generate_features(df_raw, **feat_kwargs)\n',
                '    df_pooled = pool_boj_data(df_feat)\n',
                '    \n',
                '    # return_model=True で最終フォールドを取得\n',
                '    res, model, X_test, y_test, X_train = walk_forward_validation(\n',
                '        df_pooled, target_col, START_DATE, return_model=True\n',
                '    )\n',
                '    \n',
                '    importance = pd.DataFrame({\n',
                '        "feature": X_test.columns,\n',
                '        "importance": model.feature_importance(importance_type="gain")\n',
                '    }).sort_values("importance", ascending=False)\n',
                '    \n',
                '    plt.figure(figsize=(10, 8))\n',
                '    sns.barplot(x="importance", y="feature", data=importance.head(25))\n',
                '    plt.title(f"Feature Importance (Gain) - {label} ({target_col})")\n',
                '    plt.show()\n',
                '    \n',
                '    return importance.head(10)\n',
                '\n',
                'print("--- Exp-C Feature Importance (3d) ---")\n',
                'imp_3d = plot_importance(df_raw, "Target_3d", "Exp-C", add_curve_features=True)\n',
                'print(imp_3d)\n',
                '\n',
                'print("--- Exp-C Feature Importance (5d) ---")\n',
                'imp_5d = plot_importance(df_raw, "Target_5d", "Exp-C", add_curve_features=True)\n',
                'print(imp_5d)'
            ]
        }
    ],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {
                'name': 'ipython',
                'version': 3
            },
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.8.10'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('notebooks/07_feature_engineering_experiments.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

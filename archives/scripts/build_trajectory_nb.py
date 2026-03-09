import json

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# BOJ Swap Future Trajectory Analysis\n",
                "最新日を起点として、今後5日間にわたる各会合スワップ金利の予測推移（パス）を可視化します。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import sys\n",
                "import os\n",
                "import importlib\n",
                "from catboost import CatBoostRegressor\n",
                "\n",
                "if os.path.basename(os.getcwd()) == 'notebooks':\n",
                "    os.chdir('..')\n",
                "sys.path.append(os.getcwd())\n",
                "\n",
                "from src.processing import load_and_clean_data\n",
                "from src.pooling import pool_boj_data\n",
                "\n",
                "sns.set_theme(style='whitegrid')\n",
                "plt.rcParams['font.family'] = 'Hiragino Sans'\n",
                "print(f\"Project Root: {os.getcwd()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. モデルの学習 ($n=1 \\dots 5$)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "MAX_N = 5\n",
                "D_VAL = 0.4\n",
                "df_cleaned = load_and_clean_data('data/BOJ_data.xlsx', 'data/BOJ_meeting_history.csv')\n",
                "mpm_dates = pd.to_datetime(pd.read_csv('data/BOJ_meeting_history.csv')['Date'])\n",
                "df_pooled = pool_boj_data(df_cleaned, mpm_dates=mpm_dates, d=D_VAL, max_n=MAX_N)\n",
                "\n",
                "imputed_features = [f'M{i}_Consecutive_Imputed_Days' for i in range(1, 9)]\n",
                "features = [\n",
                "    'Swap_Rate_FracDiff', 'JGB_Future_FracDiff', 'USDJPY_FracDiff', 'Nikkei225_FracDiff', 'JPY_Effective_FracDiff',\n",
                "    'Spread_M3_M1_FracDiff', 'Spread_M5_M1_FracDiff', 'Spread_M8_M5_FracDiff',\n",
                "    'Days_to_Next_MPM', 'Meeting_Index'\n",
                "] + imputed_features\n",
                "\n",
                "models = {}\n",
                "for n in range(1, MAX_N + 1):\n",
                "    target = f'Target_FracDiff_{n}d'\n",
                "    train_final = df_pooled.dropna(subset=[target] + features)\n",
                "    m = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, verbose=0, random_seed=42)\n",
                "    m.fit(train_final[features], train_final[target])\n",
                "    models[n] = m\n",
                "    print(f\"Model n={n}d trained.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. 未来予測パスの生成\n", "最新日から5日後までの推移を計算します。"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import src.pooling\n",
                "importlib.reload(src.pooling)\n",
                "from src.pooling import get_frac_diff_weights\n",
                "weights = get_frac_diff_weights(D_VAL, 50)\n",
                "\n",
                "latest_date = df_pooled['日付'].max()\n",
                "latest_data = df_pooled[df_pooled['日付'] == latest_date].sort_values('Meeting_Index').copy()\n",
                "\n",
                "trajectory_results = []\n",
                "for idx in range(1, 9):\n",
                "    row = latest_data[latest_data['Meeting_Index'] == idx]\n",
                "    current_rate = row['Swap_Rate'].values[0]\n",
                "    path = [{'Days_Ahead': 0, 'Predicted_Rate': current_rate, 'Meeting': f'M{idx}'}]\n",
                "    \n",
                "    # 各ホライゾンの予測\n",
                "    for n in range(1, MAX_N + 1):\n",
                "        pred_fd = models[n].predict(row[features])[0]\n",
                "        # 記憶成分の近似算出 (直近のSwap_Rate系列を使用)\n",
                "        full_series = df_pooled[df_pooled['Meeting_Index'] == idx].sort_values('日付')['Swap_Rate'].values\n",
                "        mem_comp = np.sum(full_series[-(50-1):] * weights[1:][::-1])\n",
                "        pred_rate = pred_fd - mem_comp\n",
                "        path.append({'Days_Ahead': n, 'Predicted_Rate': pred_rate, 'Meeting': f'M{idx}'})\n",
                "    \n",
                "    trajectory_results.extend(path)\n",
                "\n",
                "df_traj = pd.DataFrame(trajectory_results)\n",
                "print(f\"Trajectory generated from: {latest_date.date()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. 可視化: 予測パスとカーブの変化"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(14, 8))\n",
                "target_meetings = ['M1', 'M3', 'M5', 'M8']\n",
                "sns.lineplot(data=df_traj[df_traj['Meeting'].isin(target_meetings)], \n",
                "             x='Days_Ahead', y='Predicted_Rate', hue='Meeting', marker='o', linewidth=2.5)\n",
                "plt.title(f'Future Trajectory Prediction (Starting from {latest_date.date()})', fontsize=16)\n",
                "plt.xlabel('Days from Today')\n",
                "plt.ylabel('Predicted Swap Rate (%)')\n",
                "plt.xticks(range(6))\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(12, 6))\n",
                "today_curve = df_traj[df_traj['Days_Ahead'] == 0].sort_values('Meeting')\n",
                "future_curve = df_traj[df_traj['Days_Ahead'] == 5].sort_values('Meeting')\n",
                "plt.plot(range(1, 9), today_curve['Predicted_Rate'], marker='o', label='Current Curve (Today)', color='black', linewidth=2)\n",
                "plt.plot(range(1, 9), future_curve['Predicted_Rate'], marker='s', label='Predicted Curve (t+5)', color='blue', linestyle='--', linewidth=2)\n",
                "plt.title('Yield Curve Shift Prediction (Today vs 5 Days Later)', fontsize=15)\n",
                "plt.xticks(range(1, 9), [f'M{i}' for i in range(1, 9)])\n",
                "plt.ylabel('Swap Rate (%)'); plt.legend(); plt.grid(True, alpha=0.3); plt.show()"
            ]
        }
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open('notebooks/08_future_trajectory_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

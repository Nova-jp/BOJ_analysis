import json

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Model Ablation Study: Imputation Features Impact\n",
                "欠損情報（8会合分の連続欠損日数）がモデルの予測精度に与える影響を、5つの多角的な指標で検証します。"
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
                "from catboost import CatBoostRegressor\n",
                "from sklearn.metrics import mean_squared_error\n",
                "from scipy.stats import spearmanr\n",
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
            "source": ["## 1. データ準備"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "N_DAYS = 5\n",
                "D_VAL = 0.4\n",
                "df_cleaned = load_and_clean_data('data/BOJ_data.xlsx', 'data/BOJ_meeting_history.csv')\n",
                "mpm_dates = pd.to_datetime(pd.read_csv('data/BOJ_meeting_history.csv')['Date'])\n",
                "df_pooled = pool_boj_data(df_cleaned, mpm_dates=mpm_dates, d=D_VAL, n_days=N_DAYS)\n",
                "\n",
                "base_features = [\n",
                "    'Swap_Rate_FracDiff', 'JGB_Future_FracDiff', 'USDJPY_FracDiff', 'Nikkei225_FracDiff', 'JPY_Effective_FracDiff',\n",
                "    'Spread_M3_M1_FracDiff', 'Spread_M5_M1_FracDiff', 'Spread_M8_M5_FracDiff',\n",
                "    'Days_to_Next_MPM', 'Meeting_Index'\n",
                "]\n",
                "imputed_features = [f'M{i}_Consecutive_Imputed_Days' for i in range(1, 9)]\n",
                "\n",
                "target = 'Target_FracDiff_N_Day'\n",
                "df_final = df_pooled.dropna(subset=[target, 'FracDiff_Memory_Component_N', 'Target_N_Day'] + base_features + imputed_features)\n",
                "\n",
                "split_date = sorted(df_final['日付'].unique())[int(len(df_final['日付'].unique()) * 0.8)]\n",
                "train_df = df_final[df_final['日付'] < split_date]\n",
                "test_df = df_final[df_final['日付'] >= split_date].copy()\n",
                "print(f\"Test start date: {split_date.date()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. 評価関数の定義"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def calculate_metrics(actual_rate, pred_rate, current_rate, pred_move, actual_move):\n",
                "    # 1. MSE\n",
                "    mse = mean_squared_error(actual_rate, pred_rate)\n",
                "    \n",
                "    # 2. Directional Accuracy\n",
                "    hit = (np.sign(pred_move) == np.sign(actual_move)).astype(int)\n",
                "    acc = hit.mean()\n",
                "    \n",
                "    # 3. Magnitude Score (y/x if same direction, -|y/x| if opposite)\n",
                "    eps = 1e-6\n",
                "    mag_score = np.where(np.sign(pred_move) == np.sign(actual_move), \n",
                "                         np.abs(actual_move) / (np.abs(pred_move) + eps),\n",
                "                         -np.abs(actual_move) / (np.abs(pred_move) + eps))\n",
                "    mag_score = np.clip(mag_score, -10, 10).mean()\n",
                "    \n",
                "    # 4. Information Coefficient (IC)\n",
                "    ic, _ = spearmanr(pred_move, actual_move)\n",
                "    \n",
                "    # 5. Profit Factor\n",
                "    pnl = np.sign(pred_move) * actual_move\n",
                "    gross_profits = pnl[pnl > 0].sum()\n",
                "    gross_losses = np.abs(pnl[pnl < 0].sum())\n",
                "    pf = gross_profits / gross_losses if gross_losses > 0 else np.nan\n",
                "    \n",
                "    return {\n",
                "        'MSE': mse, \n",
                "        'Accuracy': acc, \n",
                "        'Magnitude_Score': mag_score, \n",
                "        'IC': ic, \n",
                "        'Profit_Factor': pf\n",
                "    }"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. モデル比較の実行"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "results = []\n",
                "for name, feat_list in [('Baseline', base_features), ('Imputation-Aware', base_features + imputed_features)]:\n",
                "    model = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, verbose=0, random_seed=42)\n",
                "    model.fit(train_df[feat_list], train_df[target])\n",
                "    \n",
                "    preds_fd = model.predict(test_df[feat_list])\n",
                "    test_df['Pred_Rate'] = preds_fd - test_df['FracDiff_Memory_Component_N']\n",
                "    \n",
                "    actual_move = test_df['Target_N_Day'] - test_df['Swap_Rate']\n",
                "    pred_move = test_df['Pred_Rate'] - test_df['Swap_Rate']\n",
                "    \n",
                "    m = calculate_metrics(test_df['Target_N_Day'], test_df['Pred_Rate'], test_df['Swap_Rate'], pred_move, actual_move)\n",
                "    m['Model'] = name\n",
                "    results.append(m)\n",
                "\n",
                "df_results = pd.DataFrame(results).set_index('Model')\n",
                "display(df_results)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. 可視化による比較"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n",
                "metrics_to_plot = ['Accuracy', 'Magnitude_Score', 'IC', 'Profit_Factor']\n",
                "for i, metric in enumerate(metrics_to_plot):\n",
                "    ax = axes[i//2, i%2]\n",
                "    df_results[metric].plot(kind='bar', ax=ax, color=['lightgrey', 'skyblue'])\n",
                "    ax.set_title(metric, fontsize=14)\n",
                "    ax.set_xticklabels(df_results.index, rotation=0)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open('notebooks/06_imputation_ablation_study.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

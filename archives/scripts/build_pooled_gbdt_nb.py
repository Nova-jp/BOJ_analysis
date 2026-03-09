import json

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# BOJ Swap Pooled GBDT: Full Integrated Analysis\n",
                "全期間を対象に、分数階差・断面スプレッド・8会合分の連続欠損日数を特徴量に用いた包括的分析です。"
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
                "from sklearn.metrics import mean_squared_error\n",
                "\n",
                "if os.path.basename(os.getcwd()) == 'notebooks':\n",
                "    os.chdir('..')\n",
                "sys.path.append(os.getcwd())\n",
                "\n",
                "from src.processing import load_and_clean_data\n",
                "import src.pooling\n",
                "importlib.reload(src.pooling)\n",
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
            "source": ["## 1. データロードと特徴量生成"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "N_DAYS = 5\n",
                "D_VAL = 0.4\n",
                "\n",
                "df_cleaned = load_and_clean_data('data/BOJ_data.xlsx', 'data/BOJ_meeting_history.csv')\n",
                "mpm_dates_raw = pd.to_datetime(pd.read_csv('data/BOJ_meeting_history.csv')['Date'])\n",
                "\n",
                "df_pooled = pool_boj_data(df_cleaned, mpm_dates=mpm_dates_raw, d=D_VAL, max_n=N_DAYS)\n",
                "\n",
                "imputed_features = [f'M{i}_Consecutive_Imputed_Days' for i in range(1, 9)]\n",
                "features = [\n",
                "    'Swap_Rate_FracDiff',\n",
                "    'JGB_Future_FracDiff',\n",
                "    'USDJPY_FracDiff',\n",
                "    'Nikkei225_FracDiff',\n",
                "    'JPY_Effective_FracDiff',\n",
                "    'Spread_M3_M1_FracDiff',\n",
                "    'Spread_M5_M1_FracDiff',\n",
                "    'Spread_M8_M5_FracDiff',\n",
                "    'Days_to_Next_MPM',\n",
                "    'Meeting_Index'\n",
                "] + imputed_features\n",
                "\n",
                "target = f'Target_FracDiff_{N_DAYS}d'\n",
                "df_final = df_pooled.dropna(subset=[target, f'Memory_Comp_{N_DAYS}d'] + features)\n",
                "print(f\"Dataset Shape: {df_final.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. 特徴量重要度"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "split_date = sorted(df_final['日付'].unique())[int(len(df_final['日付'].unique()) * 0.8)]\n",
                "train_df = df_final[df_final['日付'] < split_date]\n",
                "test_df = df_final[df_final['日付'] >= split_date].copy()\n",
                "\n",
                "model = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, verbose=0, random_seed=42)\n",
                "model.fit(train_df[features], train_df[target])\n",
                "\n",
                "importances = pd.Series(model.get_feature_importance(), index=features).sort_values(ascending=False)\n",
                "plt.figure(figsize=(10, 10))\n",
                "sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette='viridis', legend=False)\n",
                "plt.title('Feature Importances')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. 予測推移の比較 (M1, M3, M5, M5-M1)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "test_df['Pred_Rate'] = model.predict(test_df[features]) - test_df[f'Memory_Comp_{N_DAYS}d']\n",
                "\n",
                "fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)\n",
                "target_indices = [1, 3, 5]\n",
                "for i, idx in enumerate(target_indices):\n",
                "    subset = test_df[test_df['Meeting_Index'] == idx].sort_values('日付')\n",
                "    axes[i].plot(subset['日付'], subset[f'Actual_Rate_{N_DAYS}d'], label='Actual', color='black', alpha=0.5)\n",
                "    axes[i].plot(subset['日付'], subset['Pred_Rate'], label='Pred', color='blue', linestyle='--')\n",
                "    axes[i].set_title(f'M{idx} Prediction ({N_DAYS}d ahead)')\n",
                "    axes[i].legend()\n",
                "\n",
                "m1 = test_df[test_df['Meeting_Index'] == 1][['日付', f'Actual_Rate_{N_DAYS}d', 'Pred_Rate']].rename(columns={f'Actual_Rate_{N_DAYS}d': 'M1_A', 'Pred_Rate': 'M1_P'})\n",
                "m5 = test_df[test_df['Meeting_Index'] == 5][['日付', f'Actual_Rate_{N_DAYS}d', 'Pred_Rate']].rename(columns={f'Actual_Rate_{N_DAYS}d': 'M5_A', 'Pred_Rate': 'M5_P'})\n",
                "slope = pd.merge(m1, m5, on='日付')\n",
                "axes[3].plot(slope['日付'], slope['M5_A'] - slope['M1_A'], label='Actual M5-M1', color='green')\n",
                "axes[3].plot(slope['日付'], slope['M5_P'] - slope['M1_P'], label='Pred M5-M1', color='orange', linestyle='--')\n",
                "axes[3].set_title('M5-M1 Spread Prediction'); axes[3].legend()\n",
                "plt.tight_layout(); plt.show()"
            ]
        }
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open('notebooks/04_pooled_gbdt_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

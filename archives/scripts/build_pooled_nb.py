import json

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# BOJ Swap Data Pooling Analysis\n", "BOJスワップデータを横持ちから縦積みのパネルデータ形式に変換し、その構造を確認・可視化します。"]
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
                "sys.path.append(os.getcwd())\n",
                "from src.pooling import pool_boj_data\n",
                "sns.set_theme(style='whitegrid')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# データの読み込み\n",
                "df_cleaned = pd.read_csv('data/cleaned_data.csv')\n",
                "df_cleaned['日付'] = pd.to_datetime(df_cleaned['日付'])\n",
                "print(f'Original Wide Format Shape: {df_cleaned.shape}')\n",
                "df_cleaned.head()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# プーリング処理の実行\n",
                "df_pooled = pool_boj_data(df_cleaned)\n",
                "print(f'Pooled Long Format Shape: {df_pooled.shape}')\n",
                "df_pooled.head(10)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('--- Data Structure Overview ---')\n",
                "print(df_pooled.info())\n",
                "print('\\n--- Unique Meeting Indices ---')\n",
                "print(df_pooled['Meeting_Index'].unique())\n",
                "print('\\n--- Missing Values Count ---')\n",
                "print(df_pooled.isnull().sum())\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 可視化: Meeting_IndexごとのSwap_Rate推移\n",
                "plt.figure(figsize=(14, 8))\n",
                "sns.lineplot(data=df_pooled, x='日付', y='Swap_Rate', hue='Meeting_Index', palette='viridis', alpha=0.7)\n",
                "plt.title('BOJ Swap Rates by Meeting Index (M1-M8)', fontsize=15)\n",
                "plt.ylabel('Swap Rate (%)')\n",
                "plt.xlabel('Date')\n",
                "plt.legend(title='Meeting Index', bbox_to_anchor=(1.05, 1), loc='upper left')\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 可視化: Meeting_IndexごとのSwap_Spread推移\n",
                "plt.figure(figsize=(14, 8))\n",
                "sns.lineplot(data=df_pooled, x='日付', y='Swap_Spread', hue='Meeting_Index', palette='plasma', alpha=0.7)\n",
                "plt.title('BOJ Swap Spread by Meeting Index (M1-M8)', fontsize=15)\n",
                "plt.ylabel('Spread (vs Actual Policy Rate)')\n",
                "plt.xlabel('Date')\n",
                "plt.legend(title='Meeting Index', bbox_to_anchor=(1.05, 1), loc='upper left')\n",
                "plt.show()\n"
            ]
        }
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open('boj_pooled_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

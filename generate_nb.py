import json

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# BOJ Swap Analysis\n", "日銀会合日程を紐付けた分析用データの作成。"]
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
                "sns.set_theme(style='whitegrid')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.read_excel('data/BOJ_data.xlsx')\n",
                "df = df.iloc[1:].copy()\n",
                "df['日付'] = pd.to_datetime(df['日付'], format='%Y年%m月%d日')\n",
                "for col in df.columns:\n",
                "    if col != '日付':\n",
                "        df[col] = pd.to_numeric(df[col], errors='coerce')\n",
                "df = df.sort_values('日付').reset_index(drop=True)\n",
                "print(f'Loaded {len(df)} rows')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "mpm_dates = pd.to_datetime([\n",
                "    '2024-01-23', '2024-03-19', '2024-04-26', '2024-06-14', '2024-07-31', '2024-09-20', '2024-10-31', '2024-12-19',\n",
                "    '2025-01-24', '2025-03-19', '2025-04-30', '2025-06-17', '2025-07-31', '2025-09-19', '2025-10-30', '2025-12-19',\n",
                "    '2026-01-23', '2026-03-19', '2026-04-28', '2026-06-16', '2026-07-31', '2026-09-18', '2026-10-30', '2026-12-18'\n",
                "])\n",
                "\n",
                "def get_next_mpm(d):\n",
                "    f = mpm_dates[mpm_dates > d]\n",
                "    return f[0] if len(f) > 0 else None\n",
                "\n",
                "df['Next_MPM'] = df['日付'].apply(get_next_mpm)\n",
                "df['DaysToNext'] = (df['Next_MPM'] - df['日付']).dt.days\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "target_col = 'JPBOJ1ONI=TRDT (BID)'\n",
                "for i in range(1, 6):\n",
                "    df[f'Target_{i}d'] = df[target_col].shift(-i)\n",
                "df.dropna(subset=['Target_5d'], inplace=True)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(15, 6))\n",
                "plt.plot(df['日付'], df['JPBOJ1ONI=TRDT (BID)'], label='BOJ_1')\n",
                "for m in mpm_dates:\n",
                "    if df['日付'].min() <= m <= df['日付'].max():\n",
                "        plt.axvline(m, color='r', linestyle='--', alpha=0.3)\n",
                "plt.legend()\n",
                "plt.show()\n"
            ]
        }
    ],
    "metadata": {\"kernelspec\": {\"display_name\": \"Python 3\", \"name\": \"python3\"}},\n",
    "nbformat": 4,\n",
    "nbformat_minor": 4\n",
}

with open('boj_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

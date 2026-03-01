import json

def build():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# BOJ Swap Analysis V2
", "
", "最新のデータ形式（MID_PRICE）とTONAレートを用い、翌日の金利織り込みを予測します。"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd
",
                "import numpy as np
",
                "import matplotlib.pyplot as plt
",
                "import seaborn as sns
",
                "import lightgbm as lgb
",
                "from sklearn.model_selection import TimeSeriesSplit
",
                "from sklearn.metrics import mean_absolute_error, mean_squared_error
",
                "
",
                "sns.set_theme(style='whitegrid')
",
                "import matplotlib
",
                "matplotlib.rcParams['axes.unicode_minus'] = False"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.read_excel('data/BOJ_data.xlsx')
",
                "df = df.iloc[1:].copy()
",
                "df['日付'] = pd.to_datetime(df['日付'], format='%Y年%m月%d日')
",
                "
",
                "swap_cols = [f'JPBOJ{i}ONI=TRDT (MID_PRICE)' for i in range(1, 9)]
",
                "tona_col = 'JPY1DOIS=ICAP (MID_PRICE)'
",
                "jpy_col = 'JPY= (MID_PRICE)'
",
                "market_cols = ['JGBc1 (TRDPRC_1)', '.N225 (TRDPRC_1)', '.DXY (TRDPRC_1)']
",
                "all_value_cols = swap_cols + [tona_col, jpy_col] + market_cols
",
                "
",
                "for col in all_value_cols:
",
                "    df[col] = pd.to_numeric(df[col], errors='coerce')
",
                "
",
                "df = df.sort_values('日付').reset_index(drop=True)
",
                "df.dropna(subset=[swap_cols[0], tona_col], inplace=True)
",
                "print(f'Loaded {len(df)} rows')
",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "mpm_dates = pd.to_datetime([
",
                "    '2024-01-23', '2024-03-19', '2024-04-26', '2024-06-14', '2024-07-31', '2024-09-20', '2024-10-31', '2024-12-19',
",
                "    '2025-01-24', '2025-03-19', '2025-04-30', '2025-06-17', '2025-07-31', '2025-09-19', '2025-10-30', '2025-12-19',
",
                "    '2026-01-23', '2026-03-19', '2026-04-28', '2026-06-16', '2026-07-31', '2026-09-18', '2026-10-30', '2026-12-18'
",
                "])
",
                "def get_next_mpm(d):
",
                "    future = mpm_dates[mpm_dates > d]
",
                "    return future[0] if len(future) > 0 else None
",
                "df['Next_MPM'] = df['日付'].apply(get_next_mpm)
",
                "df['DaysToNextMPM'] = (df['Next_MPM'] - df['日付']).dt.days"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "features = []
",
                "df_feats = df.copy()
",
                "for col in all_value_cols:
",
                "    df_feats[f'{col}_lag1'] = df_feats[col].shift(1)
",
                "    features.append(f'{col}_lag1')
",
                "    for window in [5, 10, 20]:
",
                "        ma = df_feats[col].shift(1).rolling(window=window).mean()
",
                "        name = f'{col}_diff_MA{window}'
",
                "        df_feats[name] = df_feats[f'{col}_lag1'] - ma
",
                "        features.append(name)
",
                "features.append('DaysToNextMPM')
",
                "df_feats['Target'] = df_feats[swap_cols[0]].shift(-1)
",
                "df_feats.dropna(inplace=True)
",
                "print(f'Features: {len(features)}')
",
                "df_feats[features + ['Target']].head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X = df_feats[features]
",
                "y = df_feats['Target']
",
                "tscv = TimeSeriesSplit(n_splits=5)
",
                "scores = []
",
                "for tr, te in tscv.split(X):
",
                "    X_train, X_test = X.iloc[tr], X.iloc[te]
",
                "    y_train, y_test = y.iloc[tr], y.iloc[te]
",
                "    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, verbosity=-1)
",
                "    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', callbacks=[lgb.early_stopping(stopping_rounds=50)])
",
                "    scores.append(mean_absolute_error(y_test, model.predict(X_test)))
",
                "print(f'MAE: {np.mean(scores):.5f}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "imp = pd.DataFrame({'f': features, 'i': model.feature_importances_}).sort_values('i', ascending=False).head(15)
",
                "plt.figure(figsize=(10, 6))
",
                "sns.barplot(x='i', y='f', data=imp)
",
                "plt.show()"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (boj-analysis)", "name": "boj-analysis"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open('boj_v2_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

if __name__ == '__main__':
    build()

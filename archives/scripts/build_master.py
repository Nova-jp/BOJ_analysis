import json

def build():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# BOJ Swap & Trend Analysis: Master Notebook
",
                "
",
                "このノートブックは、これまでの要件をすべて統合した決定版です。
",
                "
",
                "### 分析のポイント
",
                "- **ターゲット**: BOJ_3スワップの **『5日後の変化幅』**。
",
                "- **特徴量**: 全8年限のスプレッド（vs TONA）の水準と5日モメンタム、外部指標の5日モメンタム。
",
                "- **シナリオ**: 次回会合の利上げ有無（リーク特徴量）を含めたシミュレーション。
",
                "- **前処理**: 欠損値の `ffill` 補完、および MPM後5日間のノイズ除去。"
            ]
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
                "matplotlib.rcParams['axes.unicode_minus'] = False
",
                "import warnings
",
                "warnings.filterwarnings('ignore')
",
                "pd.set_option('display.max_columns', 100)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. データの読み込みと補完
",
                "df_raw = pd.read_excel('data/BOJ_data.xlsx')
",
                "df = df_raw.iloc[1:].copy()
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
                "all_val_cols = swap_cols + [tona_col, jpy_col] + market_cols
",
                "
",
                "for col in all_val_cols:
",
                "    df[col] = pd.to_numeric(df[col], errors='coerce')
",
                "
",
                "df = df.sort_values('日付').reset_index(drop=True)
",
                "df[all_val_cols] = df[all_val_cols].ffill()
",
                "
",
                "df.dropna(subset=[swap_cols[0], tona_col], inplace=True)
",
                "print(f'Loaded: {len(df)} days')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. MPMカレンダーとシナリオフラグ
",
                "mpm_results = {
",
                "    '2024-01-23': 0, '2024-03-19': 1, '2024-04-26': 0, '2024-06-14': 0, 
",
                "    '2024-07-31': 1, '2024-09-20': 0, '2024-10-31': 0, '2024-12-19': 0, 
",
                "    '2025-01-24': 1, '2025-03-19': 0, '2025-04-30': 0, '2025-06-17': 0, 
",
                "    '2025-07-31': 0, '2025-09-19': 0, '2025-10-30': 0, '2025-12-19': 0,
",
                "    '2026-01-23': 0, '2026-03-19': 0, '2026-04-28': 0, '2026-06-16': 0
",
                "}
",
                "mpm_dates = pd.to_datetime(list(mpm_results.keys()))
",
                "
",
                "def get_next_mpm_info(d):
",
                "    future = mpm_dates[mpm_dates > d]
",
                "    if len(future) > 0:
",
                "        nxt = future[0]
",
                "        return nxt, mpm_results[nxt.strftime('%Y-%m-%d')]
",
                "    return None, None
",
                "
",
                "info = df['日付'].apply(get_next_mpm_info)
",
                "df['Next_MPM'] = [x[0] for x in info]
",
                "df['Next_MPM_Hike'] = [x[1] for x in info]
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
                "# 3. 特徴量生成
",
                "features = []
",
                "df_model = df.copy()
",
                "for i, col in enumerate(swap_cols):
",
                "    spread_name = f'BOJ{i+1}_Spread'
",
                "    df_model[spread_name] = df_model[col] - df_model[tona_col]
",
                "    lag1 = f'{spread_name}_lag1'
",
                "    df_model[lag1] = df_model[spread_name].shift(1)
",
                "    features.append(lag1)
",
                "    diff5 = f'{spread_name}_diff_MA5'
",
                "    df_model[diff5] = df_model[lag1] - df_model[spread_name].shift(1).rolling(5).mean()
",
                "    features.append(diff5)
",
                "
",
                "for col in [jpy_col] + market_cols:
",
                "    diff5 = f'{col}_diff_MA5'
",
                "    df_model[diff5] = df_model[col].shift(1) - df_model[col].shift(1).rolling(5).mean()
",
                "    features.append(diff5)
",
                "features.extend(['DaysToNextMPM', 'Next_MPM_Hike'])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. ターゲット設定と除外ルール
",
                "target_base = swap_cols[2] # BOJ_3
",
                "df_model['Target'] = df_model[target_base].shift(-5) - df_model[target_base]
",
                "
",
                "exclude = []
",
                "for mpm in mpm_dates: [exclude.append(mpm + pd.Timedelta(days=o)) for o in range(1, 6)]
",
                "
",
                "df_final = df_model[~df_model['日付'].isin(exclude)].dropna(subset=['Target']).copy()
",
                "print(f'Samples: {len(df_final)}')
",
                "display(df_final[features + ['Target']].head())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. 学習と評価
",
                "X, y = df_final[features], df_final['Target']
",
                "tscv = TimeSeriesSplit(n_splits=5)
",
                "metrics = []
",
                "for tr, te in tscv.split(X):
",
                "    X_train, X_test = X.iloc[tr], X.iloc[te]
",
                "    y_train, y_test = y.iloc[tr], y.iloc[te]
",
                "    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbosity=-1)
",
                "    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', callbacks=[lgb.early_stopping(stopping_rounds=50)])
",
                "    y_pred = model.predict(X_test)
",
                "    metrics.append({
",
                "        'MAE_bps': mean_absolute_error(y_test, y_pred)*100,
",
                "        'DirAcc': np.mean(np.sign(y_test) == np.sign(y_pred))
",
                "    })
",
                "print('
--- Results ---')
",
                "print(pd.DataFrame(metrics).mean().to_frame('Average').T)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. 可視化
",
                "imp = pd.DataFrame({'f': features, 'g': model.booster_.feature_importance(importance_type='gain')}).sort_values('g', ascending=False).head(15)
",
                "plt.figure(figsize=(10, 6))
",
                "sns.barplot(x='g', y='f', data=imp)
",
                "plt.title('Feature Importance (Gain)')
",
                "plt.show()
",
                "
",
                "fig, ax1 = plt.subplots(figsize=(15, 5))
",
                "ax1.plot(df_final['日付'], df_final['BOJ3_Spread_lag1'], color='blue', alpha=0.3)
",
                "ax2 = ax1.twinx()
",
                "ax2.bar(df_final['日付'], df_final['BOJ3_Spread_diff_MA5'], color='red', alpha=0.4)
",
                "plt.show()"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3 (boj-analysis)", "name": "boj-analysis"}},
        "nbformat": 4, "nbformat_minor": 4
    }
    with open('BOJ_Master_Analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

if __name__ == '__main__':
    build()

import pandas as pd
import numpy as np

def frac_diff(series, d, window=50):
    """
    分数階差を計算する。
    w_k = (-1)^k * C(d, k)
    windowサイズ分の過去データが揃わない先頭行はNaNとする。
    """
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights[::-1]) # 過去から現在への重み

    def apply_weights(x):
        if len(x) < window or np.isnan(x).any():
            return np.nan
        return np.dot(x, weights)

    return series.rolling(window=window).apply(apply_weights, raw=True)

def generate_features(
    df,
    d=0.4,
    window=50,
    add_weekday_cyclic=False,     # Exp-A
    add_days_to_mpm_cyclic=False, # Exp-B
    add_curve_features=False,     # Exp-C
):
    """
    設計仕様に基づき特徴量を生成する。
    """
    feat_df = df.copy()
    
    # 1. M{n}_spread = M{n} - Actual_Policy_Rate
    boj_cols = [f'M{i}' for i in range(1, 9)]
    for col in boj_cols:
        feat_df[f'{col}_spread'] = feat_df[col] - feat_df['Actual_Policy_Rate']
        
    # 2. M{n}_frac_diff
    for col in boj_cols:
        feat_df[f'{col}_frac_diff'] = frac_diff(feat_df[f'{col}_spread'], d=d, window=window)
        
    # 3. 外部指標の分数階差
    ext_cols = ['USDJPY', 'JGB_Future', 'Nikkei225']
    for col in ext_cols:
        feat_df[f'{col}_frac_diff'] = frac_diff(feat_df[col], d=d, window=window)
        
    # 4. Days_to_MPM: processing.py で計算済みの場合はそのまま使う
    # （processing.py が未来の予定会合を含む CSV 全体から計算するため、こちらでは上書きしない）
    if 'Days_to_MPM' not in feat_df.columns:
        meeting_dates = sorted(feat_df.loc[feat_df['Is_Meeting_Day'] == 1, 'Date'].unique())

        def get_days_to_mpm(date):
            future = [m for m in meeting_dates if m >= date]
            return (future[0] - date).days if future else np.nan

        feat_df['Days_to_MPM'] = feat_df['Date'].map(
            {d: get_days_to_mpm(d) for d in feat_df['Date'].unique()}
        )

    # --- Exp-A: 曜日の円環エンコーディング ---
    if add_weekday_cyclic:
        feat_df['DayOfWeek'] = feat_df['Date'].dt.dayofweek  # 0〜4
        feat_df['DayOfWeek_sin'] = np.sin(2 * np.pi * feat_df['DayOfWeek'] / 5)
        feat_df['DayOfWeek_cos'] = np.cos(2 * np.pi * feat_df['DayOfWeek'] / 5)

    # --- Exp-B: Days_to_MPM の円環エンコーディング ---
    if add_days_to_mpm_cyclic:
        CYCLE = 45
        feat_df['Days_to_MPM_sin'] = np.sin(2 * np.pi * feat_df['Days_to_MPM'] / CYCLE)
        feat_df['Days_to_MPM_cos'] = np.cos(2 * np.pi * feat_df['Days_to_MPM'] / CYCLE)

    # --- Exp-C: カーブ形状特徴量 ---
    if add_curve_features:
        # 1. スロープ
        feat_df['Curve_Slope'] = feat_df['M8_spread'] - feat_df['M1_spread']
        # 2. バタフライ
        for n in range(2, 8):
            feat_df[f'Butterfly_M{n}'] = (
                2 * feat_df[f'M{n}_spread']
                - feat_df[f'M{n-1}_spread']
                - feat_df[f'M{n+1}_spread']
            )
        # 3. 分数階差
        curve_cols = ['Curve_Slope'] + [f'Butterfly_M{n}' for n in range(2, 8)]
        for col in curve_cols:
            feat_df[f'{col}_frac_diff'] = frac_diff(feat_df[col], d=d, window=window)

    return feat_df


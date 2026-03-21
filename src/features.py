import pandas as pd
import numpy as np

def frac_diff(series, d, window=50):
    """
    分数階差を計算する。
    w_k = (-1)^k * C(d, k)
    windowサイズ分の過去データが揃わない先頭行はNaNとする。

    rolling().apply() を使わず numpy ループで実装することで
    Python 関数呼び出しのオーバーヘッドを回避している。
    """
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights[::-1])  # 過去から現在への重み

    arr = series.to_numpy(dtype=float)
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        window_data = arr[i - window + 1:i + 1]
        if np.isnan(window_data).any():
            continue
        result[i] = np.dot(window_data, weights)

    return pd.Series(result, index=series.index)

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

    対象レート:
      - M1〜M8: 会合先BOJスワップ（spread = レート - 政策金利）
      - T12/T18/T24: テナーOIS（同様にspreadを計算）
    """
    feat_df = df.copy()

    # 予測対象レートの一覧（存在するもののみ）
    boj_cols = [f'M{i}' for i in range(1, 9)]
    tenor_cols = [c for c in ['T12', 'T18', 'T24'] if c in feat_df.columns]
    all_rate_cols = boj_cols + tenor_cols

    # 1. spread = レート - Actual_Policy_Rate
    for col in all_rate_cols:
        feat_df[f'{col}_spread'] = feat_df[col] - feat_df['Actual_Policy_Rate']

    # 2. frac_diff（spreadに対して適用）
    for col in all_rate_cols:
        feat_df[f'{col}_frac_diff'] = frac_diff(feat_df[f'{col}_spread'], d=d, window=window)

    # 3. 外部指標の分数階差
    ext_cols = ['USDJPY', 'JGB_Future', 'Nikkei225', 'DXY']
    for col in ext_cols:
        if col in feat_df.columns:
            feat_df[f'{col}_frac_diff'] = frac_diff(feat_df[col], d=d, window=window)

    # Days_to_MPM は processing.py の load_and_clean_data が常に付与する

    # --- Exp-A: 曜日の円環エンコーディング ---
    if add_weekday_cyclic:
        # DayOfWeek（整数）は中間変数のため DataFrame には追加しない
        _day_of_week = feat_df['Date'].dt.dayofweek  # 0〜4
        feat_df['DayOfWeek_sin'] = np.sin(2 * np.pi * _day_of_week / 5)
        feat_df['DayOfWeek_cos'] = np.cos(2 * np.pi * _day_of_week / 5)

    # --- Exp-B: Days_to_MPM の円環エンコーディング ---
    if add_days_to_mpm_cyclic:
        CYCLE = 45
        feat_df['Days_to_MPM_sin'] = np.sin(2 * np.pi * feat_df['Days_to_MPM'] / CYCLE)
        feat_df['Days_to_MPM_cos'] = np.cos(2 * np.pi * feat_df['Days_to_MPM'] / CYCLE)

    # --- Exp-C: カーブ形状特徴量（M1〜M8ベース） ---
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

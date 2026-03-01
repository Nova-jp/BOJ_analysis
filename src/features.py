import pandas as pd
import numpy as np

def frac_diff_func(series, d, window=50):
    """
    分数階差を計算する内部関数
    """
    w = [1.0]
    for k in range(1, window):
        w.append(-w[-1] * (d - k + 1) / k)
    weights = np.array(w[::-1])
    # 欠損値を含む系列に対して rolling.apply を安全に実行
    res = series.rolling(window=window).apply(
        lambda x: np.sum(x * weights) if not np.isnan(x).any() else np.nan, 
        raw=True
    )
    return res

def generate_features(df, ma_window=20, d=0.4):
    """
    既存のカラム（Is_Meeting_Day, 補完フラグ等）を維持したまま特徴量を生成する
    """
    featured_df = df.copy()
    
    # 1. BOJ Swap 特徴量
    boj_cols = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
    for col in boj_cols:
        if col in featured_df.columns:
            # スプレッド
            featured_df[f'{col}_spread'] = featured_df[col] - featured_df['Actual_Policy_Rate']
            # 分数階差
            featured_df[f'{col}_frac_diff'] = frac_diff_func(featured_df[col], d)
        
    # 2. 外部指標 特徴量
    ext_cols = ['USDJPY', 'DXY', 'JPY_Effective', 'JGB_Future', 'Nikkei225']
    for col in ext_cols:
        if col in featured_df.columns:
            # 20日移動平均乖離率
            ma = featured_df[col].rolling(window=ma_window).mean()
            featured_df[f'{col}_ma_dev'] = (featured_df[col] - ma) / ma
            # 分数階差
            featured_df[f'{col}_frac_diff'] = frac_diff_func(featured_df[col], d)
        
    # 3. 傾き
    if 'M8' in featured_df.columns and 'M1' in featured_df.columns:
        featured_df['Slope_M8_M1'] = featured_df['M8'] - featured_df['M1']
    
    return featured_df

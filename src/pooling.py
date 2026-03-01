import pandas as pd
import numpy as np
import sys
import os

# プロジェクトルートへのパス追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import frac_diff_func

def get_frac_diff_weights(d, window):
    w = [1.0]
    for k in range(1, window):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)

def pool_boj_data(df, mpm_dates=None, d=0.4, window=50, max_n=5):
    """
    BOJデータをプーリングし、1〜n日後のターゲットを作成する (Date統一・シンプル版)
    """
    df = df.copy()
    if 'Date' not in df.columns and '日付' in df.columns:
        df['Date'] = pd.to_datetime(df['日付'])
    
    boj_cols = [f'M{i}' for i in range(1, 9)]
    weights = get_frac_diff_weights(d, window)
    
    # 1. 共通指標の作成 (横持ち段階)
    if 'DXY' in df.columns and 'USDJPY' in df.columns:
        df['JPY_Effective'] = df['DXY'] / df['USDJPY']
    
    # スプレッドと外部指標の分数階差
    ext_cols = ['JGB_Future', 'USDJPY', 'Nikkei225', 'JPY_Effective']
    common_feats = []
    for col in ext_cols:
        if col in df.columns:
            df[f'{col}_FracDiff'] = frac_diff_func(df[col], d=d, window=window)
            common_feats.append(f'{col}_FracDiff')
            
    # 連続欠損日数 (8特徴量)
    for i in range(1, 9):
        c = f'M{i}_is_imputed'
        if c in df.columns:
            df[f'M{i}_Consec_Imp'] = df[c].groupby((df[c] != df[c].shift()).cumsum()).cumcount() + 1 * df[c]
            common_feats.append(f'M{i}_Consec_Imp')

    # 2. メルト (縦積み)
    id_vars = ['Date', 'Actual_Policy_Rate', 'Is_Meeting_Day'] + common_feats
    id_vars = [c for c in id_vars if c in df.columns]
    
    pooled = df[id_vars + boj_cols].melt(id_vars=id_vars, value_vars=boj_cols, var_name='M_Label', value_name='Swap_Rate')
    pooled['Meeting_Index'] = pooled['M_Label'].str.extract(r'(\d+)').astype(int)
    pooled = pooled.sort_values(['Meeting_Index', 'Date']).reset_index(drop=True)

    # 3. ホライゾン別ターゲット生成
    pooled['Swap_Rate_FracDiff'] = pooled.groupby('Meeting_Index')['Swap_Rate'].transform(lambda x: frac_diff_func(x, d=d, window=window))
    
    def calc_mem(x):
        return x.rolling(window=window-1).apply(lambda v: np.sum(v * weights[1:][::-1]), raw=True)

    for n in range(1, max_n + 1):
        pooled[f'Target_FD_{n}d'] = pooled.groupby('Meeting_Index')['Swap_Rate_FracDiff'].shift(-n)
        pooled[f'Mem_{n}d'] = pooled.groupby('Meeting_Index')['Swap_Rate'].transform(calc_mem).shift(-n)
        pooled[f'Actual_{n}d'] = pooled.groupby('Meeting_Index')['Swap_Rate'].shift(-n)

    if mpm_dates is not None:
        mpm_dates = pd.to_datetime(mpm_dates).sort_values()
        date_map = {d: (mpm_dates[mpm_dates > d].iloc[0] - d).days if any(mpm_dates > d) else np.nan for d in pooled['Date'].unique()}
        pooled['Days_to_MPM'] = pooled['Date'].map(date_map)

    return pooled.drop(columns=['M_Label'])

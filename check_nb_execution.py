import pandas as pd
import numpy as np
import sys, os, importlib
from catboost import CatBoostRegressor

# プロジェクトルートの設定
sys.path.append(os.getcwd())
from src.processing import load_and_clean_data
from src.pooling import pool_boj_data

# 1. データ準備
MAX_N = 5
df_cleaned = load_and_clean_data('data/BOJ_data.xlsx', 'data/BOJ_meeting_history.csv')
mpm_dates = pd.to_datetime(pd.read_csv('data/BOJ_meeting_history.csv')['Date'])
df_pooled = pool_boj_data(df_cleaned, mpm_dates=mpm_dates, max_n=MAX_N)

features = [c for c in df_pooled.columns if 'FracDiff' in c or 'Consec_Imp' in c or c in ['Days_to_MPM', 'Meeting_Index']]
train_final = df_pooled.dropna(subset=[f'Target_FD_{MAX_N}d'] + features)
split_date = sorted(train_final['Date'].unique())[int(len(train_final['Date'].unique()) * 0.8)]
test_df = df_pooled[df_pooled['Date'] >= split_date].copy()

# 2. モデル学習
for n in range(1, MAX_N + 1):
    m = CatBoostRegressor(iterations=100, verbose=0, random_seed=42) # 高速化のためiterations=100
    train_subset = train_final[train_final['Date'] < split_date]
    m.fit(train_subset[features], train_subset[f'Target_FD_{n}d'])
    test_df[f'Pred_Rate_{n}d'] = m.predict(test_df[features]) - test_df[f'Mem_{n}d'].ffill()

# 3. 戦略関数 (plot_analysisの中身を一部抽出)
def test_strategy(df):
    df = df.copy()
    cond_long = (df['Pred_Rate_1d'] < df['Pred_Rate_3d']) & (df['Pred_Rate_3d'] < df['Pred_Rate_5d'])
    cond_short = (df['Pred_Rate_1d'] > df['Pred_Rate_3d']) & (df['Pred_Rate_3d'] > df['Pred_Rate_5d'])
    df['Signal'] = 0
    df.loc[cond_long, 'Signal'] = 1
    df.loc[cond_short, 'Signal'] = -1
    
    # 07ノートブックで使っていた 'Actual_5d' が存在するか確認
    # pooling.py では 'Actual_5d' を作成しているはず
    df['Score'] = -1 * df['Signal'] * (df['Actual_5d'] - df['Swap_Rate']) / (np.abs(df['Pred_Rate_5d'] - df['Swap_Rate']) + 1e-6)
    return df

# 実行
try:
    m1_res = test_strategy(test_df[test_df['Meeting_Index'] == 1])
    print("Execution Success: M1 strategy result generated.")
    print(f"Columns in result: {m1_res.columns.tolist()}")
except Exception as e:
    print(f"Execution Failed: {e}")

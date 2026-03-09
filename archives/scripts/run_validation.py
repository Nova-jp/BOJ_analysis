import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import os

sys.path.append(os.getcwd())
from src.pooling import pool_boj_data

def eval_model(tr, ts, features):
    m = CatBoostRegressor(iterations=500, verbose=0, random_seed=42)
    m.fit(tr[features], tr['Target_Next_Day'])
    p = m.predict(ts[features])
    rmse = np.sqrt(mean_squared_error(ts['Target_Next_Day'], p))
    mae = mean_absolute_error(ts['Target_Next_Day'], p)
    return rmse, mae

# データ読み込み
df_cleaned = pd.read_csv('data/cleaned_data.csv')
df_cleaned['日付'] = pd.to_datetime(df_cleaned['日付'])
mpm_dates = pd.to_datetime(pd.read_csv('data/BOJ_meeting_history.csv')['Date'])

# プーリング
df_pooled = pool_boj_data(df_cleaned, mpm_dates=mpm_dates)

# フィルタリング
df_final = df_pooled[df_pooled['Post_MPM_Filter'] == 0].dropna(
    subset=['Target_Next_Day', 'Swap_Rate_MA10_Diff', 'Days_to_Next_MPM']
)

# 時系列分割
dates = sorted(df_final['日付'].unique())
split_idx = int(len(dates) * 0.8)
split_date = dates[split_idx]

train_full = df_final[df_final['日付'] < split_date]
test_m1 = df_final[(df_final['日付'] >= split_date) & (df_final['Meeting_Index'] == 1)]

features = ['Swap_Rate', 'Swap_Rate_MA10_Diff', 'Days_to_Next_MPM', 'Meeting_Index']

# 個別モデル (M1)
rmse_ind, mae_ind = eval_model(train_full[train_full['Meeting_Index'] == 1], test_m1, features)

# プーリングモデル (全会合)
rmse_pool, mae_pool = eval_model(train_full, test_m1, features)

print(f"Results for M1 Prediction (Test Data after {split_date.date()}):")
print(f"Individual Model -> RMSE: {rmse_ind:.6f}, MAE: {mae_ind:.6f}")
print(f"Pooled Model     -> RMSE: {rmse_pool:.6f}, MAE: {mae_pool:.6f}")
print(f"Improvement (RMSE): {(rmse_ind - rmse_pool) / rmse_ind * 100:.2f}%")

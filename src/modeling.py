import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prepare_modeling_data(df, target_horizon=5, exclude_post_mpm_days=5):
    """
    特徴量とターゲットを整理し、MPM直後の不安定な期間を削除する
    """
    boj_cols = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
    ext_cols = ['USDJPY', 'DXY', 'JPY_Effective', 'JGB_Future', 'Nikkei225']
    
    # 1. ターゲット(Y)の生成: 5日後の各会合スプレッド
    targets = []
    for m in boj_cols:
        target_name = f'{m}_target_5d'
        df[target_name] = df[m].shift(-target_horizon) - df['Actual_Policy_Rate']
        targets.append(target_name)
    
    # 2. 特徴量(X)の選定
    feature_cols = ([f'{m}_spread' for m in boj_cols] + 
                    [f'{c}_frac_diff' for c in ext_cols] + 
                    [f'{m}_is_imputed' for m in boj_cols] + 
                    ['Actual_Policy_Rate', 'Slope_M8_M1'])
    
    # 3. MPM直後の特定と削除
    # Is_Meeting_Day が 1 の日から5日間（当日含む）を除外対象とするフラグ
    # 過去 exclude_post_mpm_days 日以内に会合があったかどうかを判定
    df['is_post_mpm'] = df['Is_Meeting_Day'].rolling(window=exclude_post_mpm_days, min_periods=1).max()
    
    # NaNの削除とMPM直後の削除
    data = df.dropna(subset=targets + feature_cols).copy()
    data = data[data['is_post_mpm'] == 0] # 会合直後でないデータのみ抽出
    
    X = data[feature_cols]
    Y = data[targets]
    dates = data['日付']
    
    return X, Y, dates, feature_cols, targets

def train_vector_leaf_model(X, Y):
    """
    CatBoostのMultiRegressionを使用してVector-leafモデルを学習する
    """
    model = CatBoostRegressor(
        loss_function='MultiRMSE',
        eval_metric='MultiRMSE',
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )
    
    # 時系列クロスバリデーション
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]
    
    model.fit(
        X_train, Y_train,
        eval_set=(X_test, Y_test),
        use_best_model=True
    )
    
    return model, X_test, Y_test

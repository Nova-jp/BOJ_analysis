import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

def get_features_and_target(df, target_col):
    """
    特徴量とターゲットを分離し、NaNを削除する。
    """
    # 設計書に基づく特徴量リスト
    boj_spread_cols = [f'M{i}_spread' for i in range(1, 9)]
    boj_fd_cols = [f'M{i}_frac_diff' for i in range(1, 9)]
    boj_imp_cols = [f'M{i}_is_imputed' for i in range(1, 9)]
    ext_fd_cols = ['USDJPY_frac_diff', 'JGB_Future_frac_diff', 'Nikkei225_frac_diff']
    context_cols = ['Meeting_Index', 'Days_to_MPM', 'Actual_Policy_Rate']
    
    base_cols = context_cols + boj_spread_cols + boj_fd_cols + boj_imp_cols + ext_fd_cols
    
    # 追加特徴量（設計書 04: 存在する場合のみ追加）
    optional_cols = (
        ['DayOfWeek_sin', 'DayOfWeek_cos',
         'Days_to_MPM_sin', 'Days_to_MPM_cos',
         'Curve_Slope'] +
        [f'Butterfly_M{n}' for n in range(2, 8)] +
        ['Curve_Slope_frac_diff'] +
        [f'Butterfly_M{n}_frac_diff' for n in range(2, 8)]
    )
    
    feature_cols = base_cols + [c for c in optional_cols if c in df.columns]
    
    # 必要データの抽出
    data = df[feature_cols + [target_col, 'Date', 'is_post_mpm']].copy()
    
    # MPM直後およびターゲット/特徴量のNaNを除外
    data = data[data['is_post_mpm'] == 0]
    data = data.dropna(subset=feature_cols + [target_col])
    
    return data[feature_cols], data[target_col], data['Date'], data['Meeting_Index']

def train_model(X_train, y_train, X_val, y_val, categorical_feature=['Meeting_Index']):
    """
    LightGBMモデルを訓練する。

    early stopping は使わない。
    理由：金融時系列の非定常性（例：ゼロ金利期 → 利上げ期）により
    バリデーション窓との分布ずれが生じ、best_iteration=1 で止まるフォールドが
    多発して IC が不安定になるため。
    代わりに固定ラウンド数 + 正則化で汎化を担保する。
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 30,  # 過学習抑制（20→30）
        'lambda_l2': 1.0,        # L2正則化（early stopping 廃止の代替）
    }

    cat_features = [c for c in categorical_feature if c in X_train.columns]

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_features)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['valid'],
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    return model

def calculate_metrics(y_true, y_pred):
    """
    設計書に基づく評価指標を計算する。
    """
    if len(y_true) < 2:
        return {'IC': np.nan, 'RMSE': np.nan, 'Direction_Accuracy': np.nan, 'Direction_Accuracy_LargeMove': np.nan}
        
    metrics = {}
    
    # IC (Spearman)
    ic, _ = spearmanr(y_true, y_pred)
    metrics['IC'] = ic
    
    # MSE / RMSE
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 方向的中率 (全体)
    y_true_sign = np.sign(y_true)
    y_pred_sign = np.sign(y_pred)
    correct_dir = (y_true_sign == y_pred_sign).sum()
    metrics['Direction_Accuracy'] = correct_dir / len(y_true)
    
    # 方向的中率 (大動き限定: 上位25%)
    threshold = np.percentile(np.abs(y_true), 75)
    large_move_idx = np.abs(y_true) > threshold
    if large_move_idx.any():
        correct_dir_large = (y_true_sign[large_move_idx] == y_pred_sign[large_move_idx]).sum()
        metrics['Direction_Accuracy_LargeMove'] = correct_dir_large / large_move_idx.sum()
    else:
        metrics['Direction_Accuracy_LargeMove'] = np.nan
        
    return metrics

def walk_forward_validation(df, target_col, start_date, test_window_days=90, purge_days=5, return_model=False):
    """
    パージ付きウォークフォワード検証。
    """
    X, y, dates, m_idx = get_features_and_target(df, target_col)
    
    unique_dates = sorted(dates.unique())
    test_start_idx = 0
    for i, d in enumerate(unique_dates):
        if d >= pd.to_datetime(start_date):
            test_start_idx = i
            break
    
    if test_start_idx == 0:
        test_start_idx = 1
            
    results = []
    
    current_idx = test_start_idx
    fold = 0
    last_model = None
    last_X_test = None
    last_y_test = None
    last_X_train = None

    while current_idx < len(unique_dates):
        train_end_date = unique_dates[current_idx - 1]
        test_start_date = unique_dates[current_idx]
        
        # パージ
        actual_train_end_date = train_end_date - pd.Timedelta(days=purge_days)
        
        # 訓練/テストデータの分割
        train_mask = dates <= actual_train_end_date
        test_mask = (dates >= test_start_date) & (dates < test_start_date + pd.Timedelta(days=test_window_days))
        
        if not test_mask.any():
            break
            
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_dates_fold = dates[test_mask]
        test_m_idx_fold = m_idx[test_mask]
        
        # バリデーションセットの生成 (Task A 修正)
        train_dates_fold = dates[train_mask]
        val_unique_dates = sorted(train_dates_fold.unique())
        
        if len(val_unique_dates) < 10: # 訓練データが少なすぎる場合
            break
            
        # 300日以上なら60日、未満なら15%
        if len(val_unique_dates) >= 300:
            val_days_count = 60
        else:
            val_days_count = max(1, int(len(val_unique_dates) * 0.15))
            
        val_threshold = val_unique_dates[-val_days_count]
        val_mask = train_dates_fold >= val_threshold
        
        X_tr, y_tr = X_train[~val_mask], y_train[~val_mask]
        X_val, y_val = X_train[val_mask], y_train[val_mask]
        
        model = train_model(X_tr, y_tr, X_val, y_val)
        
        # 予測
        test_preds = model.predict(X_test)
        train_preds = model.predict(X_train)
        
        # 訓練データの評価用
        train_metrics = calculate_metrics(y_train, train_preds)
        
        fold_res = pd.DataFrame({
            'Fold': fold,
            'Date': test_dates_fold,
            'Meeting_Index': test_m_idx_fold,
            'Actual': y_test,
            'Pred': test_preds,
            'Train_IC': train_metrics['IC']
        })
        results.append(fold_res)
        
        # 最終フォールドの情報を保持 (Task C 用)
        last_model = model
        last_X_test = X_test
        last_y_test = y_test
        last_X_train = X_train
        
        fold += 1
        # 次のフォールドへ
        next_date = test_start_date + pd.Timedelta(days=test_window_days)
        for i, d in enumerate(unique_dates):
            if d >= next_date:
                current_idx = i
                break
        else:
            break
            
    if not results:
        if return_model:
            return pd.DataFrame(), None, None, None, None
        return pd.DataFrame()
        
    final_results = pd.concat(results).reset_index(drop=True)
    if return_model:
        return final_results, last_model, last_X_test, last_y_test, last_X_train
    return final_results

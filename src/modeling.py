import bisect
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

# 特徴量として使わない列（メタデータ・ターゲット系）
# ここに列名を追加するだけでモデルから除外される。
# 特徴量を追加する際はこのリストに触れる必要はない（除外リスト方式）。
NON_FEATURE_COLS = {
    # 識別・メタデータ
    'Date', 'Rate_Label', 'Rate_Value', 'is_post_mpm',
    # ターゲット（全バリアント）
    'Target_1d',
    'Target_3d', 'Target_3d_norm', 'Target_3d_std',
    'Target_5d', 'Target_5d_norm', 'Target_5d_std',
}

# カテゴリ変数（LightGBM に明示する列）
# 特徴量リストと異なり、カテゴリの意味は安定しているためここで管理する。
CATEGORICAL_COLS = ['Meeting_Index', 'Is_Tenor_OIS', 'Absolute_Meeting_ID']

# M1〜M8 の Meeting_Index 値（CS IC 計算で BOJ 会合先のみを対象とするために使用）
BOJ_MEETING_INDICES = set(range(1, 9))


def get_features_and_target(df, target_col):
    """
    特徴量とターゲットを分離し、NaNを削除する。

    特徴量は NON_FEATURE_COLS に含まれない全列を自動的に採用する（除外リスト方式）。
    これにより、features.py や pooling.py で新しい列を追加した際に
    modeling.py を修正し忘れるサイレントバグを防ぐ。
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and c != target_col]

    # 必要データの抽出
    # Date と is_post_mpm は NON_FEATURE_COLS なので feature_cols に含まれない → 明示的に追加
    # Meeting_Index は feature_cols に含まれる → 重複追加しない
    data = df[feature_cols + [target_col, 'Date', 'is_post_mpm']].copy()

    # ターゲット/特徴量のNaNを除外
    # 注意: MPM直後5日間の除外は行わない（post-MPM期間のCS ICが高いため、2026-03-23にModel Bをメインに昇格）
    data = data.dropna(subset=feature_cols + [target_col])

    return data[feature_cols], data[target_col], data['Date'], data['Meeting_Index']


def train_model(X_train, y_train, categorical_feature=None, num_boost_round=100):
    """
    LightGBMモデルを訓練する。

    early stopping は使わない。
    理由：金融時系列の非定常性（例：ゼロ金利期 → 利上げ期）により
    バリデーション窓との分布ずれが生じ、best_iteration=1 で止まるフォールドが
    多発して IC が不安定になるため。
    代わりに固定ラウンド数 + 正則化で汎化を担保する。

    val_data を lgb.train() に渡さない理由：
    固定ラウンドかつ early stopping なしの場合、LightGBM は valid_sets を渡しても
    全イテレーションで val metric を計算し続ける（表示はしなくても）。
    これは無駄な計算コストになるため、val_data は渡さない。

    Args:
        num_boost_round: 学習ラウンド数。デフォルト100。
            感度分析（designs/archives/06_rounds_sensitivity.md）により、
            100ラウンドで3d/5dともOOS ICが最大かつGapが最小（≈0.10）と判明。
    """
    if categorical_feature is None:
        categorical_feature = CATEGORICAL_COLS

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
        'min_data_in_leaf': 30,
        'lambda_l2': 1.0,
    }

    cat_features = [c for c in categorical_feature if c in X_train.columns]

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
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

    # RMSE
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


def summarize_ic(results: pd.DataFrame, recent_n_folds: int = 3,
                 instrument_indices=None) -> dict:
    """
    walk_forward_validation の結果から IC サマリーを計算する。

    3種類の IC を返す:
    - Global IC:  全 (日付 × 銘柄) をまとめた Spearman 相関。時系列+横断面の混合。
    - CS IC:      Cross-Sectional IC。各日付で M1〜M8 をランク付けし日付平均。
                  相対価値戦略の評価に最適。市場の方向感（トレンド）の影響を受けない。
    - TS IC:      Time-Series IC。各銘柄（Meeting_Index）の時系列 IC を銘柄平均。
                  トレンド相場では実力以上に高くなりやすい（診断用）。

    Global IC と CS IC の差が大きい場合、アルファの大部分がトレンドのベータである可能性を示す。

    Args:
        results:            walk_forward_validation の返り値
        recent_n_folds:     直近フォールドとして扱うフォールド数
        instrument_indices: CS IC の計算に使う Meeting_Index の集合。
                            None → BOJ_MEETING_INDICES（M1-M8）を使用。
                            RV Curve モデル: set(range(1, 8))
                            RV Butterfly モデル: set(range(2, 8))

    Returns:
        dict with keys:
            ic_all:      Global OOS IC（Spearman、全フォールド合算）
            ic_by_fold:  フォールド番号 → Global IC の dict
            ic_recent:   直近 N フォールドの Global IC
            cs_ic:       Cross-Sectional IC（日付ごとに instrument_indices をランク付け、日付平均）
            ts_ic:       Time-Series IC（全銘柄の時系列 IC、銘柄平均）
            train_ic:    フォールド平均 Train IC
            gap:         train_ic - ic_all
    """
    if results.empty:
        return {'ic_all': np.nan, 'ic_by_fold': {}, 'ic_recent': np.nan,
                'cs_ic': np.nan, 'ts_ic': np.nan,
                'train_ic': np.nan, 'gap': np.nan}

    # --- Global IC ---
    ic_all, _ = spearmanr(results['Actual'], results['Pred'])

    folds = sorted(results['Fold'].unique())
    ic_by_fold = {}
    for f in folds:
        sub = results[results['Fold'] == f]
        if len(sub) >= 2:
            ic, _ = spearmanr(sub['Actual'], sub['Pred'])
            ic_by_fold[f] = round(ic, 4)

    recent_folds = folds[-recent_n_folds:]
    recent_data  = results[results['Fold'].isin(recent_folds)]
    ic_recent = spearmanr(recent_data['Actual'], recent_data['Pred'])[0] if len(recent_data) >= 2 else np.nan

    train_ic = results['Train_IC'].mean()

    # --- CS IC（Cross-Sectional IC）---
    # 各日付で対象銘柄をランク付けし、Spearman IC を計算して日付平均。
    # instrument_indices が None の場合は BOJ_MEETING_INDICES（M1-M8）を使用。
    # RV モデルでは全 Pair_Index / Butterfly_Index が対象なので None を渡すと
    # 全 Meeting_Index が使われる（RV モデルには M1-M8 以外の行が存在しない）。
    _cs_indices = instrument_indices if instrument_indices is not None else BOJ_MEETING_INDICES
    boj_results = results[results['Meeting_Index'].isin(_cs_indices)]
    cs_ics = []
    for _, grp in boj_results.groupby('Date'):
        if len(grp) >= 3:  # 最低3銘柄ないと相関は不安定
            ic, _ = spearmanr(grp['Actual'], grp['Pred'])
            if not np.isnan(ic):
                cs_ics.append(ic)
    cs_ic = float(np.mean(cs_ics)) if cs_ics else np.nan

    # --- TS IC（Time-Series IC）---
    # 各銘柄（Meeting_Index）の時系列 Spearman IC を計算して銘柄平均。
    # トレンド相場では高くなりやすいため診断用として使用する。
    ts_ics = []
    for _, grp in results.groupby('Meeting_Index'):
        if len(grp) >= 5:
            ic, _ = spearmanr(grp['Actual'], grp['Pred'])
            if not np.isnan(ic):
                ts_ics.append(ic)
    ts_ic = float(np.mean(ts_ics)) if ts_ics else np.nan

    return {
        'ic_all':     round(ic_all, 4),
        'ic_by_fold': ic_by_fold,
        'ic_recent':  round(ic_recent, 4),
        'cs_ic':      round(cs_ic, 4),
        'ts_ic':      round(ts_ic, 4),
        'train_ic':   round(train_ic, 4),
        'gap':        round(train_ic - ic_all, 4),
    }


def _walk_forward_core(df, target_col, start_date, test_window_days=90, purge_days=5, num_boost_round=100):
    """
    ウォークフォワード検証の共通実装。walk_forward_validation と walk_forward_with_model から呼ばれる。

    Returns:
        results (list[DataFrame]): フォールドごとの予測結果リスト
        last_model: 最終フォールドのモデル
        last_X_test, last_y_test, last_X_train: 最終フォールドのデータ

    注意: 5d ターゲットでは末尾5日分の OOS 行が生成されない（shift(-5) によるNaN除外）。
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
        train_end_date  = unique_dates[current_idx - 1]
        test_start_date = unique_dates[current_idx]

        # パージ
        actual_train_end_date = train_end_date - pd.Timedelta(days=purge_days)

        # 訓練/テストデータの分割
        train_mask = dates <= actual_train_end_date
        test_mask  = (dates >= test_start_date) & (dates < test_start_date + pd.Timedelta(days=test_window_days))

        if not test_mask.any():
            break

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        test_dates_fold  = dates[test_mask]
        test_m_idx_fold  = m_idx[test_mask]

        if len(X_train) < 10:
            break

        model = train_model(X_train, y_train, num_boost_round=num_boost_round)

        # 予測
        test_preds  = model.predict(X_test)
        train_preds = model.predict(X_train)

        # Train_IC: モデルが実際に訓練したデータ全体（X_train）での IC
        train_metrics = calculate_metrics(y_train, train_preds)

        fold_res = pd.DataFrame({
            'Fold':          fold,
            'Date':          test_dates_fold,
            'Meeting_Index': test_m_idx_fold,
            'Actual':        y_test,
            'Pred':          test_preds,
            'Train_IC':      train_metrics['IC'],
        })
        results.append(fold_res)

        last_model   = model
        last_X_test  = X_test
        last_y_test  = y_test
        last_X_train = X_train

        fold += 1
        next_date = test_start_date + pd.Timedelta(days=test_window_days)
        # bisect_left で次フォールド開始インデックスを O(log N) で探索
        next_idx = bisect.bisect_left(unique_dates, next_date)
        if next_idx >= len(unique_dates):
            break
        current_idx = next_idx

    return results, last_model, last_X_test, last_y_test, last_X_train


def walk_forward_validation(df, target_col, start_date, test_window_days=90, purge_days=5, num_boost_round=100):
    """
    パージ付きウォークフォワード検証。OOS 予測結果 DataFrame を返す。

    Returns:
        pd.DataFrame: 列 [Fold, Date, Meeting_Index, Actual, Pred, Train_IC]
    """
    results, _, _, _, _ = _walk_forward_core(
        df, target_col, start_date, test_window_days, purge_days, num_boost_round
    )
    if not results:
        return pd.DataFrame()
    return pd.concat(results).reset_index(drop=True)


def walk_forward_with_model(df, target_col, start_date, test_window_days=90, purge_days=5, num_boost_round=100):
    """
    パージ付きウォークフォワード検証。OOS 結果に加えて最終フォールドのモデルとデータを返す。

    特徴量重要度の確認やシグナル生成など、最終フォールドのモデルが必要な場合に使用する。

    Returns:
        results    (pd.DataFrame): OOS 予測結果
        last_model               : 最終フォールドの LightGBM モデル
        last_X_test (pd.DataFrame): 最終フォールドのテスト特徴量
        last_y_test (pd.Series)  : 最終フォールドのテストターゲット
        last_X_train (pd.DataFrame): 最終フォールドの訓練特徴量
    """
    results, last_model, last_X_test, last_y_test, last_X_train = _walk_forward_core(
        df, target_col, start_date, test_window_days, purge_days, num_boost_round
    )
    if not results:
        return pd.DataFrame(), None, None, None, None
    return pd.concat(results).reset_index(drop=True), last_model, last_X_test, last_y_test, last_X_train

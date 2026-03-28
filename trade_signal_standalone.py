#!/usr/bin/env python3
"""
BOJ OIS Trade Signal Generator  [Model B: post-MPM 5-day window NOT excluded]
Usage (from project root): python trade_signal_standalone.py

Standalone version: requires only data/ directory. No src/ imports.
Output: outputs/trade_signal_nofilter_YYYYMMDD.pdf
"""
import os, sys, bisect, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr, linregress

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# =============================================================================
# Inlined: src/processing.py
# =============================================================================

def load_meeting_dates(meeting_csv_path):
    df_meetings = pd.read_csv(meeting_csv_path)
    df_meetings['Date'] = pd.to_datetime(df_meetings['Date'])
    return sorted(df_meetings['Date'].unique())


def load_and_clean_data(excel_path, meeting_csv_path):
    df_raw = pd.read_excel(excel_path)
    df = df_raw.iloc[1:].copy()
    df['日付'] = pd.to_datetime(df['日付'], format='%Y年%m月%d日')
    df = df.sort_values('日付').reset_index(drop=True)

    rename_dict = {
        'JPBOJ1ONI=TRDT (MID_PRICE)': 'M1', 'JPBOJ2ONI=TRDT (MID_PRICE)': 'M2',
        'JPBOJ3ONI=TRDT (MID_PRICE)': 'M3', 'JPBOJ4ONI=TRDT (MID_PRICE)': 'M4',
        'JPBOJ5ONI=TRDT (MID_PRICE)': 'M5', 'JPBOJ6ONI=TRDT (MID_PRICE)': 'M6',
        'JPBOJ7ONI=TRDT (MID_PRICE)': 'M7', 'JPBOJ8ONI=TRDT (MID_PRICE)': 'M8',
        'JPY= (MID_PRICE)': 'USDJPY', 'JGBc1 (TRDPRC_1)': 'JGB_Future',
        '.N225 (TRDPRC_1)': 'Nikkei225', '.DXY (TRDPRC_1)': 'DXY',
        'JP12MONI=TRDT (BID)': 'T12', 'JP18MONI=TRDT (BID)': 'T18',
        'JP24MONI=TRDT (BID)': 'T24',
    }
    df = df.rename(columns=rename_dict)

    cols_to_keep = ['日付', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8',
                    'USDJPY', 'JGB_Future', 'Nikkei225', 'DXY', 'T12', 'T18', 'T24']
    df = df[[c for c in cols_to_keep if c in df.columns]].copy()

    for col in df.columns.drop('日付'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_meetings = pd.read_csv(meeting_csv_path)
    df_meetings['Date'] = pd.to_datetime(df_meetings['Date'])
    df = pd.merge(df, df_meetings[['Date', 'Policy_Rate', 'Event']],
                  left_on='日付', right_on='Date', how='left', suffixes=('', '_mtg'))
    df = df.drop(columns=['Date']).rename(columns={'日付': 'Date'})

    df['Is_Meeting_Day'] = df['Event'].notnull().astype(int)
    df['Actual_Policy_Rate'] = df['Policy_Rate'].ffill().bfill()

    all_rate_cols = [c for c in ['M1','M2','M3','M4','M5','M6','M7','M8','T12','T18','T24']
                     if c in df.columns]
    for col in all_rate_cols:
        df[f'{col}_is_imputed'] = df[col].isnull().astype(int)

    impute_cols = all_rate_cols + [c for c in ['USDJPY','JGB_Future','Nikkei225','DXY']
                                   if c in df.columns]
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)
    df[impute_cols] = imputer.fit_transform(df[impute_cols])

    all_meeting_dates = sorted(df_meetings['Date'].unique())

    def days_to_next_mpm(date):
        idx = bisect.bisect_left(all_meeting_dates, date)
        return (all_meeting_dates[idx] - date).days if idx < len(all_meeting_dates) else np.nan

    df['Days_to_MPM'] = df['Date'].map({d: days_to_next_mpm(d) for d in df['Date'].unique()})
    df['is_post_mpm'] = (
        df['Is_Meeting_Day'].rolling(window=6, min_periods=1).max() == 1
    ).astype(int)

    tenor_cols = [c for c in ['T12','T18','T24'] if c in df.columns]
    ext_cols   = [c for c in ['USDJPY','JGB_Future','Nikkei225','DXY'] if c in df.columns]
    final_cols = (['Date','M1','M2','M3','M4','M5','M6','M7','M8'] + tenor_cols + ext_cols +
                  ['Actual_Policy_Rate','Is_Meeting_Day','is_post_mpm','Days_to_MPM'] +
                  [f'{c}_is_imputed' for c in all_rate_cols])
    return df[final_cols]


# =============================================================================
# Inlined: src/features.py  (frac_diff + generate_features)
# =============================================================================

def frac_diff(series, d, window=50):
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights[::-1])
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window:
        return pd.Series(result, index=series.index)
    windows = sliding_window_view(arr, window_shape=window)
    dots = windows @ weights
    dots[np.isnan(windows).any(axis=1)] = np.nan
    result[window - 1:] = dots
    return pd.Series(result, index=series.index)


def generate_features(df, d=0.4, window=50):
    feat_df = df.copy()
    boj_cols   = [f'M{i}' for i in range(1, 9)]
    tenor_cols = [c for c in ['T12','T18','T24'] if c in feat_df.columns]
    all_rate_cols = boj_cols + tenor_cols
    for col in all_rate_cols:
        feat_df[f'{col}_spread'] = feat_df[col] - feat_df['Actual_Policy_Rate']
    for col in all_rate_cols:
        feat_df[f'{col}_frac_diff'] = frac_diff(feat_df[f'{col}_spread'], d=d, window=window)
    for col in ['USDJPY','JGB_Future','Nikkei225','DXY']:
        if col in feat_df.columns:
            feat_df[f'{col}_frac_diff'] = frac_diff(feat_df[col], d=d, window=window)
    return feat_df


# =============================================================================
# Inlined: src/features_rv.py
# =============================================================================

def generate_rv_features(df, d=0.4, window=50):
    feat_df = df.copy()
    for n in range(1, 9):
        feat_df[f'M{n}_spread'] = feat_df[f'M{n}'] - feat_df['Actual_Policy_Rate']
    for n in range(1, 8):
        feat_df[f'C{n}'] = feat_df[f'M{n}_spread'] - feat_df[f'M{n+1}_spread']
        feat_df[f'C{n}_frac_diff'] = frac_diff(feat_df[f'C{n}'], d=d, window=window)
    for n in range(2, 8):
        feat_df[f'B{n}'] = (2 * feat_df[f'M{n}_spread']
                            - feat_df[f'M{n-1}_spread'] - feat_df[f'M{n+1}_spread'])
        feat_df[f'B{n}_frac_diff'] = frac_diff(feat_df[f'B{n}'], d=d, window=window)
    feat_df['M1_frac_diff']         = frac_diff(feat_df['M1_spread'], d=d, window=window)
    feat_df['Slope_M1M8']           = feat_df['M1_spread'] - feat_df['M8_spread']
    feat_df['Slope_M1M8_frac_diff'] = frac_diff(feat_df['Slope_M1M8'], d=d, window=window)
    for col in ['USDJPY','JGB_Future','Nikkei225','DXY']:
        if col in feat_df.columns:
            feat_df[f'{col}_frac_diff'] = frac_diff(feat_df[col], d=d, window=window)
    return feat_df


# =============================================================================
# Inlined: src/pooling.py
# =============================================================================

def pool_boj_data(df, meeting_dates):
    boj_rate_cols   = [f'M{i}' for i in range(1, 9)]
    tenor_rate_cols = [c for c in ['T12','T18','T24'] if c in df.columns]
    raw_rate_cols   = boj_rate_cols + tenor_rate_cols
    id_cols = [col for col in df.columns if col not in raw_rate_cols]

    pooled_boj = df.melt(id_vars=id_cols, value_vars=boj_rate_cols,
                         var_name='Rate_Label', value_name='Rate_Value')
    pooled_boj['Meeting_Index'] = pooled_boj['Rate_Label'].str.extract(r'(\d+)').astype(int)
    pooled_boj['Is_Tenor_OIS']  = 0

    tenor_index_map = {'T12': 10, 'T18': 11, 'T24': 12}
    if tenor_rate_cols:
        pooled_tenor = df.melt(id_vars=id_cols, value_vars=tenor_rate_cols,
                               var_name='Rate_Label', value_name='Rate_Value')
        pooled_tenor['Meeting_Index'] = pooled_tenor['Rate_Label'].map(tenor_index_map)
        pooled_tenor['Is_Tenor_OIS']  = 1
        pooled = pd.concat([pooled_boj, pooled_tenor], ignore_index=True)
    else:
        pooled = pooled_boj

    pooled = pooled.sort_values(['Rate_Label', 'Date']).reset_index(drop=True)

    all_meeting_dates   = [pd.Timestamp(m) for m in meeting_dates]
    meeting_date_to_rank = {m: i for i, m in enumerate(all_meeting_dates)}

    def _get_next_meeting_rank(date):
        idx = bisect.bisect_left(all_meeting_dates, date)
        return meeting_date_to_rank[all_meeting_dates[idx]] if idx < len(all_meeting_dates) else None

    date_to_next_rank = {d: _get_next_meeting_rank(d) for d in df['Date'].unique()}
    pooled['_next_rank'] = pooled['Date'].map(date_to_next_rank)
    pooled['Absolute_Meeting_ID'] = (pooled['_next_rank'] + pooled['Meeting_Index'] - 1).astype('Int64')
    pooled.loc[pooled['Is_Tenor_OIS'] == 1, 'Absolute_Meeting_ID'] = pd.NA
    pooled = pooled.drop(columns=['_next_rank'])

    data_start_date = df['Date'].min()

    def get_first_seen_date(abs_id):
        if pd.isna(abs_id):
            return pd.NaT
        k = int(abs_id)
        pred_idx = k - 8
        if pred_idx < 0:
            return data_start_date
        elif pred_idx < len(all_meeting_dates):
            return all_meeting_dates[pred_idx]
        return pd.NaT

    abs_id_to_first_seen = {k: get_first_seen_date(k)
                             for k in pooled['Absolute_Meeting_ID'].dropna().unique()}
    pooled['First_Seen_Date'] = pooled['Absolute_Meeting_ID'].map(abs_id_to_first_seen)
    pooled['Days_since_first_seen'] = (pooled['Date'] - pooled['First_Seen_Date']).dt.days
    pooled = pooled.drop(columns=['First_Seen_Date'])

    truncated_mask = (pooled['Is_Tenor_OIS'] == 0) & (pooled['Absolute_Meeting_ID'] < 8)
    pooled = pooled[~truncated_mask].reset_index(drop=True)

    for h in [1, 3, 5]:
        pooled[f'Target_{h}d'] = (
            pooled.groupby('Rate_Label')['Rate_Value'].shift(-h) - pooled['Rate_Value']
        )
    for h in [3, 5]:
        instr_std = pooled.groupby('Rate_Label')[f'Target_{h}d'].transform('std')
        pooled[f'Target_{h}d_std']  = instr_std
        pooled[f'Target_{h}d_norm'] = pooled[f'Target_{h}d'] / instr_std

    boj_spread_cols   = [f'M{i}_spread'     for i in range(1, 9)]
    boj_fd_cols       = [f'M{i}_frac_diff'  for i in range(1, 9)]
    boj_imp_cols      = [f'M{i}_is_imputed' for i in range(1, 9)]
    tenor_spread_cols = [f'{c}_spread'      for c in tenor_rate_cols]
    tenor_fd_cols     = [f'{c}_frac_diff'   for c in tenor_rate_cols]
    tenor_imp_cols    = [f'{c}_is_imputed'  for c in tenor_rate_cols]
    ext_fd_cols       = ['USDJPY_frac_diff','JGB_Future_frac_diff','Nikkei225_frac_diff','DXY_frac_diff']
    basic_cols  = ['Date','Rate_Label','Meeting_Index','Is_Tenor_OIS',
                   'is_post_mpm','Days_to_MPM','Actual_Policy_Rate',
                   'Absolute_Meeting_ID','Days_since_first_seen']
    target_cols = ['Target_1d',
                   'Target_3d','Target_3d_norm','Target_3d_std',
                   'Target_5d','Target_5d_norm','Target_5d_std']
    final_cols  = (basic_cols + boj_spread_cols + boj_fd_cols + boj_imp_cols
                   + tenor_spread_cols + tenor_fd_cols + tenor_imp_cols
                   + ext_fd_cols + target_cols)
    available = [c for c in final_cols if c in pooled.columns]
    return pooled[available]


# =============================================================================
# Inlined: src/pooling_curve.py
# =============================================================================

def pool_curve_data(df):
    curve_raw_cols = [f'C{n}' for n in range(1, 8)]
    id_cols = [c for c in df.columns if c not in curve_raw_cols]
    pooled = df.melt(id_vars=id_cols, value_vars=curve_raw_cols,
                     var_name='Rate_Label', value_name='Rate_Value')
    pooled['Meeting_Index'] = pooled['Rate_Label'].str.extract(r'(\d+)').astype(int)
    pooled = pooled.sort_values(['Rate_Label', 'Date']).reset_index(drop=True)
    for h in [1, 3, 5]:
        pooled[f'Target_{h}d'] = (
            pooled.groupby('Rate_Label')['Rate_Value'].shift(-h) - pooled['Rate_Value']
        )
    for h in [3, 5]:
        instr_std = pooled.groupby('Rate_Label')[f'Target_{h}d'].transform('std')
        pooled[f'Target_{h}d_std']  = instr_std
        pooled[f'Target_{h}d_norm'] = pooled[f'Target_{h}d'] / instr_std
    pooled['Curve_Level'] = pooled['Rate_Value']

    imputed_cols_curve = []
    for n in range(1, 8):
        for col in [f'M{n}_is_imputed', f'M{n+1}_is_imputed']:
            if col in pooled.columns and col not in imputed_cols_curve:
                imputed_cols_curve.append(col)

    basic_cols    = ['Date','Rate_Label','Meeting_Index','is_post_mpm',
                     'Days_to_MPM','Actual_Policy_Rate']
    anchor_cols   = ['M1_spread','M1_frac_diff']
    curve_fd_cols = [f'C{n}_frac_diff' for n in range(1, 8)]
    ext_fd_cols   = ['USDJPY_frac_diff','JGB_Future_frac_diff',
                     'Nikkei225_frac_diff','DXY_frac_diff']
    target_cols   = ['Target_1d',
                     'Target_3d','Target_3d_norm','Target_3d_std',
                     'Target_5d','Target_5d_norm','Target_5d_std']
    final_cols = (basic_cols + ['Curve_Level'] + anchor_cols + curve_fd_cols
                  + imputed_cols_curve + ext_fd_cols + target_cols)
    return pooled[[c for c in final_cols if c in pooled.columns]]


# =============================================================================
# Inlined: src/pooling_butterfly.py
# =============================================================================

def pool_butterfly_data(df):
    fly_raw_cols = [f'B{n}' for n in range(2, 8)]
    id_cols = [c for c in df.columns if c not in fly_raw_cols]
    pooled = df.melt(id_vars=id_cols, value_vars=fly_raw_cols,
                     var_name='Rate_Label', value_name='Rate_Value')
    pooled['Meeting_Index'] = pooled['Rate_Label'].str.extract(r'(\d+)').astype(int)
    pooled = pooled.sort_values(['Rate_Label', 'Date']).reset_index(drop=True)
    for h in [1, 3, 5]:
        pooled[f'Target_{h}d'] = (
            pooled.groupby('Rate_Label')['Rate_Value'].shift(-h) - pooled['Rate_Value']
        )
    for h in [3, 5]:
        instr_std = pooled.groupby('Rate_Label')[f'Target_{h}d'].transform('std')
        pooled[f'Target_{h}d_std']  = instr_std
        pooled[f'Target_{h}d_norm'] = pooled[f'Target_{h}d'] / instr_std
    pooled['Fly_Level'] = pooled['Rate_Value']

    imputed_cols_fly = []
    for n in range(2, 8):
        for col in [f'M{n-1}_is_imputed', f'M{n}_is_imputed', f'M{n+1}_is_imputed']:
            if col in pooled.columns and col not in imputed_cols_fly:
                imputed_cols_fly.append(col)

    basic_cols  = ['Date','Rate_Label','Meeting_Index','is_post_mpm',
                   'Days_to_MPM','Actual_Policy_Rate']
    anchor_cols = ['M1_spread','M1_frac_diff','Slope_M1M8','Slope_M1M8_frac_diff']
    fly_fd_cols = [f'B{n}_frac_diff' for n in range(2, 8)]
    ext_fd_cols = ['USDJPY_frac_diff','JGB_Future_frac_diff',
                   'Nikkei225_frac_diff','DXY_frac_diff']
    target_cols = ['Target_1d',
                   'Target_3d','Target_3d_norm','Target_3d_std',
                   'Target_5d','Target_5d_norm','Target_5d_std']
    final_cols  = (basic_cols + ['Fly_Level'] + anchor_cols + fly_fd_cols
                   + imputed_cols_fly + ext_fd_cols + target_cols)
    return pooled[[c for c in final_cols if c in pooled.columns]]


# =============================================================================
# Inlined: src/modeling.py
# =============================================================================

NON_FEATURE_COLS = {
    'Date', 'Rate_Label', 'Rate_Value', 'is_post_mpm',
    'Target_1d',
    'Target_3d', 'Target_3d_norm', 'Target_3d_std',
    'Target_5d', 'Target_5d_norm', 'Target_5d_std',
}
CATEGORICAL_COLS    = ['Meeting_Index', 'Is_Tenor_OIS', 'Absolute_Meeting_ID']
BOJ_MEETING_INDICES = set(range(1, 9))


def get_features_and_target(df, target_col):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and c != target_col]
    data = df[feature_cols + [target_col, 'Date', 'is_post_mpm']].copy()
    # post-MPM rows are included (Model B, 2026-03-23)
    data = data.dropna(subset=feature_cols + [target_col])
    return data[feature_cols], data[target_col], data['Date'], data['Meeting_Index']


def train_model(X_train, y_train, categorical_feature=None, num_boost_round=100):
    if categorical_feature is None:
        categorical_feature = CATEGORICAL_COLS
    params = {
        'objective': 'regression', 'metric': 'rmse', 'verbosity': -1,
        'boosting_type': 'gbdt', 'random_state': 42, 'learning_rate': 0.05,
        'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 5, 'min_data_in_leaf': 30, 'lambda_l2': 1.0,
    }
    cat_features = [c for c in categorical_feature if c in X_train.columns]
    train_data   = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    return lgb.train(params, train_data, num_boost_round=num_boost_round)


def calculate_metrics(y_true, y_pred):
    if len(y_true) < 2:
        return {'IC': np.nan}
    ic, _ = spearmanr(y_true, y_pred)
    return {'IC': ic}


def summarize_ic(results, recent_n_folds=3, instrument_indices=None):
    if results.empty:
        return {'ic_all': np.nan, 'ic_by_fold': {}, 'ic_recent': np.nan,
                'cs_ic': np.nan, 'ts_ic': np.nan, 'train_ic': np.nan, 'gap': np.nan}

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
    train_ic  = results['Train_IC'].mean()

    _cs_indices = instrument_indices if instrument_indices is not None else BOJ_MEETING_INDICES
    boj_results = results[results['Meeting_Index'].isin(_cs_indices)]
    cs_ics = []
    for _, grp in boj_results.groupby('Date'):
        if len(grp) >= 3:
            ic, _ = spearmanr(grp['Actual'], grp['Pred'])
            if not np.isnan(ic):
                cs_ics.append(ic)
    cs_ic = float(np.mean(cs_ics)) if cs_ics else np.nan

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


def _walk_forward_core(df, target_col, start_date,
                       test_window_days=90, purge_days=5, num_boost_round=100):
    X, y, dates, m_idx = get_features_and_target(df, target_col)
    unique_dates = sorted(dates.unique())

    test_start_idx = 0
    for i, d in enumerate(unique_dates):
        if d >= pd.to_datetime(start_date):
            test_start_idx = i
            break
    if test_start_idx == 0:
        test_start_idx = 1

    results, current_idx, fold = [], test_start_idx, 0
    last_model = last_X_test = last_y_test = last_X_train = None

    while current_idx < len(unique_dates):
        train_end_date       = unique_dates[current_idx - 1]
        test_start_date      = unique_dates[current_idx]
        actual_train_end     = train_end_date - pd.Timedelta(days=purge_days)
        train_mask = dates <= actual_train_end
        test_mask  = ((dates >= test_start_date) &
                      (dates < test_start_date + pd.Timedelta(days=test_window_days)))
        if not test_mask.any():
            break

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        if len(X_train) < 10:
            break

        model       = train_model(X_train, y_train, num_boost_round=num_boost_round)
        test_preds  = model.predict(X_test)
        train_preds = model.predict(X_train)
        train_ic    = calculate_metrics(y_train, train_preds)['IC']

        results.append(pd.DataFrame({
            'Fold':          fold,
            'Date':          dates[test_mask],
            'Meeting_Index': m_idx[test_mask],
            'Actual':        y_test,
            'Pred':          test_preds,
            'Train_IC':      train_ic,
        }))

        last_model = model; last_X_test = X_test
        last_y_test = y_test; last_X_train = X_train

        fold += 1
        next_date = test_start_date + pd.Timedelta(days=test_window_days)
        next_idx  = bisect.bisect_left(unique_dates, next_date)
        if next_idx >= len(unique_dates):
            break
        current_idx = next_idx

    return results, last_model, last_X_test, last_y_test, last_X_train


def walk_forward_with_model(df, target_col, start_date,
                             test_window_days=90, purge_days=5, num_boost_round=100):
    results, last_model, last_X_test, last_y_test, last_X_train = _walk_forward_core(
        df, target_col, start_date, test_window_days, purge_days, num_boost_round)
    if not results:
        return pd.DataFrame(), None, None, None, None
    return (pd.concat(results).reset_index(drop=True),
            last_model, last_X_test, last_y_test, last_X_train)


# =============================================================================
# Config
# =============================================================================
EXCEL_PATH      = os.path.join(SCRIPT_DIR, 'data', 'BOJ_data.xlsx')
MEETING_PATH    = os.path.join(SCRIPT_DIR, 'data', 'BOJ_meeting_history.csv')
OUT_DIR         = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE      = '2024-01-01'
FOLD_DIAG_START = 5
TOP_N_FEATURES  = 15
FLY_SHORT       = {2: 'B2', 3: 'B3', 4: 'B4', 5: 'B5', 6: 'B6', 7: 'B7'}

# =============================================================================
# 1. Data loading
# =============================================================================
print('Loading data...', flush=True)
df_raw        = load_and_clean_data(EXCEL_PATH, MEETING_PATH)
meeting_dates = load_meeting_dates(MEETING_PATH)

df_rv  = generate_rv_features(df_raw)
df_fly = pool_butterfly_data(df_rv)
df_crv = pool_curve_data(df_rv)
df_out = pool_boj_data(generate_features(df_raw), meeting_dates)

print(f'Data through : {df_raw["Date"].max().date()}')
print(f'Fly {df_fly.shape}  Crv {df_crv.shape}  Out {df_out.shape}')


# =============================================================================
# 2. Walk-forward
# =============================================================================
print('Running walk-forward (6 models)...', flush=True)

res_fly_3d, mdl_fly_3d, *_ = walk_forward_with_model(df_fly, 'Target_3d_norm', START_DATE)
res_fly_5d, mdl_fly_5d, *_ = walk_forward_with_model(df_fly, 'Target_5d_norm', START_DATE)
res_crv_3d, mdl_crv_3d, *_ = walk_forward_with_model(df_crv, 'Target_3d_norm', START_DATE)
res_crv_5d, mdl_crv_5d, *_ = walk_forward_with_model(df_crv, 'Target_5d_norm', START_DATE)
res_out_3d, mdl_out_3d, *_ = walk_forward_with_model(df_out, 'Target_3d_norm', START_DATE)
res_out_5d, mdl_out_5d, *_ = walk_forward_with_model(df_out, 'Target_5d_norm', START_DATE)

_std_cols = ['Date', 'Meeting_Index', 'Target_3d_std', 'Target_5d_std']
std_fly = df_fly[_std_cols].drop_duplicates()
std_crv = df_crv[_std_cols].drop_duplicates()
std_out = df_out[_std_cols].drop_duplicates()

res_fly_3d = res_fly_3d.merge(std_fly, on=['Date', 'Meeting_Index'], how='left')
res_fly_5d = res_fly_5d.merge(std_fly, on=['Date', 'Meeting_Index'], how='left')
res_crv_3d = res_crv_3d.merge(std_crv, on=['Date', 'Meeting_Index'], how='left')
res_crv_5d = res_crv_5d.merge(std_crv, on=['Date', 'Meeting_Index'], how='left')
res_out_3d = res_out_3d.merge(std_out, on=['Date', 'Meeting_Index'], how='left')
res_out_5d = res_out_5d.merge(std_out, on=['Date', 'Meeting_Index'], how='left')

print(f'OOS: {res_fly_3d["Date"].min().date()} -> {res_fly_3d["Date"].max().date()}  '
      f'({res_fly_3d["Fold"].max() + 1} folds)')


# =============================================================================
# 3. Signal date & signal functions
# =============================================================================

def _predict_for_signal(df, model, target_col, signal_date):
    """最終フォールドモデルを使って signal_date の特徴量から予測を生成する。

    ターゲット（shift -h）が NaN の最新日でも特徴量が揃っていれば予測可能。
    OOS 結果（res_*）は過去の精度評価用。このシグナル予測は「今日から将来を予測」用。

    注意: Actual = NaN（未来の実績値は存在しない）。
         _m2m5_signal が Actual_Xd_bp を返すが、現在の表示コードはこれを参照しない。
         将来 Actual_Xd_bp を表示に使う場合は NaN 処理が必要。
    """
    rows = df[df['Date'] == signal_date].copy()
    if rows.empty:
        raise ValueError(f'signal_date {signal_date.date()} が df に存在しない')

    # model.feature_name() で訓練時と完全に同じ列セット・列順を保証
    # NON_FEATURE_COLS の変更や df への列追加があっても列不一致が生じない
    feature_cols = model.feature_name()
    X = rows[feature_cols]

    nan_features = X.columns[X.isnull().any()].tolist()
    valid = X.notna().all(axis=1)
    if not valid.any():
        raise ValueError(
            f'signal_date {signal_date.date()} の全行に NaN 特徴量あり: {nan_features}\n'
            f'  対処: BOJ_meeting_history.csv に次回会合日程が登録されているか確認'
        )
    if not valid.all():
        print(f'[WARNING] signal_date {signal_date.date()}: {(~valid).sum()} 行を'
              f' NaN 特徴量で除外 ({nan_features})', flush=True)

    rows = rows[valid].copy()
    X    = X[valid]
    rows['Pred']   = model.predict(X)
    rows['Actual'] = np.nan  # 推論モード: 未来の実績値は存在しない
    return rows.reset_index(drop=True)


# シグナル基準日: 全モデルで特徴量が揃っている最新日
# （OOS 結果の最新日ではなく、実際に予測したい日付）
SIGNAL_DATE = min(df_fly['Date'].max(), df_crv['Date'].max(), df_out['Date'].max())
print(f'Signal date  : {SIGNAL_DATE.date()}', flush=True)


def _fly_signal(r3, r5, date):
    d3 = r3[r3['Date'] == date][['Meeting_Index', 'Pred', 'Target_3d_std']].copy()
    d5 = r5[r5['Date'] == date][['Meeting_Index', 'Pred', 'Target_5d_std']].copy()
    d3['Pred_3d_bp'] = d3['Pred'] * d3['Target_3d_std'] * 100
    d5['Pred_5d_bp'] = d5['Pred'] * d5['Target_5d_std'] * 100
    sig = (d3[['Meeting_Index', 'Pred', 'Pred_3d_bp']]
           .merge(d5[['Meeting_Index', 'Pred_5d_bp']], on='Meeting_Index')
           .rename(columns={'Pred': 'Score_3d'}))
    lvl = (df_fly[df_fly['Date'] == date][['Meeting_Index', 'Fly_Level']]
           .drop_duplicates()
           .assign(Fly_Level_bp=lambda x: x['Fly_Level'] * 100))
    sig = sig.merge(lvl[['Meeting_Index', 'Fly_Level_bp']], on='Meeting_Index', how='left')
    sig['Label'] = sig['Meeting_Index'].map(FLY_SHORT)
    return sig.sort_values('Score_3d', ascending=False).reset_index(drop=True)


def _m2m5_signal(r3, r5, date):
    out = {}
    for h, res in [(3, r3), (5, r5)]:
        sc = f'Target_{h}d_std'
        ps, acs = 0.0, 0.0
        for n in [2, 3, 4]:
            row = res[(res['Date'] == date) & (res['Meeting_Index'] == n)]
            if row.empty:
                return None
            sv   = float(row[sc].values[0])
            ps  += float(row['Pred'].values[0])   * sv * 100
            acs += float(row['Actual'].values[0]) * sv * 100
        out[f'Pred_{h}d_bp']   = ps
        out[f'Actual_{h}d_bp'] = acs
    lvl = 0.0
    for n in [2, 3, 4]:
        row_crv = df_crv[(df_crv['Date'] == date) & (df_crv['Meeting_Index'] == n)]
        if not row_crv.empty:
            lvl += float(row_crv['Curve_Level'].values[0]) * 100
    out['M2M5_level_bp'] = lvl
    return out


def _m4_signal(r3, r5, date):
    d3 = r3[(r3['Date'] == date) & (r3['Meeting_Index'] == 4)]
    d5 = r5[(r5['Date'] == date) & (r5['Meeting_Index'] == 4)]
    if d3.empty or d5.empty:
        return None
    m4_row    = df_out[(df_out['Date'] == date) & (df_out['Meeting_Index'] == 4)]
    spread_bp = float(m4_row['M4_spread'].values[0]) * 100 if not m4_row.empty else np.nan
    policy_bp = float(df_raw[df_raw['Date'] == date]['Actual_Policy_Rate'].values[0]) * 100
    return {
        'M4_abs_bp':    spread_bp + policy_bp,
        'M4_spread_bp': spread_bp,
        'Pred_3d_bp':   float(d3['Pred'].values[0]) * float(d3['Target_3d_std'].values[0]) * 100,
        'Pred_5d_bp':   float(d5['Pred'].values[0]) * float(d5['Target_5d_std'].values[0]) * 100,
    }


# 最終フォールドモデルを SIGNAL_DATE の特徴量に直接適用してシグナルを生成
sig_fly_3d = _predict_for_signal(df_fly, mdl_fly_3d, 'Target_3d_norm', SIGNAL_DATE)
sig_fly_5d = _predict_for_signal(df_fly, mdl_fly_5d, 'Target_5d_norm', SIGNAL_DATE)
sig_crv_3d = _predict_for_signal(df_crv, mdl_crv_3d, 'Target_3d_norm', SIGNAL_DATE)
sig_crv_5d = _predict_for_signal(df_crv, mdl_crv_5d, 'Target_5d_norm', SIGNAL_DATE)
# Outright モデルはテナーOIS行（T12/T18/T24）を除外してから渡す
# テナー行は Absolute_Meeting_ID = NaN（設計上の意図的な欠損）なので WARNING を抑制するため
_df_out_boj = df_out[df_out['Is_Tenor_OIS'] == 0]
sig_out_3d = _predict_for_signal(_df_out_boj, mdl_out_3d, 'Target_3d_norm', SIGNAL_DATE)
sig_out_5d = _predict_for_signal(_df_out_boj, mdl_out_5d, 'Target_5d_norm', SIGNAL_DATE)

fly_sig  = _fly_signal(sig_fly_3d, sig_fly_5d, SIGNAL_DATE)
m2m5_sig = _m2m5_signal(sig_crv_3d, sig_crv_5d, SIGNAL_DATE)
m4_sig   = _m4_signal(sig_out_3d, sig_out_5d, SIGNAL_DATE)


# ─────────────────────────────────────────────────────────────────────────────
# 3b. M1-M8 vs T12 (1y): correlation & slope over last 50 days
# ─────────────────────────────────────────────────────────────────────────────
_CORR_WINDOW = 50
_raw_w   = df_raw.sort_values('Date').tail(_CORR_WINDOW + 1)
_t12_chg = _raw_w['T12'].diff().dropna() if 'T12' in _raw_w.columns else None

_corr_t12  = {}
_slope_t12 = {}
for n in range(1, 9):
    col = f'M{n}'
    if _t12_chg is None or col not in _raw_w.columns:
        _corr_t12[col] = np.nan; _slope_t12[col] = np.nan
        continue
    _mn_chg = _raw_w[col].diff().dropna()
    _idx    = _t12_chg.index.intersection(_mn_chg.index)
    t12v    = _t12_chg.loc[_idx].values
    mnv     = _mn_chg.loc[_idx].values
    mask    = ~(np.isnan(t12v) | np.isnan(mnv))
    if mask.sum() >= 5:
        slope, _, r_val, _, _ = linregress(t12v[mask], mnv[mask])
        _corr_t12[col]  = r_val
        _slope_t12[col] = slope
    else:
        _corr_t12[col] = np.nan; _slope_t12[col] = np.nan


# =============================================================================
# 4. M1-M8 reconstruction
# =============================================================================
def _reconstruct_delta(fly_sig, m2m5_sig, m4_sig, h):
    A = np.zeros((8, 8))
    b = np.zeros(8)
    pc = f'Pred_{h}_bp'
    for i, n in enumerate(range(2, 8)):
        A[i, n - 2] = -1; A[i, n - 1] = 2; A[i, n] = -1
        row = fly_sig[fly_sig['Meeting_Index'] == n]
        b[i] = float(row[pc].values[0]) if not row.empty else 0.0
    A[6, 3] = 1
    b[6] = m4_sig[pc] if m4_sig else 0.0
    A[7, 1] = 1; A[7, 4] = -1
    b[7] = m2m5_sig[pc] if m2m5_sig else 0.0
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.full(8, np.nan)


raw_row   = df_raw[df_raw['Date'] == SIGNAL_DATE].iloc[0]
curr_bp   = np.array([raw_row[f'M{n}'] * 100 for n in range(1, 9)])
policy_bp = raw_row['Actual_Policy_Rate'] * 100
delta_3d  = _reconstruct_delta(fly_sig, m2m5_sig, m4_sig, '3d')
delta_5d  = _reconstruct_delta(fly_sig, m2m5_sig, m4_sig, '5d')

recon_df = pd.DataFrame({
    'Series':   [f'M{n}' for n in range(1, 9)],
    'Current':  curr_bp,
    'Delta_3d': delta_3d,
    'Pred_3d':  curr_bp + delta_3d,
    'Delta_5d': delta_5d,
    'Pred_5d':  curr_bp + delta_5d,
})


# =============================================================================
# 5. Console output
# =============================================================================
SEP = '=' * 68
print(f'\n{SEP}')
print(f'  BOJ OIS Trade Signal [Model B: post-MPM included]   {SIGNAL_DATE.date()}')
print(SEP)

print('\n[ Butterfly B2-B7  (Score_3d desc) ]')
print(f"  {'Rk':>2}  {'Sym':<4}  {'Level(bp)':>9}  {'Pred_3d(bp)':>11}  {'Pred_5d(bp)':>11}  Dir")
print('  ' + '-' * 55)
for rk, row in enumerate(fly_sig.itertuples(), 1):
    lvl = f'{row.Fly_Level_bp:+.3f}' if not pd.isna(row.Fly_Level_bp) else '  N/A '
    d   = 'UP  ' if row.Pred_3d_bp > 0 else 'DOWN'
    print(f'  {rk:>2}  {row.Label:<4}  {lvl:>9}  {row.Pred_3d_bp:>+10.3f}  '
          f'{row.Pred_5d_bp:>+10.3f}  {d}')

print('\n[ M2-M5 Curve Signal ]')
if m2m5_sig:
    d3 = 'Steeper' if m2m5_sig['Pred_3d_bp'] > 0 else 'Flatter'
    d5 = 'Steeper' if m2m5_sig['Pred_5d_bp'] > 0 else 'Flatter'
    print(f"  Level: {m2m5_sig['M2M5_level_bp']:+.2f} bp  |  "
          f"Pred 3d: {m2m5_sig['Pred_3d_bp']:+.3f} bp  {d3}  |  "
          f"Pred 5d: {m2m5_sig['Pred_5d_bp']:+.3f} bp  {d5}")

print('\n[ M4 Outright Signal ]')
if m4_sig:
    d3 = 'UP  ' if m4_sig['Pred_3d_bp'] > 0 else 'DOWN'
    d5 = 'UP  ' if m4_sig['Pred_5d_bp'] > 0 else 'DOWN'
    print(f"  M4={m4_sig['M4_abs_bp']:.1f}bp  spread={m4_sig['M4_spread_bp']:+.1f}bp  |  "
          f"Pred 3d: {m4_sig['Pred_3d_bp']:+.3f}bp {d3}  |  "
          f"Pred 5d: {m4_sig['Pred_5d_bp']:+.3f}bp {d5}")
print(f'\n{SEP}')


# =============================================================================
# 6. IC summaries & feature importance
# =============================================================================
ic_fly_3d = summarize_ic(res_fly_3d, instrument_indices=set(range(2, 8)))
ic_fly_5d = summarize_ic(res_fly_5d, instrument_indices=set(range(2, 8)))
ic_crv_3d = summarize_ic(res_crv_3d)
ic_crv_5d = summarize_ic(res_crv_5d)
ic_out_3d = summarize_ic(res_out_3d)
ic_out_5d = summarize_ic(res_out_5d)


def _feat_imp(model):
    df_i = pd.DataFrame({'feature': model.feature_name(),
                         'gain':    model.feature_importance(importance_type='gain')})
    return df_i.sort_values('gain', ascending=False).head(TOP_N_FEATURES).reset_index(drop=True)


imp_fly_3d = _feat_imp(mdl_fly_3d)
imp_fly_5d = _feat_imp(mdl_fly_5d)
imp_crv_3d = _feat_imp(mdl_crv_3d)
imp_crv_5d = _feat_imp(mdl_crv_5d)
imp_out_3d = _feat_imp(mdl_out_3d)
imp_out_5d = _feat_imp(mdl_out_5d)


# =============================================================================
# PDF output
# =============================================================================
PDF_PATH = os.path.join(OUT_DIR, f'trade_signal_nofilter_{SIGNAL_DATE.strftime("%Y%m%d")}.pdf')
_pdf = PdfPages(PDF_PATH)
print(f'\nGenerating PDF: {os.path.basename(PDF_PATH)}')

# ─────────────────────────────────────────────────────────────────────────────
# Page 1: Signal summary + M1-M8 reconstruction
# ─────────────────────────────────────────────────────────────────────────────
fly_plot  = fly_sig.sort_values('Meeting_Index').reset_index(drop=True)
fly_lbls  = fly_plot['Label'].tolist()
fly_lvls  = fly_plot['Fly_Level_bp'].tolist()
fly_p3    = fly_plot['Pred_3d_bp'].tolist()
fly_p5    = fly_plot['Pred_5d_bp'].tolist()
rate_lbls = [f'M{n}' for n in range(1, 9)]
rate_vals = [raw_row[c] * 100 for c in rate_lbls]

fig = plt.figure(figsize=(18, 24))
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)
fig.suptitle(
    f'BOJ OIS Trade Signal  [Model B: post-MPM included]   |   {SIGNAL_DATE.date()}',
    fontsize=14, fontweight='bold')

# (0,0) Butterfly levels
ax = fig.add_subplot(gs[0, 0])
ax.bar(fly_lbls, fly_lvls,
       color=['#d62728' if v < 0 else '#ff7f0e' for v in fly_lvls], edgecolor='white')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('bp'); ax.set_title('Butterfly Levels (bp)'); ax.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(fly_lvls):
    if not np.isnan(v):
        ax.text(i, v + (0.02 if v >= 0 else -0.04), f'{v:+.2f}',
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=8.5)

# (0,1) Butterfly predicted changes
ax = fig.add_subplot(gs[0, 1])
x, w = np.arange(len(fly_lbls)), 0.35
ax.bar(x - w/2, fly_p3, w, label='3d',
       color=['#2ca02c' if v >= 0 else '#d62728' for v in fly_p3], alpha=0.85, edgecolor='white')
ax.bar(x + w/2, fly_p5, w, label='5d',
       color=['#1f77b4' if v >= 0 else '#e377c2' for v in fly_p5], alpha=0.85, edgecolor='white')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(fly_lbls)
ax.set_ylabel('Delta (bp)'); ax.set_title('Butterfly Predicted Changes (bp)')
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(fly_p3):
    ax.text(i - w/2, v + (0.01 if v >= 0 else -0.02), f'{v:+.2f}',
            ha='center', va='bottom' if v >= 0 else 'top', fontsize=7.5)

# (1,0) OIS curve
ax = fig.add_subplot(gs[1, 0])
ax.plot(rate_lbls, rate_vals, marker='o', color='#1f77b4', lw=2)
ax.axhline(policy_bp, color='gray', ls='--', alpha=0.6, label=f'Policy {policy_bp:.0f} bp')
ax.scatter(['M4'], [rate_vals[3]], color='red', s=80, zorder=5, label='M4 (Outright)')
ax.scatter(['M2', 'M5'], [rate_vals[1], rate_vals[4]], color='#2ca02c',
           s=60, zorder=5, label='M2, M5 (Curve)')
ax.set_ylabel('bp'); ax.set_title('OIS Curve (bp)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
for i, v in enumerate(rate_vals):
    ax.text(i, v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=7.5)

# (1,1) Curve & Outright signal text
ax = fig.add_subplot(gs[1, 1]); ax.axis('off')
lines = []
if m2m5_sig:
    d3 = 'Steeper ^' if m2m5_sig['Pred_3d_bp'] > 0 else 'Flatter v'
    d5 = 'Steeper ^' if m2m5_sig['Pred_5d_bp'] > 0 else 'Flatter v'
    lines += ['M2-M5 Curve Signal',
              f"  Level : {m2m5_sig['M2M5_level_bp']:+.2f} bp",
              f"  Pred 3d: {m2m5_sig['Pred_3d_bp']:+.3f} bp  [{d3}]",
              f"  Pred 5d: {m2m5_sig['Pred_5d_bp']:+.3f} bp  [{d5}]", '']
if m4_sig:
    d3 = 'UP ^' if m4_sig['Pred_3d_bp'] > 0 else 'DOWN v'
    d5 = 'UP ^' if m4_sig['Pred_5d_bp'] > 0 else 'DOWN v'
    lines += ['M4 Outright Signal',
              f"  Level  : {m4_sig['M4_abs_bp']:.1f} bp",
              f"  Spread : {m4_sig['M4_spread_bp']:+.1f} bp (vs policy)",
              f"  Pred 3d: {m4_sig['Pred_3d_bp']:+.3f} bp  [{d3}]",
              f"  Pred 5d: {m4_sig['Pred_5d_bp']:+.3f} bp  [{d5}]"]
ax.text(0.05, 0.95, '\n'.join(lines), ha='left', va='top', fontsize=11,
        transform=ax.transAxes, family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f4f8', alpha=0.9))

# (2,:) M1-M8 reconstruction table
ax = fig.add_subplot(gs[2, :])
ax.axis('off')
ax.set_title('M1-M8 Reconstruction  --  '
             'B2-B7 x6 + M4(Outright) + M2-M5(Curve) = 8x8 Linear System',
             fontsize=10, fontweight='bold', pad=8)
col_labels = ['Series', 'Current (bp)', 'Pred 3d (bp)', 'Delta 3d (bp)',
              'Pred 5d (bp)', 'Delta 5d (bp)']
cell_data  = []
for _, r in recon_df.iterrows():
    cell_data.append([r['Series'], f"{r['Current']:.2f}", f"{r['Pred_3d']:.2f}",
                      f"{r['Delta_3d']:+.3f}", f"{r['Pred_5d']:.2f}", f"{r['Delta_5d']:+.3f}"])
tbl = ax.table(cellText=cell_data, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 1.8)
for col_i, h_col in [(3, 'Delta_3d'), (5, 'Delta_5d')]:
    for row_i, (_, r) in enumerate(recon_df.iterrows()):
        cell = tbl[row_i + 1, col_i]; val = r[h_col]
        cell.set_facecolor('#c8e6c9' if val > 0.005 else '#ffcdd2' if val < -0.005 else '#fffde7')

# (3,:) Delta line chart
ax = fig.add_subplot(gs[3, :])
x_pos  = np.arange(8)
x_lbls = [f'M{n}' for n in range(1, 9)]
ax.plot(x_pos, delta_3d, marker='o', color='#2ca02c', lw=2, label='Delta 3d (bp)')
ax.plot(x_pos, delta_5d, marker='s', color='#1f77b4', lw=2, label='Delta 5d (bp)')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(x_lbls)
ax.set_ylabel('Delta (bp)'); ax.set_title('Predicted Delta M1-M8 (bp)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
for i, (v3, v5) in enumerate(zip(delta_3d, delta_5d)):
    if not np.isnan(v3):
        ax.text(i - 0.12, v3 + (0.05 if v3 >= 0 else -0.08), f'{v3:+.2f}',
                ha='center', va='bottom' if v3 >= 0 else 'top', fontsize=8, color='#2ca02c')
    if not np.isnan(v5):
        ax.text(i + 0.12, v5 + (0.05 if v5 >= 0 else -0.08), f'{v5:+.2f}',
                ha='center', va='bottom' if v5 >= 0 else 'top', fontsize=8, color='#1f77b4')

# (4,0) M1-M8 vs T12 Correlation
ax = fig.add_subplot(gs[4, 0])
_m_lbls   = [f'M{n}' for n in range(1, 9)]
corr_vals = [_corr_t12.get(l, np.nan) for l in _m_lbls]
ax.bar(_m_lbls, corr_vals,
       color=['#2ca02c' if (not np.isnan(v) and v >= 0) else '#d62728' for v in corr_vals],
       edgecolor='white')
ax.axhline(0, color='black', lw=0.8); ax.set_ylim(-1.1, 1.1)
ax.set_ylabel('Pearson r')
ax.set_title(f'M1-M8 vs T12 (1y) Correlation  (last {_CORR_WINDOW}d)')
ax.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(corr_vals):
    if not np.isnan(v):
        ax.text(i, v + (0.03 if v >= 0 else -0.05), f'{v:.2f}',
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)

# (4,1) M1-M8 vs T12 Regression Slope
ax = fig.add_subplot(gs[4, 1])
slope_vals = [_slope_t12.get(l, np.nan) for l in _m_lbls]
ax.bar(_m_lbls, slope_vals,
       color=['#1f77b4' if (not np.isnan(v) and v >= 0) else '#e377c2' for v in slope_vals],
       edgecolor='white')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('Slope  dM / dT12')
ax.set_title(f'M1-M8 vs T12 (1y) Regression Slope  (last {_CORR_WINDOW}d)')
ax.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(slope_vals):
    if not np.isnan(v):
        ax.text(i, v + (0.01 if v >= 0 else -0.02), f'{v:.2f}',
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)

_pdf.savefig(fig, bbox_inches='tight')
plt.close()
print('  Page 1: Signal summary')


# ─────────────────────────────────────────────────────────────────────────────
# Page 2: IC diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def _scatter_per_inst(ax, results, mi, folds_from, title, color='steelblue'):
    sub = results[(results['Fold'] >= folds_from) & (results['Meeting_Index'] == mi)]
    if sub.empty:
        ax.set_title(title, fontsize=8.5)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    ic, _ = spearmanr(sub['Actual'], sub['Pred'])
    ax.scatter(sub['Actual'], sub['Pred'], s=8, alpha=0.45, color=color)
    lo = min(sub['Actual'].min(), sub['Pred'].min())
    hi = max(sub['Actual'].max(), sub['Pred'].max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=0.8, alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.35)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.35)
    ax.set_title(f'{title}\nIC={ic:.3f}', fontsize=8.5, fontweight='bold')
    ax.set_xlabel('Actual (norm)', fontsize=7); ax.set_ylabel('Pred (norm)', fontsize=7)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.2)


def _ic_fold_bar(ax, ic_dict, label, color_hi, folds_from):
    folds  = sorted(ic_dict['ic_by_fold'].keys())
    vals   = [ic_dict['ic_by_fold'][f] for f in folds]
    colors = [color_hi if f >= folds_from else '#cccccc' for f in folds]
    ax.bar([str(f) for f in folds], vals, color=colors, edgecolor='white')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('Fold', fontsize=8); ax.set_ylabel('Global IC', fontsize=8)
    title_lines = (f'{label}  Global IC by Fold\n'
                   f'All={ic_dict["ic_all"]:.3f}  Train={ic_dict["train_ic"]:.3f}  '
                   f'Gap={ic_dict["gap"]:.3f}')
    if 'cs_ic' in ic_dict and not np.isnan(ic_dict['cs_ic']):
        title_lines += f'  CS={ic_dict["cs_ic"]:.3f}'
    ax.set_title(title_lines, fontsize=8.5, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.005 if v >= 0 else -0.018), f'{v:.3f}', ha='center', fontsize=7)


fig = plt.figure(figsize=(20, 24))
gs  = gridspec.GridSpec(5, 6, figure=fig, hspace=0.60, wspace=0.42)
fig.suptitle(
    f'IC Diagnostics (Fold >= {FOLD_DIAG_START} highlighted)  '
    f'[Model B: post-MPM included]   |   {SIGNAL_DATE.date()}',
    fontsize=13, fontweight='bold')

for i, n in enumerate(range(2, 8)):
    _scatter_per_inst(fig.add_subplot(gs[0, i]), res_fly_3d, n, FOLD_DIAG_START,
                      f'Fly {FLY_SHORT[n]} 3d', color='#ff7f0e')
for i, n in enumerate(range(2, 8)):
    _scatter_per_inst(fig.add_subplot(gs[1, i]), res_fly_5d, n, FOLD_DIAG_START,
                      f'Fly {FLY_SHORT[n]} 5d', color='#d62728')

_ic_fold_bar(fig.add_subplot(gs[2, :3]), ic_fly_3d, 'Butterfly 3d', '#ff7f0e', FOLD_DIAG_START)
_ic_fold_bar(fig.add_subplot(gs[2, 3:]), ic_fly_5d, 'Butterfly 5d', '#d62728', FOLD_DIAG_START)
_ic_fold_bar(fig.add_subplot(gs[3, :3]), ic_crv_3d, 'Curve 3d',    '#2ca02c', FOLD_DIAG_START)
_ic_fold_bar(fig.add_subplot(gs[3, 3:]), ic_crv_5d, 'Curve 5d',    '#98df8a', FOLD_DIAG_START)
_ic_fold_bar(fig.add_subplot(gs[4, :3]), ic_out_3d, 'Outright 3d', '#1f77b4', FOLD_DIAG_START)
_ic_fold_bar(fig.add_subplot(gs[4, 3:]), ic_out_5d, 'Outright 5d', '#aec7e8', FOLD_DIAG_START)

_pdf.savefig(fig, bbox_inches='tight')
plt.close()
print('  Page 2: IC diagnostics')


# ─────────────────────────────────────────────────────────────────────────────
# Page 3: Feature importance
# ─────────────────────────────────────────────────────────────────────────────
imp_pairs = [
    (imp_fly_3d, 'Butterfly 3d', '#ff7f0e'), (imp_fly_5d, 'Butterfly 5d', '#d62728'),
    (imp_crv_3d, 'Curve 3d',     '#2ca02c'), (imp_crv_5d, 'Curve 5d',     '#98df8a'),
    (imp_out_3d, 'Outright 3d',  '#1f77b4'), (imp_out_5d, 'Outright 5d',  '#aec7e8'),
]

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle(
    f'Feature Importance (gain, last fold, top {TOP_N_FEATURES})  '
    f'[Model B: post-MPM included]   |   {SIGNAL_DATE.date()}',
    fontsize=13, fontweight='bold')
for ax, (df_imp, title, color) in zip(axes.flat, imp_pairs):
    ax.barh(df_imp['feature'][::-1], df_imp['gain'][::-1], color=color, edgecolor='white')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Gain', fontsize=9); ax.tick_params(labelsize=8)
    ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
_pdf.savefig(fig, bbox_inches='tight')
plt.close()
print('  Page 3: Feature importance')

_pdf.close()
print(f'\nSaved: {PDF_PATH}')
print('All done.')

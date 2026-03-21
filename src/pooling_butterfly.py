import pandas as pd
import numpy as np


def pool_butterfly_data(df):
    """
    RV Butterfly モデル用プーリング。

    入力: generate_rv_features() の出力（B2-B7, B{n}_frac_diff, M1アンカー, スロープ 等）
    出力: 6系列（B2=2M2-M1-M3, ..., B7=2M7-M6-M8）× 日付のロング形式 DataFrame

    Meeting_Index の意味:
        Butterfly_Index 2〜7。B{n} = 2*M{n}_spread - M{n-1}_spread - M{n+1}_spread の n に対応。

    Days_to_MPM の意味:
        入力 DataFrame の Days_to_MPM をそのまま使用（次回MPMまでの日数、日付レベルで全銘柄共通）。
    """
    # -------------------------------------------------------------------
    # Melt: B2〜B7 を縦持ちに変換
    # -------------------------------------------------------------------
    fly_raw_cols = [f'B{n}' for n in range(2, 8)]
    id_cols = [c for c in df.columns if c not in fly_raw_cols]

    pooled = df.melt(
        id_vars=id_cols,
        value_vars=fly_raw_cols,
        var_name='Rate_Label',   # 'B2', 'B3', ..., 'B7'
        value_name='Rate_Value',
    )
    pooled['Meeting_Index'] = pooled['Rate_Label'].str.extract(r'(\d+)').astype(int)

    # Days_to_MPM は入力 DataFrame の値をそのまま使う（次回MPMまでの日数、全銘柄共通）
    # melt後も id_cols 経由で既に列に含まれている

    # -------------------------------------------------------------------
    # 目的変数の生成（バタフライ系列ごとに h 日後の変化）
    # -------------------------------------------------------------------
    pooled = pooled.sort_values(['Rate_Label', 'Date']).reset_index(drop=True)
    for h in [1, 3, 5]:
        pooled[f'Target_{h}d'] = (
            pooled.groupby('Rate_Label')['Rate_Value'].shift(-h) - pooled['Rate_Value']
        )

    # per-butterfly std 正規化
    for h in [3, 5]:
        instr_std = pooled.groupby('Rate_Label')[f'Target_{h}d'].transform('std')
        pooled[f'Target_{h}d_std']  = instr_std
        pooled[f'Target_{h}d_norm'] = pooled[f'Target_{h}d'] / instr_std

    # -------------------------------------------------------------------
    # is_post_mpm フラグ
    # -------------------------------------------------------------------
    df_date_meeting = df[['Date', 'Is_Meeting_Day']].copy()
    df_date_meeting['is_post_mpm'] = (
        df_date_meeting['Is_Meeting_Day'].rolling(window=6, min_periods=1).max() == 1
    ).astype(int)
    pooled = pd.merge(
        pooled,
        df_date_meeting[['Date', 'is_post_mpm']],
        on='Date',
        how='left',
    )

    # -------------------------------------------------------------------
    # Fly_Level: B{n} の現在水準（mean reversion の起点）
    # Rate_Value は NON_FEATURE_COLS で除外されるため、明示的な列名で保持する。
    # -------------------------------------------------------------------
    pooled['Fly_Level'] = pooled['Rate_Value']

    # -------------------------------------------------------------------
    # 列の整理
    # -------------------------------------------------------------------
    # バタフライ B{n} の構成銘柄: M{n-1}, M{n}, M{n+1} → 3本の補完フラグ
    imputed_cols_fly = []
    for n in range(2, 8):
        for col in [f'M{n-1}_is_imputed', f'M{n}_is_imputed', f'M{n+1}_is_imputed']:
            if col in pooled.columns and col not in imputed_cols_fly:
                imputed_cols_fly.append(col)

    basic_cols   = ['Date', 'Rate_Label', 'Meeting_Index', 'is_post_mpm',
                    'Days_to_MPM', 'Actual_Policy_Rate']
    anchor_cols  = ['M1_spread', 'M1_frac_diff',
                    'Slope_M1M8', 'Slope_M1M8_frac_diff']
    fly_fd_cols  = [f'B{n}_frac_diff' for n in range(2, 8)]
    ext_fd_cols  = ['USDJPY_frac_diff', 'JGB_Future_frac_diff',
                    'Nikkei225_frac_diff', 'DXY_frac_diff']
    target_cols  = ['Target_1d',
                    'Target_3d', 'Target_3d_norm', 'Target_3d_std',
                    'Target_5d', 'Target_5d_norm', 'Target_5d_std']

    final_cols = (
        basic_cols
        + ['Fly_Level']      # B{n} 現在水準（mean reversion 起点）
        + anchor_cols
        + fly_fd_cols
        + imputed_cols_fly
        + ext_fd_cols
        + target_cols
    )
    available = [c for c in final_cols if c in pooled.columns]
    return pooled[available]

import pandas as pd
import numpy as np

def pool_boj_data(df):
    """
    設計仕様に基づきデータをプーリング（縦持ち変換）し、ターゲットを生成する。

    レート種別:
      - M1〜M8: 会合先BOJスワップ → Meeting_Index=1〜8, Tenor_Months=0, Is_Tenor_OIS=0
      - T12/T18/T24: テナーOIS（スポット） → Meeting_Index=0, Tenor_Months=12/18/24, Is_Tenor_OIS=1
    """
    # 生レート列（meltの対象、id_varsには含めない）
    boj_rate_cols  = [f'M{i}' for i in range(1, 9)]
    tenor_rate_cols = [c for c in ['T12', 'T18', 'T24'] if c in df.columns]

    # id_cols: 生レート列以外の全列（spread/frac_diff/is_imputed 等の派生列を含む）
    raw_rate_cols = boj_rate_cols + tenor_rate_cols
    id_cols = [col for col in df.columns if col not in raw_rate_cols]

    # --- M1〜M8 のmelt ---
    pooled_boj = df.melt(
        id_vars=id_cols,
        value_vars=boj_rate_cols,
        var_name='Rate_Label',
        value_name='Rate_Value',
    )
    pooled_boj['Meeting_Index'] = pooled_boj['Rate_Label'].str.extract(r'(\d+)').astype(int)
    pooled_boj['Is_Tenor_OIS']  = 0

    # --- T12/T18/T24 のmelt ---
    # Meeting_Index: T12=10, T18=11, T24=12（M1-M8の連番を引き継ぐ）
    tenor_index_map = {'T12': 10, 'T18': 11, 'T24': 12}
    if tenor_rate_cols:
        pooled_tenor = df.melt(
            id_vars=id_cols,
            value_vars=tenor_rate_cols,
            var_name='Rate_Label',
            value_name='Rate_Value',
        )
        pooled_tenor['Meeting_Index'] = pooled_tenor['Rate_Label'].map(tenor_index_map)
        pooled_tenor['Is_Tenor_OIS']  = 1
        pooled = pd.concat([pooled_boj, pooled_tenor], ignore_index=True)
    else:
        pooled = pooled_boj

    # 2. 目的変数の生成（Rate_Labelグループ内でshift）
    pooled = pooled.sort_values(['Rate_Label', 'Date']).reset_index(drop=True)

    for h in [1, 3, 5]:
        pooled[f'Target_{h}d'] = (
            pooled.groupby('Rate_Label')['Rate_Value'].shift(-h) - pooled['Rate_Value']
        )

    # 3. 目的変数の正規化（銘柄ごとのstdで割る）
    # 異なるスケールの銘柄をプーリングする際の異分散補正（WLS相当）。
    # stdは時系列的に安定しているため全期間stdを使用（lookaheadは軽微）。
    # Target_{h}d_norm: 訓練用（等スケール）、Target_{h}d_std: 逆変換・P&L計算用
    for h in [3, 5]:
        instr_std = pooled.groupby('Rate_Label')[f'Target_{h}d'].transform('std')
        pooled[f'Target_{h}d_std']  = instr_std
        pooled[f'Target_{h}d_norm'] = pooled[f'Target_{h}d'] / instr_std

    # 4. MPM当日・直後5日間の除外フラグ
    df_date_meeting = df[['Date', 'Is_Meeting_Day']].copy()
    df_date_meeting['is_post_mpm'] = (
        df_date_meeting['Is_Meeting_Day'].rolling(window=6, min_periods=1).max() == 1
    ).astype(int)
    pooled = pd.merge(pooled, df_date_meeting[['Date', 'is_post_mpm']], on='Date', how='left')

    # 5. カラムの整理
    boj_spread_cols  = [f'M{i}_spread'     for i in range(1, 9)]
    boj_fd_cols      = [f'M{i}_frac_diff'  for i in range(1, 9)]
    boj_imp_cols     = [f'M{i}_is_imputed' for i in range(1, 9)]
    tenor_spread_cols = [f'{c}_spread'     for c in tenor_rate_cols]
    tenor_fd_cols     = [f'{c}_frac_diff'  for c in tenor_rate_cols]
    tenor_imp_cols    = [f'{c}_is_imputed' for c in tenor_rate_cols]
    ext_fd_cols      = ['USDJPY_frac_diff', 'JGB_Future_frac_diff', 'Nikkei225_frac_diff']

    basic_cols  = ['Date', 'Rate_Label', 'Meeting_Index', 'Is_Tenor_OIS',
                   'is_post_mpm', 'Days_to_MPM', 'Actual_Policy_Rate']
    target_cols = ['Target_1d',
                   'Target_3d', 'Target_3d_norm', 'Target_3d_std',
                   'Target_5d', 'Target_5d_norm', 'Target_5d_std']

    # 追加特徴量（設計書 04: Exp-C）
    curve_cols    = ['Curve_Slope'] + [f'Butterfly_M{n}' for n in range(2, 8)]
    curve_fd_cols = [f'{c}_frac_diff' for c in curve_cols]
    cyclic_cols   = ['DayOfWeek_sin', 'DayOfWeek_cos', 'Days_to_MPM_sin', 'Days_to_MPM_cos']

    final_cols = (
        basic_cols
        + boj_spread_cols + boj_fd_cols + boj_imp_cols
        + tenor_spread_cols + tenor_fd_cols + tenor_imp_cols
        + ext_fd_cols
        + curve_cols + curve_fd_cols
        + cyclic_cols
        + target_cols
    )

    # 存在する列のみを選択
    available_cols = [c for c in final_cols if c in pooled.columns]
    return pooled[available_cols]

import bisect
import pandas as pd
import numpy as np

def pool_boj_data(df, meeting_dates):
    """
    設計仕様に基づきデータをプーリング（縦持ち変換）し、ターゲットを生成する。

    Args:
        df:             load_and_clean_data + generate_features の出力 DataFrame
        meeting_dates:  load_meeting_dates() が返す会合日リスト（pd.Timestamp のリスト）
                        Absolute_Meeting_ID の計算に使用する。

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

    # --- Exp-D: Absolute_Meeting_ID の計算 ---
    # meeting_dates は load_meeting_dates() から直接受け取る（Days_to_MPM 逆算不要）
    all_meeting_dates = [pd.Timestamp(m) for m in meeting_dates]
    meeting_date_to_rank = {m: i for i, m in enumerate(all_meeting_dates)}

    # 各日付に対して「次の会合日」を二分探索で求め、そのランクを取得する
    def _get_next_meeting_rank(date):
        idx = bisect.bisect_left(all_meeting_dates, date)
        if idx < len(all_meeting_dates):
            return meeting_date_to_rank[all_meeting_dates[idx]]
        return None

    date_to_next_rank = {d: _get_next_meeting_rank(d) for d in df['Date'].unique()}

    # 4. pooled DataFrame に適用（melt後なので Meeting_Index 列が使える）
    #    Absolute_Meeting_ID = rank(次の会合) + Meeting_Index - 1
    #    注：Is_Tenor_OIS=1 の行は Meeting_Index が 10 以上なので、Absolute_Meeting_ID は意味を持たない（が計算はされる）
    pooled['_next_rank'] = pooled['Date'].map(date_to_next_rank)
    pooled['Absolute_Meeting_ID'] = (pooled['_next_rank'] + pooled['Meeting_Index'] - 1).astype('Int64')
    
    # Is_Tenor_OIS=1 の行は NaN にする
    pooled.loc[pooled['Is_Tenor_OIS'] == 1, 'Absolute_Meeting_ID'] = pd.NA
    pooled = pooled.drop(columns=['_next_rank'])

    # --- Exp-E: Days_since_first_seen の計算 ---
    data_start_date = df['Date'].min()

    def get_first_seen_date(abs_id):
        if pd.isna(abs_id):
            return pd.NaT
        k = int(abs_id)
        predecessor_idx = k - 8
        if predecessor_idx < 0:
            return data_start_date
        elif predecessor_idx < len(all_meeting_dates):
            return all_meeting_dates[predecessor_idx]
        else:
            return pd.NaT

    abs_id_to_first_seen = {
        k: get_first_seen_date(k) for k in pooled['Absolute_Meeting_ID'].dropna().unique()
    }

    pooled['First_Seen_Date'] = pooled['Absolute_Meeting_ID'].map(abs_id_to_first_seen)
    pooled['Days_since_first_seen'] = (pooled['Date'] - pooled['First_Seen_Date']).dt.days
    pooled = pooled.drop(columns=['First_Seen_Date'])

    # --- I4対応: Days_since_first_seen が打ち切られた初期会合を除外 ---
    # Absolute_Meeting_ID < 8 の会合はデータ開始前に初観測されており、
    # Days_since_first_seen が実際より小さい不正確な値になる。
    # これらはゼロ金利時代初期のデータであり、除外しても学習への影響は軽微。
    # Is_Tenor_OIS=1 の行（T12/T18/T24）は Absolute_Meeting_ID=NaN のため影響なし。
    truncated_mask = (pooled['Is_Tenor_OIS'] == 0) & (pooled['Absolute_Meeting_ID'] < 8)
    pooled = pooled[~truncated_mask].reset_index(drop=True)

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

    # 4. is_post_mpm は processing.py で生成済み → id_cols 経由で melt 後も pooled に存在する

    # 5. カラムの整理
    boj_spread_cols  = [f'M{i}_spread'     for i in range(1, 9)]
    boj_fd_cols      = [f'M{i}_frac_diff'  for i in range(1, 9)]
    boj_imp_cols     = [f'M{i}_is_imputed' for i in range(1, 9)]
    tenor_spread_cols = [f'{c}_spread'     for c in tenor_rate_cols]
    tenor_fd_cols     = [f'{c}_frac_diff'  for c in tenor_rate_cols]
    tenor_imp_cols    = [f'{c}_is_imputed' for c in tenor_rate_cols]
    ext_fd_cols      = ['USDJPY_frac_diff', 'JGB_Future_frac_diff', 'Nikkei225_frac_diff', 'DXY_frac_diff']

    basic_cols  = ['Date', 'Rate_Label', 'Meeting_Index', 'Is_Tenor_OIS',
                   'is_post_mpm', 'Days_to_MPM', 'Actual_Policy_Rate',
                   'Absolute_Meeting_ID', 'Days_since_first_seen']
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


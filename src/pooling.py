import pandas as pd
import numpy as np

def pool_boj_data(df):
    """
    設計仕様に基づきデータをプーリング（縦持ち変換）し、ターゲットを生成する。
    """
    # 1. 縦持ちに変換（M1〜M8のレート値をターゲット計算用に保持）
    id_cols = [col for col in df.columns if not col.startswith('M') or '_is_imputed' in col or '_spread' in col or '_frac_diff' in col]
    # M1〜M8の列を抽出（純粋なレート）
    boj_rate_cols = [f'M{i}' for i in range(1, 9)]
    
    # melt
    pooled = df.melt(id_vars=id_cols, value_vars=boj_rate_cols, var_name='M_Label', value_name='M_Value')
    pooled['Meeting_Index'] = pooled['M_Label'].str.extract(r'(\d+)').astype(int)
    
    # 2. 目的変数の生成 (Meeting_Indexグループ内でshift)
    pooled = pooled.sort_values(['Meeting_Index', 'Date']).reset_index(drop=True)
    
    for h in [1, 3, 5]:
        pooled[f'Target_{h}d'] = pooled.groupby('Meeting_Index')['M_Value'].shift(-h) - pooled['M_Value']
        
    # 3. MPM当日・直後5日間の除外フラグを追加
    # 注：横持ち段階で計算しても良いが、縦持ち後でもMeeting_Indexごとに計算すれば同じ（もしくは共通カラムとして保持）
    # ここでは共通フラグとして作成
    # DateごとのIs_Meeting_Dayを取得（横持ちから）
    df_date_meeting = df[['Date', 'Is_Meeting_Day']].copy()
    # 当日含め直後5日間 = rolling(6)
    df_date_meeting['is_post_mpm'] = (df_date_meeting['Is_Meeting_Day'].rolling(window=6, min_periods=1).max() == 1).astype(int)
    
    # マージして結合
    pooled = pd.merge(pooled, df_date_meeting[['Date', 'is_post_mpm']], on='Date', how='left')
    
    # 4. カラムの整理
    # 仕様書で指定された列のみを抽出
    boj_spread_cols = [f'M{i}_spread' for i in range(1, 9)]
    boj_fd_cols = [f'M{i}_frac_diff' for i in range(1, 9)]
    boj_imp_cols = [f'M{i}_is_imputed' for i in range(1, 9)]
    ext_fd_cols = ['USDJPY_frac_diff', 'JGB_Future_frac_diff', 'Nikkei225_frac_diff']
    basic_cols = ['Date', 'Meeting_Index', 'is_post_mpm', 'Days_to_MPM', 'Actual_Policy_Rate']
    target_cols = ['Target_1d', 'Target_3d', 'Target_5d']
    
    # 追加特徴量（設計書 04）
    curve_cols = ['Curve_Slope'] + [f'Butterfly_M{n}' for n in range(2, 8)]
    curve_fd_cols = [f'{c}_frac_diff' for c in curve_cols]
    cyclic_cols = ['DayOfWeek_sin', 'DayOfWeek_cos', 'Days_to_MPM_sin', 'Days_to_MPM_cos']
    
    final_cols = (
        basic_cols + boj_spread_cols + boj_fd_cols + boj_imp_cols + ext_fd_cols + 
        curve_cols + curve_fd_cols + cyclic_cols + target_cols
    )
    
    # 存在する列のみを選択（念のため）
    available_cols = [c for c in final_cols if c in pooled.columns]
    
    return pooled[available_cols]

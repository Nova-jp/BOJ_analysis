import pandas as pd
import numpy as np
from src.features import frac_diff  # 分数階差の実装を共有


def generate_rv_features(df, d=0.4, window=50):
    """
    RV Curve / RV Butterfly モデル共通の特徴量を生成する。

    入力: load_and_clean_data() の出力（M1-M8, Actual_Policy_Rate, 外部指標 等）
    出力: 以下を追加した DataFrame
        - M{n}_spread   : M{n} - Actual_Policy_Rate（n=1..8）
        - C{n}          : M{n}_spread - M{n+1}_spread（n=1..7）隣接差分
        - C{n}_frac_diff: 上記の分数階差
        - B{n}          : 2*M{n}_spread - M{n-1}_spread - M{n+1}_spread（n=2..7）バタフライ
        - B{n}_frac_diff: 上記の分数階差
        - M1_frac_diff  : M1_spread の分数階差（アンカー用）
        - Slope_M1M8    : M1_spread - M8_spread（バタフライモデルのコンテキスト用）
        - Slope_M1M8_frac_diff
        - USDJPY/JGB_Future/Nikkei225/DXY _frac_diff: 外部指標

    Note:
        - features.py の generate_features() とは独立したパイプライン。
        - M{n}_is_imputed フラグは processing.py で付与済みのためここでは生成しない。
    """
    feat_df = df.copy()

    # 1. スプレッド = M{n} - Actual_Policy_Rate（C/B の基底）
    for n in range(1, 9):
        feat_df[f'M{n}_spread'] = feat_df[f'M{n}'] - feat_df['Actual_Policy_Rate']

    # 2. 隣接差分 C{n} = M{n}_spread - M{n+1}_spread（n=1..7）
    for n in range(1, 8):
        feat_df[f'C{n}'] = feat_df[f'M{n}_spread'] - feat_df[f'M{n+1}_spread']
        feat_df[f'C{n}_frac_diff'] = frac_diff(feat_df[f'C{n}'], d=d, window=window)

    # 3. バタフライ B{n} = 2*M{n}_spread - M{n-1}_spread - M{n+1}_spread（n=2..7）
    for n in range(2, 8):
        feat_df[f'B{n}'] = (
            2 * feat_df[f'M{n}_spread']
            - feat_df[f'M{n-1}_spread']
            - feat_df[f'M{n+1}_spread']
        )
        feat_df[f'B{n}_frac_diff'] = frac_diff(feat_df[f'B{n}'], d=d, window=window)

    # 4. M1 アンカー（両モデルで保持：金利水準全体のコンテキスト）
    feat_df['M1_frac_diff'] = frac_diff(feat_df['M1_spread'], d=d, window=window)

    # 5. M1-M8 スロープ（バタフライモデルで保持：スロープとバタフライは相関する）
    feat_df['Slope_M1M8'] = feat_df['M1_spread'] - feat_df['M8_spread']
    feat_df['Slope_M1M8_frac_diff'] = frac_diff(feat_df['Slope_M1M8'], d=d, window=window)

    # 6. 外部指標の分数階差
    ext_cols = ['USDJPY', 'JGB_Future', 'Nikkei225', 'DXY']
    for col in ext_cols:
        if col in feat_df.columns:
            feat_df[f'{col}_frac_diff'] = frac_diff(feat_df[col], d=d, window=window)

    return feat_df

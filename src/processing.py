import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

def load_and_clean_data(excel_path, meeting_csv_path):
    """
    MICEを用いた補完、フラグ生成、および会合日フラグの作成を確実に行う
    """
    # 1. 生データの読み込み
    df_raw = pd.read_excel(excel_path)
    df = df_raw.iloc[1:].copy()
    df['日付'] = pd.to_datetime(df['日付'], format='%Y年%m月%d日')
    df = df.sort_values('日付').reset_index(drop=True)

    # 1D_OISは除外
    rename_dict = {
        'JPBOJ1ONI=TRDT (MID_PRICE)': 'M1', 'JPBOJ2ONI=TRDT (MID_PRICE)': 'M2',
        'JPBOJ3ONI=TRDT (MID_PRICE)': 'M3', 'JPBOJ4ONI=TRDT (MID_PRICE)': 'M4',
        'JPBOJ5ONI=TRDT (MID_PRICE)': 'M5', 'JPBOJ6ONI=TRDT (MID_PRICE)': 'M6',
        'JPBOJ7ONI=TRDT (MID_PRICE)': 'M7', 'JPBOJ8ONI=TRDT (MID_PRICE)': 'M8',
        'JPY= (MID_PRICE)': 'USDJPY', 'JGBc1 (TRDPRC_1)': 'JGB_Future',
        '.N225 (TRDPRC_1)': 'Nikkei225', '.DXY (TRDPRC_1)': 'DXY'
    }
    df = df.rename(columns=rename_dict)
    
    # 1D_OISカラムがあれば削除
    if 'JPY1DOIS=ICAP (MID_PRICE)' in df.columns:
        df = df.drop(columns=['JPY1DOIS=ICAP (MID_PRICE)'])

    # 数値化
    numeric_cols = df.columns.drop('日付')
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 指標算出
    df['JPY_Effective'] = df['DXY'] / df['USDJPY']
    
    # 2. 会合履歴のマージと会合日フラグ (Is_Meeting_Day) の作成
    df_meetings = pd.read_csv(meeting_csv_path)
    df_meetings['Date'] = pd.to_datetime(df_meetings['Date'])
    
    # マージ
    df = pd.merge(df, df_meetings, left_on='日付', right_on='Date', how='left')
    
    # 会合日フラグ (EventがNaNでない日を1とする)
    df['Is_Meeting_Day'] = df['Event'].notnull().astype(int)
    
    # 政策金利の埋め合わせ
    df['Actual_Policy_Rate'] = df['Policy_Rate'].ffill()

    # 3. 補完フラグの作成
    boj_cols = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
    for col in boj_cols:
        df[f'{col}_is_imputed'] = df[col].isnull().astype(int)

    # 4. MICE補完 (数値データのみを対象とし、フラグ等は除外)
    impute_target_cols = boj_cols + ['USDJPY', 'DXY', 'JPY_Effective', 'JGB_Future', 'Nikkei225', 'Actual_Policy_Rate']
    available_cols = [c for c in impute_target_cols if c in df.columns]
    
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)
    imputed_values = imputer.fit_transform(df[available_cols])
    
    # 補完値を元のDataFrameに書き戻す
    df.loc[:, available_cols] = imputed_values

    # 日付型変換と名称変更
    if '日付' in df.columns:
        df['Date'] = pd.to_datetime(df['日付'])
        df = df.drop(columns=['日付'])
    
    return df

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

def load_and_clean_data(excel_path, meeting_csv_path):
    """
    MICEを用いた補完、フラグ生成、および会合日フラグの作成を行う。

    対象レート:
      - M1〜M8: BOJ会合先OISスワップ（会合日付き先物レート）
      - T12/T18/T24: 12m/18m/24mテナーOIS（スポットレート）

    Returns:
        pd.DataFrame: 指定されたカラムを持つDataFrame
    """
    # 1. 生データの読み込み
    df_raw = pd.read_excel(excel_path)
    # 最初の行がヘッダーの一部（単位など）の場合があるため調整
    df = df_raw.iloc[1:].copy()
    df['日付'] = pd.to_datetime(df['日付'], format='%Y年%m月%d日')
    df = df.sort_values('日付').reset_index(drop=True)

    # カラム名のリネーム
    rename_dict = {
        'JPBOJ1ONI=TRDT (MID_PRICE)': 'M1',
        'JPBOJ2ONI=TRDT (MID_PRICE)': 'M2',
        'JPBOJ3ONI=TRDT (MID_PRICE)': 'M3',
        'JPBOJ4ONI=TRDT (MID_PRICE)': 'M4',
        'JPBOJ5ONI=TRDT (MID_PRICE)': 'M5',
        'JPBOJ6ONI=TRDT (MID_PRICE)': 'M6',
        'JPBOJ7ONI=TRDT (MID_PRICE)': 'M7',
        'JPBOJ8ONI=TRDT (MID_PRICE)': 'M8',
        'JPY= (MID_PRICE)': 'USDJPY',
        'JGBc1 (TRDPRC_1)': 'JGB_Future',
        '.N225 (TRDPRC_1)': 'Nikkei225',
        '.DXY (TRDPRC_1)': 'DXY',
        'JP12MONI=TRDT (BID)': 'T12',
        'JP18MONI=TRDT (BID)': 'T18',
        'JP24MONI=TRDT (BID)': 'T24',
    }
    df = df.rename(columns=rename_dict)

    # 不要なカラムの削除（OIS_1Dは使用しない）
    cols_to_keep = ['日付', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8',
                    'USDJPY', 'JGB_Future', 'Nikkei225', 'DXY',
                    'T12', 'T18', 'T24']
    df = df[[c for c in cols_to_keep if c in df.columns]].copy()

    # 数値化
    numeric_cols = df.columns.drop('日付')
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. 会合履歴のマージ
    df_meetings = pd.read_csv(meeting_csv_path)
    df_meetings['Date'] = pd.to_datetime(df_meetings['Date'])

    df = pd.merge(df, df_meetings[['Date', 'Policy_Rate', 'Event']], left_on='日付', right_on='Date', how='left', suffixes=('', '_mtg'))
    df = df.drop(columns=['Date']).rename(columns={'日付': 'Date'})

    # 会合日フラグ (Is_Meeting_Day)
    df['Is_Meeting_Day'] = df['Event'].notnull().astype(int)

    # 政策金利の前方・後方補完 (Actual_Policy_Rate)
    df['Actual_Policy_Rate'] = df['Policy_Rate'].ffill().bfill()

    # 3. 補完フラグの作成 (MICE補完前)
    # M1-M8: 会合先BOJスワップ、T12/T18/T24: テナーOIS（両方とも予測対象）
    all_rate_cols = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'T12', 'T18', 'T24']
    all_rate_cols = [c for c in all_rate_cols if c in df.columns]
    for col in all_rate_cols:
        df[f'{col}_is_imputed'] = df[col].isnull().astype(int)

    # 4. MICE補完
    impute_target_cols = all_rate_cols + ['USDJPY', 'JGB_Future', 'Nikkei225', 'DXY']
    impute_target_cols = [c for c in impute_target_cols if c in df.columns]

    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)
    df[impute_target_cols] = imputer.fit_transform(df[impute_target_cols])

    # 5. Days_to_MPM（会合CSV全日付を使うため、未来の予定会合も含む）
    all_meeting_dates = sorted(df_meetings['Date'].unique())

    def days_to_next_mpm(date):
        future = [m for m in all_meeting_dates if m >= date]
        return (future[0] - date).days if future else np.nan

    date_to_days = {d: days_to_next_mpm(d) for d in df['Date'].unique()}
    df['Days_to_MPM'] = df['Date'].map(date_to_days)

    # 必要な列のみを選択して返す
    tenor_cols = [c for c in ['T12', 'T18', 'T24'] if c in df.columns]
    ext_cols = [c for c in ['USDJPY', 'JGB_Future', 'Nikkei225', 'DXY'] if c in df.columns]
    final_cols = (['Date', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'] +
                  tenor_cols + ext_cols +
                  ['Actual_Policy_Rate', 'Is_Meeting_Day', 'Days_to_MPM'] +
                  [f'{c}_is_imputed' for c in all_rate_cols])

    return df[final_cols]

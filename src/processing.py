import bisect
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

def load_meeting_dates(meeting_csv_path):
    """
    MPM 会合日一覧を昇順で返す。

    pool_boj_data の Absolute_Meeting_ID 計算など、
    会合日リストを直接参照する箇所で使用する。

    Returns:
        list[pd.Timestamp]: 会合日の昇順リスト
    """
    df_meetings = pd.read_csv(meeting_csv_path)
    df_meetings['Date'] = pd.to_datetime(df_meetings['Date'])
    return sorted(df_meetings['Date'].unique())


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
    # Excelの2行目（index=1）はヘッダー直下の不要行（単位・注釈等）のため除去。
    # Reuters Eikon エクスポート形式の固定仕様。フォーマットが変わった場合はここを要確認。
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

    # 政策金利の補完 (Actual_Policy_Rate)
    # ffill(): 各会合の決定金利を次の会合まで前方補完。
    # bfill(): データ開始〜最初の会合日の間（通常数日〜数週間）を補完。
    # 注意: 以前は bfill() によりデータ先頭に未来の金利が混入するリークがあったが、
    # BOJ_meeting_history.csv が 2007年1月まで整備された現在は安全。
    # bfill() が埋める期間は「データ開始〜最初の会合日」のみで、率も正しい。
    df['Actual_Policy_Rate'] = df['Policy_Rate'].ffill().bfill()

    # 3. 補完フラグの作成 (MICE補完前)
    # M1-M8: 会合先BOJスワップ、T12/T18/T24: テナーOIS（両方とも予測対象）
    all_rate_cols = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'T12', 'T18', 'T24']
    all_rate_cols = [c for c in all_rate_cols if c in df.columns]
    for col in all_rate_cols:
        df[f'{col}_is_imputed'] = df[col].isnull().astype(int)

    # 4. MICE補完
    # 【設計上の注意】imputer を全期間データで fit_transform している（フォールド外情報を含む）。
    # フォールド内 fit 化が理想だが、以下の理由で全期間 fit を許容する:
    #   - 欠損は主に休日・データ欠落由来で欠損率が低い（影響行が少ない）
    #   - BayesianRidge が学習する「金利間の線形相関」は長期的に安定しており、
    #     フォールド別 fit でも結果がほぼ変わらない
    #   - {col}_is_imputed フラグで補完箇所をモデルに伝えているため学習時に識別可能
    #   - 特徴量重要度分析でも is_imputed フラグは上位に出ず、補完の歪みが
    #     予測に影響していないことを確認済み（2026-03-21）
    impute_target_cols = all_rate_cols + ['USDJPY', 'JGB_Future', 'Nikkei225', 'DXY']
    impute_target_cols = [c for c in impute_target_cols if c in df.columns]

    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)
    df[impute_target_cols] = imputer.fit_transform(df[impute_target_cols])

    # 5. Days_to_MPM（会合CSV全日付を使うため、未来の予定会合も含む）
    all_meeting_dates = sorted(df_meetings['Date'].unique())

    def days_to_next_mpm(date):
        # bisect_left で O(log N) の二分探索
        idx = bisect.bisect_left(all_meeting_dates, date)
        if idx < len(all_meeting_dates):
            return (all_meeting_dates[idx] - date).days
        return np.nan

    date_to_days = {d: days_to_next_mpm(d) for d in df['Date'].unique()}
    df['Days_to_MPM'] = df['Date'].map(date_to_days)

    # 6. is_post_mpm フラグ（MPM当日 + 翌5日間）
    # rolling(window=6) で「今日含む過去6日のうちに会合日があるか」を判定。
    # pooling*.py の3ファイルに同一ロジックが重複していたためここに集約。
    df['is_post_mpm'] = (
        df['Is_Meeting_Day'].rolling(window=6, min_periods=1).max() == 1
    ).astype(int)

    # 必要な列のみを選択して返す
    tenor_cols = [c for c in ['T12', 'T18', 'T24'] if c in df.columns]
    ext_cols = [c for c in ['USDJPY', 'JGB_Future', 'Nikkei225', 'DXY'] if c in df.columns]
    final_cols = (['Date', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'] +
                  tenor_cols + ext_cols +
                  ['Actual_Policy_Rate', 'Is_Meeting_Day', 'is_post_mpm', 'Days_to_MPM'] +
                  [f'{c}_is_imputed' for c in all_rate_cols])

    return df[final_cols]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages
from catboost import CatBoostRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# ポップアップを防ぐためのバックエンド設定
import matplotlib
matplotlib.use('Agg')

# ==========================================
# 1. データ処理・特徴量生成ロジック (Internal)
# ==========================================

def frac_diff_func(series, d, window=50):
    w = [1.0]
    for k in range(1, window):
        w.append(-w[-1] * (d - k + 1) / k)
    weights = np.array(w)
    return series.rolling(window=window).apply(lambda v: np.sum(v * weights[::-1]), raw=True)

def get_frac_diff_weights(d, window):
    w = [1.0]
    for k in range(1, window):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)

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
        '.N225 (TRDPRC_1)': 'Nikkei225', '.DXY (TRDPRC_1)': 'DXY'
    }
    df = df.rename(columns=rename_dict)
    if 'JPY1DOIS=ICAP (MID_PRICE)' in df.columns:
        df = df.drop(columns=['JPY1DOIS=ICAP (MID_PRICE)'])

    numeric_cols = df.columns.drop('日付')
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['JPY_Effective'] = df['DXY'] / df['USDJPY']
    df_meetings = pd.read_csv(meeting_csv_path)
    df_meetings['Date'] = pd.to_datetime(df_meetings['Date'])
    df = pd.merge(df, df_meetings, left_on='日付', right_on='Date', how='left')
    df['Is_Meeting_Day'] = df['Event'].notnull().astype(int)
    df['Actual_Policy_Rate'] = df['Policy_Rate'].ffill()

    boj_cols = [f'M{i}' for i in range(1, 9)]
    for col in boj_cols:
        df[f'{col}_is_imputed'] = df[col].isnull().astype(int)

    impute_target_cols = boj_cols + ['USDJPY', 'DXY', 'JPY_Effective', 'JGB_Future', 'Nikkei225', 'Actual_Policy_Rate']
    available_cols = [c for c in impute_target_cols if c in df.columns]
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)
    df.loc[:, available_cols] = imputer.fit_transform(df[available_cols])

    df['Date'] = pd.to_datetime(df['日付'])
    return df.drop(columns=['日付'])

def pool_boj_data(df, mpm_dates, d=0.4, window=50, max_n=5):
    df = df.copy()
    boj_cols = [f'M{i}' for i in range(1, 9)]
    weights = get_frac_diff_weights(d, window)
    
    ext_cols = ['JGB_Future', 'USDJPY', 'Nikkei225', 'JPY_Effective']
    common_feats = []
    for col in ext_cols:
        if col in df.columns:
            df[f'{col}_FracDiff'] = frac_diff_func(df[col], d=d, window=window)
            common_feats.append(f'{col}_FracDiff')
            
    for i in range(1, 9):
        c = f'M{i}_is_imputed'
        if c in df.columns:
            df[f'M{i}_Consec_Imp'] = df[c].groupby((df[c] != df[c].shift()).cumsum()).cumcount() + 1 * df[c]
            common_feats.append(f'M{i}_Consec_Imp')

    id_vars = ['Date', 'Actual_Policy_Rate', 'Is_Meeting_Day'] + common_feats
    pooled = df[id_vars + boj_cols].melt(id_vars=id_vars, value_vars=boj_cols, var_name='M_Label', value_name='Swap_Rate')
    pooled['Meeting_Index'] = pooled['M_Label'].str.extract(r'(\d+)').astype(int)
    pooled = pooled.sort_values(['Meeting_Index', 'Date']).reset_index(drop=True)
    pooled['Swap_Rate_FracDiff'] = pooled.groupby('Meeting_Index')['Swap_Rate'].transform(lambda x: frac_diff_func(x, d=d, window=window))
    
    def calc_mem(x):
        return x.rolling(window=window-1).apply(lambda v: np.sum(v * weights[1:][::-1]), raw=True)

    for n in range(1, max_n + 1):
        pooled[f'Target_FD_{n}d'] = pooled.groupby('Meeting_Index')['Swap_Rate_FracDiff'].shift(-n)
        pooled[f'Mem_{n}d'] = pooled.groupby('Meeting_Index')['Swap_Rate'].transform(calc_mem).shift(-n)
        pooled[f'Actual_{n}d'] = pooled.groupby('Meeting_Index')['Swap_Rate'].shift(-n)

    mpm_dates = pd.to_datetime(mpm_dates).sort_values()
    date_map = {d: (mpm_dates[mpm_dates > d].iloc[0] - d).days if any(mpm_dates > d) else np.nan for d in pooled['Date'].unique()}
    pooled['Days_to_MPM'] = pooled['Date'].map(date_map)
    return pooled.drop(columns=['M_Label'])

# ==========================================
# 2. メイン分析プロセス
# ==========================================

def run_full_analysis():
    print("Starting Analysis Process...")
    sns.set_theme(style='whitegrid')
    plt.rcParams['font.family'] = 'Hiragino Sans' # Mac
    if not any(f in plt.rcParams['font.family'] for f in ['Hiragino Sans', 'MS Gothic', 'sans-serif']):
        plt.rcParams['font.family'] = 'sans-serif'

    if not os.path.exists('outputs'): os.makedirs('outputs')
    pdf_path = 'outputs/analysis_report.pdf'

    # データロード
    MAX_N = 5
    df_cleaned = load_and_clean_data('data/BOJ_data.xlsx', 'data/BOJ_meeting_history.csv')
    mpm_dates = pd.to_datetime(pd.read_csv('data/BOJ_meeting_history.csv')['Date'])
    df_pooled = pool_boj_data(df_cleaned, mpm_dates=mpm_dates, max_n=MAX_N)

    # 特徴量準備
    features = [c for c in df_pooled.columns if 'FracDiff' in c or 'Consec_Imp' in c or c in ['Days_to_MPM', 'Meeting_Index']]
    train_final = df_pooled.dropna(subset=[f'Target_FD_{MAX_N}d'] + features)
    split_date = sorted(train_final['Date'].unique())[int(len(train_final['Date'].unique()) * 0.8)]
    test_df = df_pooled[df_pooled['Date'] >= split_date].copy()

    # 学習 & 予測
    models = {}
    print(f"Training models for horizons n=[1, 3, 5]...")
    for n in [1, 3, 5]:
        m = CatBoostRegressor(iterations=800, learning_rate=0.05, verbose=0, random_seed=42)
        m.fit(train_final[train_final['Date'] < split_date][features], 
              train_final[train_final['Date'] < split_date][f'Target_FD_{n}d'])
        test_df[f'Pred_Rate_{n}d'] = m.predict(test_df[features]) - test_df[f'Mem_{n}d'].ffill()
        models[n] = m

    with PdfPages(pdf_path) as pdf:
        print(f"Generating PDF report: {pdf_path}")

        # --- 1. M1 to M8 Levels (8 graphs, 2 pages) ---
        for p in range(2):
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            for i, ax in enumerate(axes.flatten()):
                m_idx = p * 4 + i + 1
                col = f'M{m_idx}'
                ax.plot(df_cleaned['Date'], df_cleaned[col], color='darkblue')
                ax.set_title(f'{col} Level')
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # --- 2. Adjacent Spreads (M2-M1...M8-M7 / 7 graphs, 2 pages) ---
        adj_pairs = [(f'M{i+1}', f'M{i}') for i in range(1, 8)]
        for p in range(2):
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            for i, ax in enumerate(axes.flatten()):
                idx = p * 4 + i
                if idx < len(adj_pairs):
                    m_high, m_low = adj_pairs[idx]
                    ax.plot(df_cleaned['Date'], df_cleaned[m_high] - df_cleaned[m_low], color='darkred')
                    ax.set_title(f'Spread {m_high}-{m_low}')
                else:
                    ax.axis('off')
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # --- 3. All Curve Spreads (Gaps >= 2) ---
        curve_pairs = []
        for gap in range(2, 8):
            for i in range(1, 9 - gap):
                curve_pairs.append((f'M{i+gap}', f'M{i}'))
        
        for i in range(0, len(curve_pairs), 4):
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            for j, ax in enumerate(axes.flatten()):
                idx = i + j
                if idx < len(curve_pairs):
                    m_high, m_low = curve_pairs[idx]
                    ax.plot(df_cleaned['Date'], df_cleaned[m_high] - df_cleaned[m_low], color='purple')
                    ax.set_title(f'Curve Spread {m_high}-{m_low}')
                else:
                    ax.axis('off')
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # --- 4. Strategy Results (Trajectory & PnL for M1, M5) ---
        for m_idx in [1, 5]:
            m_df = test_df[test_df['Meeting_Index'] == m_idx].copy().sort_values('Date')
            m_df['Pred_1d_lag1'] = m_df['Pred_Rate_1d'].shift(1)
            cond_short = (m_df['Pred_1d_lag1'] < m_df['Pred_Rate_1d']) & (m_df['Pred_Rate_1d'] < m_df['Pred_Rate_3d']) & (m_df['Pred_Rate_3d'] < m_df['Pred_Rate_5d'])
            cond_long = (m_df['Pred_1d_lag1'] > m_df['Pred_Rate_1d']) & (m_df['Pred_Rate_1d'] > m_df['Pred_Rate_3d']) & (m_df['Pred_Rate_3d'] > m_df['Pred_Rate_5d'])
            m_df['Signal'] = 0
            m_df.loc[cond_long, 'Signal'] = 1; m_df.loc[cond_short, 'Signal'] = -1
            m_df['PnL'] = m_df['Signal'] * (m_df['Swap_Rate'] - m_df['Actual_5d'])
            
            latest_t = m_df[m_df['Swap_Rate'].notnull()]['Date'].max()
            ext_dates = pd.bdate_range(start=latest_t + pd.Timedelta(days=1), periods=5)
            df_ext = pd.concat([m_df, pd.DataFrame({'Date': ext_dates})], ignore_index=True).sort_values('Date')
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
            ax1.plot(df_ext['Date'], df_ext['Swap_Rate'], color='black', linewidth=3, label='Actual')
            colors = sns.color_palette("Reds", 3)
            for i, n in enumerate([1, 3, 5]):
                ax1.plot(df_ext['Date'], df_ext[f'Pred_Rate_{n}d'].shift(n), label=f'Pred {n}d ago', linestyle='--', color=colors[i], alpha=0.7)
            longs = df_ext[df_ext['Signal'] == 1]; shorts = df_ext[df_ext['Signal'] == -1]
            ax1.scatter(longs['Date'], longs['Swap_Rate'], marker='^', color='blue', s=100, label='Long(Recv)', zorder=5)
            ax1.scatter(shorts['Date'], shorts['Swap_Rate'], marker='v', color='red', s=100, label='Short(Pay)', zorder=5)
            ax1.set_title(f'M{m_idx} Trajectory & Trend Signals', fontsize=16); ax1.legend(bbox_to_anchor=(1.05, 1))
            ax2.plot(df_ext['Date'], df_ext['PnL'].fillna(0).cumsum(), color='green', linewidth=2)
            ax2.fill_between(df_ext['Date'], 0, df_ext['PnL'].fillna(0).cumsum(), color='green', alpha=0.1)
            ax2.set_title(f'M{m_idx} Cumulative PnL (5d Unwind)', fontsize=13)
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # --- 5. Feature Importance ---
        imp_df = pd.DataFrame({'Feature': features, 'Importance': models[5].get_feature_importance()}).sort_values('Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=imp_df.head(20), x='Importance', y='Feature', palette='viridis', ax=ax)
        ax.set_title('Top 20 Feature Importance (n=5d Model)')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    print(f"Analysis complete. Report saved as {pdf_path}")

if __name__ == "__main__":
    run_full_analysis()

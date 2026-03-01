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

# Disable popup and set PDF settings
import matplotlib
matplotlib.use('Agg')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'

# ==========================================
# 1. Processing Logic
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
    if '日付' in df.columns:
        df['Date_Raw'] = pd.to_datetime(df['日付'], format='%Y年%m月%d日', errors='coerce')
        df = df.sort_values('Date_Raw').reset_index(drop=True)
    else:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df = df.sort_values(df.columns[0]).reset_index(drop=True)
        df['Date_Raw'] = df.iloc[:, 0]

    rename_dict = {
        'JPBOJ1ONI=TRDT (MID_PRICE)': 'M1', 'JPBOJ2ONI=TRDT (MID_PRICE)': 'M2',
        'JPBOJ3ONI=TRDT (MID_PRICE)': 'M3', 'JPBOJ4ONI=TRDT (MID_PRICE)': 'M4',
        'JPBOJ5ONI=TRDT (MID_PRICE)': 'M5', 'JPBOJ6ONI=TRDT (MID_PRICE)': 'M6',
        'JPBOJ7ONI=TRDT (MID_PRICE)': 'M7', 'JPBOJ8ONI=TRDT (MID_PRICE)': 'M8',
        'JPY= (MID_PRICE)': 'USDJPY', 'JGBc1 (TRDPRC_1)': 'JGB_Future',
        '.N225 (TRDPRC_1)': 'Nikkei225', '.DXY (TRDPRC_1)': 'DXY'
    }
    df = df.rename(columns=rename_dict)
    cols_to_drop = [c for c in df.columns if 'JPY1DOIS' in c or '日付' in c]
    df = df.drop(columns=cols_to_drop)

    numeric_cols = df.columns.drop('Date_Raw')
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['JPY_Effective'] = df['DXY'] / df['USDJPY']
    df_meetings = pd.read_csv(meeting_csv_path)
    df_meetings['Date'] = pd.to_datetime(df_meetings['Date'])
    df = pd.merge(df, df_meetings, left_on='Date_Raw', right_on='Date', how='left')
    df['Is_Meeting_Day'] = df['Event'].notnull().astype(int)
    df['Actual_Policy_Rate'] = df['Policy_Rate'].ffill()

    boj_cols = [f'M{i}' for i in range(1, 9)]
    for col in boj_cols:
        df[f'{col}_is_imputed'] = df[col].isnull().astype(int)

    impute_target_cols = boj_cols + ['USDJPY', 'DXY', 'JPY_Effective', 'JGB_Future', 'Nikkei225', 'Actual_Policy_Rate']
    available_cols = [c for c in impute_target_cols if c in df.columns]
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)
    df.loc[:, available_cols] = imputer.fit_transform(df[available_cols])

    df['Date'] = df['Date_Raw']
    return df.drop(columns=['Date_Raw'])

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
# 2. Main Analysis and Strategy Plotting
# ==========================================

def run_full_analysis():
    print("Starting Analysis Process...")
    sns.set_theme(style='whitegrid')
    if not os.path.exists('outputs'): os.makedirs('outputs')
    pdf_path = 'outputs/analysis_report.pdf'

    # Load Data
    MAX_N = 5
    df_cleaned = load_and_clean_data('data/BOJ_data.xlsx', 'data/BOJ_meeting_history.csv')
    mpm_dates = pd.to_datetime(pd.read_csv('data/BOJ_meeting_history.csv')['Date'])
    df_pooled = pool_boj_data(df_cleaned, mpm_dates=mpm_dates, max_n=MAX_N)

    # Features and Split
    features = [c for c in df_pooled.columns if 'FracDiff' in c or 'Consec_Imp' in c or c in ['Days_to_MPM', 'Meeting_Index']]
    train_final = df_pooled.dropna(subset=[f'Target_FD_{MAX_N}d'] + features)
    split_date = sorted(train_final['Date'].unique())[int(len(train_final['Date'].unique()) * 0.8)]
    test_df = df_pooled[df_pooled['Date'] >= split_date].copy()

    # Model Training and Prediction for all M-Indices
    print("Training models for all Meeting Indices...")
    for n in [1, 3, 5]:
        m = CatBoostRegressor(iterations=800, learning_rate=0.05, verbose=0, random_seed=42)
        m.fit(train_final[train_final['Date'] < split_date][features], 
              train_final[train_final['Date'] < split_date][f'Target_FD_{n}d'])
        test_df[f'Pred_Rate_{n}d'] = m.predict(test_df[features]) - test_df[f'Mem_{n}d'].ffill()

    # Helper to plot a strategy block
    def plot_strategy_block(ax_price, ax_pnl, df_series, title):
        df = df_series.copy().sort_values('Date')
        df['Pred_1d_lag1'] = df['Pred_Rate_1d'].shift(1)
        
        # Trend Logic
        cond_short = (df['Pred_1d_lag1'] < df['Pred_Rate_1d']) & (df['Pred_Rate_1d'] < df['Pred_Rate_3d']) & (df['Pred_Rate_3d'] < df['Pred_Rate_5d'])
        cond_long = (df['Pred_1d_lag1'] > df['Pred_Rate_1d']) & (df['Pred_Rate_1d'] > df['Pred_Rate_3d']) & (df['Pred_Rate_3d'] > df['Pred_Rate_5d'])
        
        df['Signal'] = 0
        df.loc[cond_long, 'Signal'] = 1; df.loc[cond_short, 'Signal'] = -1
        df['PnL'] = df['Signal'] * (df['Actual_Val'] - df['Actual_5d_Val'])
        
        # Plot Trajectory
        ax_price.plot(df['Date'], df['Actual_Val'], color='black', linewidth=1.5, label='Actual')
        colors = sns.color_palette("Reds", 3)
        for i, n in enumerate([1, 3, 5]):
            ax_price.plot(df['Date'], df[f'Pred_Rate_{n}d'].shift(n), linestyle='--', color=colors[i], alpha=0.5, linewidth=0.8)
        
        longs = df[df['Signal'] == 1]; shorts = df[df['Signal'] == -1]
        ax_price.scatter(longs['Date'], longs['Actual_Val'], marker='^', color='blue', s=40, zorder=5)
        ax_price.scatter(shorts['Date'], shorts['Actual_Val'], marker='v', color='red', s=40, zorder=5)
        ax_price.set_title(title, fontsize=10)
        ax_price.tick_params(axis='both', which='major', labelsize=8)
        
        # Plot Cumulative PnL
        ax_pnl.plot(df['Date'], df['PnL'].fillna(0).cumsum(), color='green', linewidth=1)
        ax_pnl.fill_between(df['Date'], 0, df['PnL'].fillna(0).cumsum(), color='green', alpha=0.1)
        ax_pnl.set_ylabel('CumPnL', fontsize=8)
        ax_pnl.tick_params(axis='both', which='major', labelsize=8)

    # Collect items to plot
    plot_items = []
    # Levels M1-M8
    for i in range(1, 9):
        m_df = test_df[test_df['Meeting_Index'] == i].copy()
        m_df = m_df.rename(columns={'Swap_Rate': 'Actual_Val', 'Actual_5d': 'Actual_5d_Val'})
        plot_items.append((f'M{i} Strategy', m_df))
    # Adjacent Spreads
    for i in range(1, 8):
        m_hi = test_df[test_df['Meeting_Index'] == i+1].set_index('Date')
        m_lo = test_df[test_df['Meeting_Index'] == i].set_index('Date')
        s_df = (m_hi - m_lo).reset_index()
        s_df = s_df.rename(columns={'Swap_Rate': 'Actual_Val', 'Actual_5d': 'Actual_5d_Val'})
        plot_items.append((f'Spread M{i+1}-M{i}', s_df))
    # All Curve Spreads
    for gap in range(2, 8):
        for i in range(1, 9 - gap):
            m_hi = test_df[test_df['Meeting_Index'] == i+gap].set_index('Date')
            m_lo = test_df[test_df['Meeting_Index'] == i].set_index('Date')
            s_df = (m_hi - m_lo).reset_index()
            s_df = s_df.rename(columns={'Swap_Rate': 'Actual_Val', 'Actual_5d': 'Actual_5d_Val'})
            plot_items.append((f'Curve M{i+gap}-M{i}', s_df))

    # PDF Generation
    with PdfPages(pdf_path) as pdf:
        print(f"Generating PDF: {pdf_path}")
        for i in range(0, len(plot_items), 4):
            fig = plt.figure(figsize=(12, 16))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
            for j in range(4):
                if i + j >= len(plot_items): break
                title, data = plot_items[i+j]
                # Subgrid for each block (Price top, PnL bottom)
                sub_gs = gs[j // 2, j % 2].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.1)
                ax1 = fig.add_subplot(sub_gs[0])
                ax2 = fig.add_subplot(sub_gs[1], sharex=ax1)
                plot_strategy_block(ax1, ax2, data, title)
            pdf.savefig(fig); plt.close(fig)

    print(f"Analysis complete. Report saved as {pdf_path}")

if __name__ == "__main__":
    run_full_analysis()

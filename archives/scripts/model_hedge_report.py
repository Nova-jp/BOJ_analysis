"""
model_hedge_report.py
=====================
Generates a PDF report covering:
  Part 1 – Model Evaluation  (notebooks/09_model_evaluation.ipynb)
  Part 2 – Hedge Ratio       (notebooks/10_hedge_ratio.ipynb)

Usage:
    python model_hedge_report.py
Output:
    outputs/model_hedge_report.pdf
"""

import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import spearmanr

plt.rcParams['pdf.fonttype']       = 42
plt.rcParams['font.family']        = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

from src.processing import load_and_clean_data
from src.features   import generate_features
from src.pooling    import pool_boj_data
from src.modeling   import walk_forward_validation, get_features_and_target, calculate_metrics

# ── Paths & constants ────────────────────────────────────────────────────────
EXCEL_PATH  = 'data/BOJ_data.xlsx'
MEETING_CSV = 'data/BOJ_meeting_history.csv'
OUTPUT_PATH = 'outputs/model_hedge_report.pdf'
START_DATE  = '2024-01-01'

MI_LABEL     = {1:'M1',2:'M2',3:'M3',4:'M4',5:'M5',6:'M6',7:'M7',8:'M8',
                10:'T12',11:'T18',12:'T24'}
BOJ_INSTR    = [f'M{i}' for i in range(1, 9)]
TENOR_INSTR  = ['T12', 'T18', 'T24']
ALL_INSTR    = BOJ_INSTR + TENOR_INSTR
LABEL_ORDER  = ALL_INSTR

ROLL_WINDOW  = 252
BAR_WIDTH    = 0.38

# ════════════════════════════════════════════════════════════════════════════════
# 1. DATA PIPELINE
# ════════════════════════════════════════════════════════════════════════════════
print('Loading data...')
df_raw  = load_and_clean_data(EXCEL_PATH, MEETING_CSV)
df_feat = generate_features(df_raw,
                             add_weekday_cyclic=True,
                             add_days_to_mpm_cyclic=True,
                             add_curve_features=True)
df_pool = pool_boj_data(df_feat)

# ════════════════════════════════════════════════════════════════════════════════
# 2. WALK-FORWARD VALIDATION
# ════════════════════════════════════════════════════════════════════════════════
print('Running walk-forward (3d_norm)...')
res3, mdl3, *_ = walk_forward_validation(
    df_pool, 'Target_3d_norm', START_DATE, return_model=True)

print('Running walk-forward (5d_norm)...')
res5, mdl5, *_ = walk_forward_validation(
    df_pool, 'Target_5d_norm', START_DATE, return_model=True)

# ════════════════════════════════════════════════════════════════════════════════
# 3. DENORMALIZE PREDICTIONS  (norm → bps)
# ════════════════════════════════════════════════════════════════════════════════
instr_std_3d = df_pool.groupby('Rate_Label')['Target_3d'].std()  # in % units
instr_std_5d = df_pool.groupby('Rate_Label')['Target_5d'].std()

def add_bps_cols(res_df, instr_std):
    """Add Rate_Label, instr_std, Actual_bps, Pred_bps columns."""
    df = res_df.copy()
    df['Rate_Label'] = df['Meeting_Index'].map(MI_LABEL)
    df['instr_std']  = df['Rate_Label'].map(instr_std)
    df['Actual_bps'] = df['Actual'] * df['instr_std'] * 100
    df['Pred_bps']   = df['Pred']   * df['instr_std'] * 100
    return df

res3 = add_bps_cols(res3, instr_std_3d)
res5 = add_bps_cols(res5, instr_std_5d)

# attach Days_to_MPM for bucket analysis
dtm_ref = df_pool[['Date', 'Rate_Label', 'Days_to_MPM']].drop_duplicates()
res3 = res3.merge(dtm_ref, on=['Date', 'Rate_Label'], how='left')
res5 = res5.merge(dtm_ref, on=['Date', 'Rate_Label'], how='left')

# ════════════════════════════════════════════════════════════════════════════════
# 4. METRICS
# ════════════════════════════════════════════════════════════════════════════════
overall3 = calculate_metrics(res3['Actual'], res3['Pred'])
overall5 = calculate_metrics(res5['Actual'], res5['Pred'])

# IC by Rate Label
def ic_by_group(res_df, col):
    rows = []
    for grp, sub in res_df.groupby(col):
        ic, _ = spearmanr(sub['Actual'], sub['Pred'])
        rows.append({col: grp, 'IC': round(ic, 3), 'N': len(sub)})
    return pd.DataFrame(rows)

ic_lbl3 = ic_by_group(res3, 'Rate_Label')
ic_lbl5 = ic_by_group(res5, 'Rate_Label')

# IC by Days_to_MPM bucket
# bins start at -1 so Days_to_MPM=0 (meeting day) is captured in <=5d bucket
DTM_BINS   = [-1, 5, 15, 30, 9999]
DTM_LABELS = ['<=5d', '6-15d', '16-30d', '>30d']

def ic_by_dtm(res_df):
    df = res_df.assign(bucket=pd.cut(res_df['Days_to_MPM'], bins=DTM_BINS,
                                     labels=DTM_LABELS, right=True))
    rows = []
    for b, sub in df.groupby('bucket', observed=True):
        ic, _ = spearmanr(sub['Actual'], sub['Pred'])
        rows.append({'bucket': str(b), 'IC': round(ic, 3), 'N': len(sub)})
    return pd.DataFrame(rows)

ic_dtm3 = ic_by_dtm(res3)
ic_dtm5 = ic_by_dtm(res5)

# Fold IC
fold3 = (res3.groupby('Fold')
         .apply(lambda x: spearmanr(x['Actual'], x['Pred'])[0])
         .reset_index(name='IC'))
fold5 = (res5.groupby('Fold')
         .apply(lambda x: spearmanr(x['Actual'], x['Pred'])[0])
         .reset_index(name='IC'))

# ════════════════════════════════════════════════════════════════════════════════
# 5. CROSS-SECTIONAL P&L  (Top4 Long / Bottom4 Short, per day)
# ════════════════════════════════════════════════════════════════════════════════
def cs_pnl(res_df, top_n=4):
    rows = []
    for date, grp in res_df.groupby('Date'):
        if len(grp) < top_n * 2:
            continue
        grp     = grp.sort_values('Pred', ascending=False)
        longs   = grp.head(top_n)
        shorts  = grp.tail(top_n)
        rows.append({
            'Date':     date,
            'PnL_norm': longs['Actual'].mean()     - shorts['Actual'].mean(),
            'PnL_bps':  longs['Actual_bps'].mean() - shorts['Actual_bps'].mean(),
        })
    df_pnl = pd.DataFrame(rows).sort_values('Date')
    df_pnl['Cum_norm'] = df_pnl['PnL_norm'].cumsum()
    df_pnl['Cum_bps']  = df_pnl['PnL_bps'].cumsum()
    return df_pnl

pnl3 = cs_pnl(res3)
pnl5 = cs_pnl(res5)

# ════════════════════════════════════════════════════════════════════════════════
# 6. LATEST PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════════
def get_latest_preds(mdl, instr_std, target_norm_col):
    X_all, _, dates_all, _ = get_features_and_target(df_pool, target_norm_col)
    latest_date = dates_all.max()
    mask        = dates_all == latest_date
    X_today     = X_all[mask]
    if X_today.empty:
        return pd.DataFrame()
    labels    = df_pool.loc[X_today.index, 'Rate_Label']
    pred_norm = mdl.predict(X_today)
    pred_bps  = pred_norm * labels.map(instr_std).values * 100
    return pd.DataFrame({'Rate_Label': labels.values,
                         'Pred_norm':  pred_norm,
                         'Pred_bps':   pred_bps})

latest3 = get_latest_preds(mdl3, instr_std_3d, 'Target_3d_norm')
latest5 = get_latest_preds(mdl5, instr_std_5d, 'Target_5d_norm')

# ════════════════════════════════════════════════════════════════════════════════
# 7. HEDGE RATIO (OLS)
# ════════════════════════════════════════════════════════════════════════════════
spread_cols = ([f'M{i}_spread' for i in range(1, 9)] +
               [c for c in ['T12_spread', 'T18_spread', 'T24_spread']
                if c in df_feat.columns])
tenor_instr_avail = [c.replace('_spread', '') for c in spread_cols if c.startswith('T')]

delta = df_feat.set_index('Date')[spread_cols + ['Days_to_MPM']].copy()
delta[spread_cols] = delta[spread_cols].diff()
delta = delta.dropna(subset=spread_cols)
delta = delta.rename(columns={f'{inst}_spread': inst
                                for inst in BOJ_INSTR + tenor_instr_avail})

def ols_hedge(y_s, x_s):
    y = np.asarray(y_s, dtype=float)
    x = np.asarray(x_s, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(x))
    y, x = y[mask], x[mask]
    if len(y) < 30:
        return None
    slope, _, r, _, _ = stats.linregress(x, y)
    sigma_y = y.std()
    return {
        'H*':              round(slope, 4),
        'Vol Ratio':       round(sigma_y / x.std(), 4),
        'rho':             round(r, 4),
        'R2 (%)':          round(r**2 * 100, 1),
        'sigma_Mn (bps)':  round(sigma_y * 100, 4),
        'Resid std (bps)': round(sigma_y * np.sqrt(1 - r**2) * 100, 4),
        'N':               len(y),
    }

ols_dict = {inst: ols_hedge(delta[inst], delta['T12'])
            for inst in (BOJ_INSTR + tenor_instr_avail)
            if inst != 'T12' and inst in delta.columns}
df_ols = pd.DataFrame({k: v for k, v in ols_dict.items() if v}).T

boj_in_ols = [i for i in BOJ_INSTR if i in df_ols.index]

# ── DV01-adjusted notional hedge ─────────────────────────────────────────────
# Derive meeting statistics from already-loaded df_feat (avoids re-reading CSV)
_meeting_dates = df_feat.loc[df_feat['Is_Meeting_Day'] == 1, 'Date'].sort_values()
avg_interval   = _meeting_dates.diff().dt.days.dropna().mean()
avg_m1_tenor   = df_feat['Days_to_MPM'].mean()
tenor_map    = {f'M{n}': avg_m1_tenor + (n - 1) * avg_interval for n in range(1, 9)}
T12_DAYS     = 365.25

dv01_rows = []
for inst in boj_in_ols:
    h   = float(df_ols.loc[inst, 'H*'])
    rho = float(df_ols.loc[inst, 'rho'])
    r2  = float(df_ols.loc[inst, 'R2 (%)'])
    sm  = float(df_ols.loc[inst, 'sigma_Mn (bps)'])
    sr  = float(df_ols.loc[inst, 'Resid std (bps)'])
    dv  = tenor_map[inst] / T12_DAYS
    dv01_rows.append({
        'Instrument':     inst,
        'Tenor (d)':      int(tenor_map[inst]),
        'DV01 ratio':     round(dv, 3),
        'H* (rate)':      round(h, 3),
        'Notional hedge': round(h * dv, 3),
        'rho':            round(rho, 3),
        'R2 (%)':         round(r2, 1),
        'sigma (bps)':    round(sm, 3),
        'Resid (bps)':    round(sr, 3),
        'Risk redn (%)':  round((1 - sr / sm) * 100, 1),
    })
df_dv01 = pd.DataFrame(dv01_rows).set_index('Instrument')

# ── Rolling beta ──────────────────────────────────────────────────────────────
def rolling_beta(y_arr, x_arr, window, min_periods=30):
    """Vectorized rolling OLS beta = rolling_cov(y,x) / rolling_var(x)."""
    y = pd.Series(y_arr)
    x = pd.Series(x_arr)
    cov = y.rolling(window, min_periods=min_periods).cov(x)
    var = x.rolling(window, min_periods=min_periods).var()
    return (cov / var).values

dates_arr = delta.index.values
x_arr     = delta['T12'].values

# ── Pre-MPM vs Off-MPM ────────────────────────────────────────────────────────
pre_mask = delta['Days_to_MPM'] <= 5
off_mask = delta['Days_to_MPM'] >  5

regime_rows = []
for inst in boj_in_ols:
    for label, mask in [('Pre-MPM', pre_mask), ('Off-MPM', off_mask)]:
        r = ols_hedge(delta[mask][inst], delta[mask]['T12'])
        if r:
            regime_rows.append({'Instrument': inst, 'Period': label,
                                 'H*': r['H*'], 'rho': r['rho'], 'R2': r['R2 (%)']})
df_regime = pd.DataFrame(regime_rows)

# ════════════════════════════════════════════════════════════════════════════════
# PDF GENERATION
# ════════════════════════════════════════════════════════════════════════════════
os.makedirs('outputs', exist_ok=True)

def style_table(tbl, header_color='#4472C4'):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 1:
            cell.set_facecolor('#f5f5f5')
        else:
            cell.set_facecolor('white')

def add_title_page(pdf, title, subtitle=''):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    fig.patch.set_facecolor('#f0f4fa')
    ax.text(0.5, 0.65, title,    ha='center', va='center', fontsize=24,
            fontweight='bold', transform=ax.transAxes, color='#1a3a6b')
    ax.text(0.5, 0.50, subtitle, ha='center', va='center', fontsize=14,
            color='#555555', transform=ax.transAxes)
    date_str = (f"Data: {df_feat['Date'].min().date()} to {df_feat['Date'].max().date()}"
                f"   |   OOS from: {START_DATE}")
    ax.text(0.5, 0.35, date_str, ha='center', va='center', fontsize=10,
            color='#888888', transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)

print('Building PDF...')
with PdfPages(OUTPUT_PATH) as pdf:

    # ══════════════════════════════════════════════════════════════════════════
    # COVER
    # ══════════════════════════════════════════════════════════════════════════
    add_title_page(pdf,
        'BOJ OIS Swap Rate Prediction',
        'Model Evaluation & Hedge Ratio Report\n'
        '11 instruments: M1-M8 (forward) + T12/T18/T24 (spot)')

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: MODEL EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    add_title_page(pdf,
        'Part 1 — Model Evaluation',
        'LightGBM  |  Walk-forward OOS  |  3d & 5d horizons')

    # ── Page 1-1: Overall summary + IC by horizon ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Overall OOS Performance (start: {})'.format(START_DATE), fontsize=14)

    ax = axes[0]
    ax.axis('off')
    tdata = [
        ['Metric',               '3d Horizon',                          '5d Horizon'],
        ['IC (Spearman)',         f"{overall3['IC']:.3f}",               f"{overall5['IC']:.3f}"],
        ['RMSE (norm)',           f"{overall3['RMSE']:.4f}",             f"{overall5['RMSE']:.4f}"],
        ['Direction Accuracy',   f"{overall3['Direction_Accuracy']:.1%}",f"{overall5['Direction_Accuracy']:.1%}"],
        ['Dir Acc (large move)', f"{overall3['Direction_Accuracy_LargeMove']:.1%}",
                                  f"{overall5['Direction_Accuracy_LargeMove']:.1%}"],
        ['Instruments',          '11 (M1-M8, T12/T18/T24)',             ''],
        ['Observations (OOS)',    str(len(res3)),                        str(len(res5))],
        ['Walk-forward folds',    str(res3['Fold'].max() + 1),           str(res5['Fold'].max() + 1)],
    ]
    tbl = ax.table(cellText=tdata[1:], colLabels=tdata[0],
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    style_table(tbl)
    ax.set_title('Key Metrics', fontsize=11, pad=12)

    ax = axes[1]
    horizons = ['3d', '5d']
    ics      = [overall3['IC'], overall5['IC']]
    bars = ax.bar(horizons, ics, color=['steelblue', 'coral'],
                  edgecolor='black', linewidth=0.6, width=0.4)
    for bar, ic in zip(bars, ics):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{ic:.3f}', ha='center', fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_ylim(0, max(ics) * 1.4)
    ax.set_title('OOS IC by Horizon', fontsize=11)
    ax.set_ylabel('IC (Spearman)')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 1-2: IC by instrument ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('OOS IC by Instrument', fontsize=14)

    for ax, ic_df, title in [(axes[0], ic_lbl3, '3d Horizon'),
                              (axes[1], ic_lbl5, '5d Horizon')]:
        order = [l for l in LABEL_ORDER if l in ic_df['Rate_Label'].values]
        vals  = ic_df.set_index('Rate_Label').reindex(order)['IC']
        clrs  = ['steelblue' if not l.startswith('T') else 'coral' for l in order]
        bars  = ax.bar(order, vals.values, color=clrs, edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.005 if v >= 0 else -0.018),
                    f'{v:.3f}', ha='center', fontsize=8)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('IC')
        ax.legend(handles=[Patch(color='steelblue', label='BOJ Swap (M1-M8)'),
                            Patch(color='coral',     label='Tenor OIS (T12/T18/T24)')],
                  fontsize=9, loc='lower right')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 1-3: IC by Days_to_MPM + Fold stability ─────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('IC by Days-to-MPM  &  Walk-forward Fold Stability', fontsize=14)

    for ax, ic_df, title in [(axes[0, 0], ic_dtm3, '3d: IC by Days to MPM'),
                              (axes[0, 1], ic_dtm5, '5d: IC by Days to MPM')]:
        vals  = ic_df.set_index('bucket').reindex(DTM_LABELS)['IC']
        clrs  = ['coral' if b == '<=5d' else 'steelblue' for b in DTM_LABELS]
        bars  = ax.bar(DTM_LABELS, vals.values, color=clrs, edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals.values):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + (0.005 if v >= 0 else -0.02),
                        f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('IC')

    for ax, fold_df, oic, title in [(axes[1, 0], fold3, overall3['IC'], '3d: Fold IC'),
                                     (axes[1, 1], fold5, overall5['IC'], '5d: Fold IC')]:
        clrs = ['steelblue' if v >= 0 else 'salmon' for v in fold_df['IC']]
        ax.bar(fold_df['Fold'], fold_df['IC'], color=clrs, edgecolor='black', linewidth=0.5)
        ax.axhline(oic, color='red', linestyle='--', linewidth=1.5,
                   label=f'Overall IC = {oic:.3f}')
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Fold')
        ax.set_ylabel('IC')
        ax.legend(fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 1-4: Cross-sectional P&L ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-sectional P&L  (Top4 Long / Bottom4 Short, daily)', fontsize=14)

    for row, (pnl_df, lbl) in enumerate([(pnl3, '3d'), (pnl5, '5d')]):
        # Daily bars
        ax = axes[row, 0]
        clrs = ['steelblue' if v >= 0 else 'salmon' for v in pnl_df['PnL_bps']]
        ax.bar(pnl_df['Date'], pnl_df['PnL_bps'], color=clrs, width=1.5, linewidth=0)
        ax.axhline(0, color='black', linewidth=0.5)
        avg = pnl_df['PnL_bps'].mean()
        ax.axhline(avg, color='red', linestyle='--', linewidth=1.0,
                   label=f'Mean = {avg:.2f} bps')
        ax.set_title(f'{lbl} Daily P&L (bps)', fontsize=11)
        ax.set_ylabel('bps')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

        # Cumulative
        ax = axes[row, 1]
        ax.plot(pnl_df['Date'], pnl_df['Cum_bps'], color='steelblue', linewidth=1.5)
        ax.fill_between(pnl_df['Date'], pnl_df['Cum_bps'], 0,
                        where=pnl_df['Cum_bps'] >= 0, alpha=0.2, color='steelblue')
        ax.fill_between(pnl_df['Date'], pnl_df['Cum_bps'], 0,
                        where=pnl_df['Cum_bps'] < 0,  alpha=0.2, color='salmon')
        ax.axhline(0, color='black', linewidth=0.6)
        total = pnl_df['Cum_bps'].iloc[-1]
        ax.set_title(f'{lbl} Cumulative P&L = {total:.1f} bps', fontsize=11)
        ax.set_ylabel('Cumulative bps')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 1-5: Latest predictions ──────────────────────────────────────
    latest_date = df_feat['Date'].max().date()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Latest Predictions  (as of {latest_date})', fontsize=14)

    for ax, lat_df, title in [
            (axes[0], latest3, '3d Prediction (bps)'),
            (axes[1], latest5, '5d Prediction (bps)')]:
        if lat_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue
        order  = [l for l in LABEL_ORDER if l in lat_df['Rate_Label'].values]
        vals   = lat_df.set_index('Rate_Label').reindex(order)['Pred_bps'].dropna()
        clrs   = ['steelblue' if v >= 0 else 'salmon' for v in vals.values]
        ax.bar(vals.index, vals.values, color=clrs, edgecolor='black', linewidth=0.5)
        for i, (lbl, v) in enumerate(vals.items()):
            ax.text(i, v + (0.15 if v >= 0 else -0.35),
                    f'{v:.1f}', ha='center', fontsize=9)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Predicted change (bps)')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: HEDGE RATIO
    # ══════════════════════════════════════════════════════════════════════════
    add_title_page(pdf,
        'Part 2 — Hedge Ratio Analysis',
        f'OLS Minimum Variance Hedge  |  M1-M8 vs T12\n'
        f'avg meeting interval = {avg_interval:.1f}d  |  avg M1 tenor = {avg_m1_tenor:.1f}d')

    # ── Page 2-1: OLS results + hedge effectiveness ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('OLS Hedge Ratio vs T12 — Full Period', fontsize=14)
    x = np.arange(len(boj_in_ols))

    ax = axes[0]
    beta = df_ols.loc[boj_in_ols, 'H*'].astype(float)
    volr = df_ols.loc[boj_in_ols, 'Vol Ratio'].astype(float)
    ax.bar(x - 0.2, beta.values, BAR_WIDTH, label='OLS beta (H*)',
           color='steelblue', edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.2, volr.values, BAR_WIDTH, label='Vol ratio (ref)',
           color='lightgray',  edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(boj_in_ols)
    ax.set_title('Hedge Ratio H* vs Vol Ratio'); ax.set_ylabel('Ratio')
    ax.legend(fontsize=8)

    ax = axes[1]
    rho = df_ols.loc[boj_in_ols, 'rho'].astype(float)
    clrs = ['steelblue' if v >= 0 else 'salmon' for v in rho]
    ax.bar(boj_in_ols, rho.values, color=clrs, edgecolor='black', linewidth=0.5)
    for i, v in enumerate(rho.values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_title('Correlation (rho) with T12'); ax.set_ylabel('rho'); ax.set_ylim(-0.1, 1.05)

    ax = axes[2]
    sm  = df_ols.loc[boj_in_ols, 'sigma_Mn (bps)'].astype(float)
    sr  = df_ols.loc[boj_in_ols, 'Resid std (bps)'].astype(float)
    ax.bar(x - 0.2, sm.values,  BAR_WIDTH, label='Unhedged sigma',   color='salmon',    edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.2, sr.values,  BAR_WIDTH, label='Residual (hedged)', color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(boj_in_ols)
    ax.set_title('Risk Before vs After Hedge (bps/day)'); ax.set_ylabel('Std (bps/day)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 2-2: Scatter plots M_n vs T12 ────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    fig.suptitle('Daily Spread Change: M_n vs T12  (OLS regression line)', fontsize=14)

    for i, inst in enumerate(boj_in_ols):
        ax  = axes[i]
        dfp = pd.DataFrame({'x': delta['T12'] * 100, 'y': delta[inst] * 100}).dropna()
        h   = float(df_ols.loc[inst, 'H*'])
        r   = float(df_ols.loc[inst, 'rho'])
        r2  = float(df_ols.loc[inst, 'R2 (%)'])
        ax.scatter(dfp['x'], dfp['y'], alpha=0.25, s=8, color='steelblue')
        xln = np.linspace(dfp['x'].min(), dfp['x'].max(), 100)
        ax.plot(xln, h * xln, color='red', linewidth=1.5,
                label=f'H*={h:.3f}  rho={r:.3f}  R2={r2:.1f}%')
        ax.axhline(0, color='black', linewidth=0.4)
        ax.axvline(0, color='black', linewidth=0.4)
        ax.set_title(inst, fontweight='bold')
        ax.set_xlabel('Delta T12 (bps/day)')
        ax.set_ylabel(f'Delta {inst} (bps/day)')
        ax.legend(fontsize=7.5, loc='upper left')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 2-3: Rolling hedge ratio ─────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    fig.suptitle(f'Rolling Hedge Ratio H* ({ROLL_WINDOW}-day window) — M_n vs T12', fontsize=14)

    for i, inst in enumerate(boj_in_ols):
        ax     = axes[i]
        betas  = rolling_beta(delta[inst].values, x_arr, ROLL_WINDOW)
        fb     = float(df_ols.loc[inst, 'H*'])
        ax.plot(dates_arr, betas, color='steelblue', linewidth=1.0, label='Rolling H*')
        ax.axhline(fb, color='red', linestyle='--', linewidth=1.2,
                   label=f'Full-period H*={fb:.3f}')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(inst, fontweight='bold')
        ax.set_ylabel('H*')
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 2-4: Pre-MPM vs Off-MPM ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Hedge Ratio H* and Correlation — Pre-MPM (<=5d) vs Off-MPM (>5d)', fontsize=14)

    for ax, metric, ylabel in [(axes[0], 'H*', 'Hedge Ratio H*'),
                                (axes[1], 'rho', 'Correlation rho')]:
        pre_v = (df_regime[df_regime['Period'] == 'Pre-MPM']
                 .set_index('Instrument')[metric].reindex(boj_in_ols))
        off_v = (df_regime[df_regime['Period'] == 'Off-MPM']
                 .set_index('Instrument')[metric].reindex(boj_in_ols))
        xr = np.arange(len(boj_in_ols))
        ax.bar(xr - 0.2, pre_v.values.astype(float), BAR_WIDTH,
               label='Pre-MPM (<=5d)', color='coral',     edgecolor='black', linewidth=0.5)
        ax.bar(xr + 0.2, off_v.values.astype(float), BAR_WIDTH,
               label='Off-MPM  (>5d)', color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(xr); ax.set_xticklabels(boj_in_ols)
        ax.set_title(ylabel); ax.set_ylabel(ylabel); ax.legend(fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 2-5: DV01-adjusted bar chart + summary table ─────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    fig.suptitle('DV01-Adjusted Notional Hedge: M_n vs T12', fontsize=14)

    boj_dv = df_dv01.index.tolist()
    xd = np.arange(len(boj_dv))

    # H* vs Notional hedge
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(xd - 0.2, df_dv01['H* (rate)'].values,      BAR_WIDTH,
           label='H* (rate, same face)',       color='lightgray',  edgecolor='black', linewidth=0.5)
    ax.bar(xd + 0.2, df_dv01['Notional hedge'].values,  BAR_WIDTH,
           label='Notional hedge (DV01-adj.)', color='steelblue',  edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(xd); ax.set_xticklabels(boj_dv)
    ax.set_title('Rate H* vs DV01-Adjusted Notional'); ax.set_ylabel('Ratio')
    ax.legend(fontsize=8)

    # Correlation
    ax = fig.add_subplot(gs[0, 1])
    rho_d = df_dv01['rho'].values.astype(float)
    clrs_d = ['steelblue' if v >= 0 else 'salmon' for v in rho_d]
    ax.bar(boj_dv, rho_d, color=clrs_d, edgecolor='black', linewidth=0.5)
    for i, v in enumerate(rho_d):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    ax.set_title('Correlation rho with T12'); ax.set_ylabel('rho'); ax.set_ylim(-0.1, 1.05)

    # Risk reduction
    ax = fig.add_subplot(gs[0, 2])
    rr = df_dv01['Risk redn (%)'].values.astype(float)
    ax.bar(boj_dv, rr, color='steelblue', edgecolor='black', linewidth=0.5)
    for i, v in enumerate(rr):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    ax.set_title('Risk Reduction via T12 Hedge (%)'); ax.set_ylabel('%')

    # Summary table
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')
    cols  = list(df_dv01.reset_index().columns)
    rdata = df_dv01.reset_index().round(3).values.tolist()
    tbl   = ax.table(cellText=rdata, colLabels=cols,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    style_table(tbl)
    ax.set_title('DV01-Adjusted Notional Hedge Table\n'
                 'Notional hedge = H* x (tenor_Mn / 365d)   '
                 'Interpretation: for 1 unit of M_n, short [Notional hedge] units of T12',
                 fontsize=10, pad=15)

    pdf.savefig(fig)
    plt.close(fig)

print(f'Done. Report saved to: {OUTPUT_PATH}')

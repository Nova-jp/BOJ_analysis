#!/usr/bin/env python3
"""
BOJ OIS Trade Signal Generator
Usage (from project root): python trade_signal.py

出力:
    outputs/ts_fig1_signal.png     予測値サマリー + M1-M8 逆算
    outputs/ts_fig2_ic.png         IC 診断 (Fold >= FOLD_DIAG_START)
    outputs/ts_fig3_importance.png 特徴量重要度

M1-M8 逆算の仕組み:
    Butterfly (B2-B7): 6式
    Outright  (M4)   : 1式
    Curve     (M2-M5): 1式
    計 8×8 連立一次方程式 → np.linalg.solve で ΔM1..ΔM8 (bp) を解く
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # ヘッドレス実行（plt.show() 不要）
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr, linregress

from src.processing        import load_and_clean_data, load_meeting_dates
from src.features          import generate_features
from src.features_rv       import generate_rv_features
from src.pooling           import pool_boj_data
from src.pooling_curve     import pool_curve_data
from src.pooling_butterfly import pool_butterfly_data
from src.modeling          import walk_forward_with_model, summarize_ic, NON_FEATURE_COLS

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_PATH      = os.path.join(SCRIPT_DIR, 'data', 'BOJ_data.xlsx')
MEETING_PATH    = os.path.join(SCRIPT_DIR, 'data', 'BOJ_meeting_history.csv')
OUT_DIR         = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE      = '2024-01-01'
FOLD_DIAG_START = 5          # IC 診断で強調表示するフォールドの開始番号
TOP_N_FEATURES  = 15         # 特徴量重要度の表示件数
FLY_SHORT       = {2: 'B2', 3: 'B3', 4: 'B4', 5: 'B5', 6: 'B6', 7: 'B7'}

# ─────────────────────────────────────────────────────────────────────────────
# 1. データ読み込み
# ─────────────────────────────────────────────────────────────────────────────
print('Loading data...', flush=True)
df_raw        = load_and_clean_data(EXCEL_PATH, MEETING_PATH)
meeting_dates = load_meeting_dates(MEETING_PATH)

df_rv  = generate_rv_features(df_raw)
df_fly = pool_butterfly_data(df_rv)
df_crv = pool_curve_data(df_rv)
df_out = pool_boj_data(generate_features(df_raw), meeting_dates)

print(f'Data through : {df_raw["Date"].max().date()}')
print(f'Fly {df_fly.shape}  Crv {df_crv.shape}  Out {df_out.shape}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Walk-forward（最終フォールドのモデルも取得）
# ─────────────────────────────────────────────────────────────────────────────
print('Running walk-forward (6 models)...', flush=True)

res_fly_3d, mdl_fly_3d, *_ = walk_forward_with_model(df_fly, 'Target_3d_norm', START_DATE)
res_fly_5d, mdl_fly_5d, *_ = walk_forward_with_model(df_fly, 'Target_5d_norm', START_DATE)
res_crv_3d, mdl_crv_3d, *_ = walk_forward_with_model(df_crv, 'Target_3d_norm', START_DATE)
res_crv_5d, mdl_crv_5d, *_ = walk_forward_with_model(df_crv, 'Target_5d_norm', START_DATE)
res_out_3d, mdl_out_3d, *_ = walk_forward_with_model(df_out, 'Target_3d_norm', START_DATE)
res_out_5d, mdl_out_5d, *_ = walk_forward_with_model(df_out, 'Target_5d_norm', START_DATE)

# 逆正規化用 std を結合（Target_Xd_std × pred = 予測変化量 in bp×100）
_std_cols = ['Date', 'Meeting_Index', 'Target_3d_std', 'Target_5d_std']
std_fly = df_fly[_std_cols].drop_duplicates()
std_crv = df_crv[_std_cols].drop_duplicates()
std_out = df_out[_std_cols].drop_duplicates()

res_fly_3d = res_fly_3d.merge(std_fly, on=['Date', 'Meeting_Index'], how='left')
res_fly_5d = res_fly_5d.merge(std_fly, on=['Date', 'Meeting_Index'], how='left')
res_crv_3d = res_crv_3d.merge(std_crv, on=['Date', 'Meeting_Index'], how='left')
res_crv_5d = res_crv_5d.merge(std_crv, on=['Date', 'Meeting_Index'], how='left')
res_out_3d = res_out_3d.merge(std_out, on=['Date', 'Meeting_Index'], how='left')
res_out_5d = res_out_5d.merge(std_out, on=['Date', 'Meeting_Index'], how='left')

print(f'OOS: {res_fly_3d["Date"].min().date()} → {res_fly_3d["Date"].max().date()}  '
      f'({res_fly_3d["Fold"].max() + 1} folds)')


# ─────────────────────────────────────────────────────────────────────────────
# 3. シグナル基準日と各シグナル計算
# ─────────────────────────────────────────────────────────────────────────────

def _predict_for_signal(df, model, target_col, signal_date):
    """最終フォールドモデルを使って signal_date の特徴量から予測を生成する。

    ターゲット（shift -h）が NaN の最新日でも特徴量が揃っていれば予測可能。
    OOS 結果（res_*）は過去の精度評価用。このシグナル予測は「今日から将来を予測」用。
    """
    feature_cols = [c for c in df.columns
                    if c not in NON_FEATURE_COLS and c != target_col]
    rows = df[df['Date'] == signal_date].copy()
    if rows.empty:
        return pd.DataFrame()
    X     = rows[feature_cols]
    valid = X.notna().all(axis=1)
    rows  = rows[valid].copy()
    X     = X[valid]
    if rows.empty:
        return pd.DataFrame()
    rows['Pred']   = model.predict(X)
    rows['Actual'] = np.nan
    return rows.reset_index(drop=True)


# シグナル基準日: 全モデルで特徴量が揃っている最新日
# （OOS 結果の最新日ではなく、実際に予測したい日付）
SIGNAL_DATE = min(df_fly['Date'].max(), df_crv['Date'].max(), df_out['Date'].max())
print(f'Signal date  : {SIGNAL_DATE.date()}', flush=True)


def _fly_signal(r3, r5, date):
    """Butterfly B2-B7 の予測値 (bp) + 現在水準を返す (Score_3d 降順)"""
    d3 = r3[r3['Date'] == date][['Meeting_Index', 'Pred', 'Target_3d_std']].copy()
    d5 = r5[r5['Date'] == date][['Meeting_Index', 'Pred', 'Target_5d_std']].copy()
    d3['Pred_3d_bp'] = d3['Pred'] * d3['Target_3d_std'] * 100
    d5['Pred_5d_bp'] = d5['Pred'] * d5['Target_5d_std'] * 100
    sig = (d3[['Meeting_Index', 'Pred', 'Pred_3d_bp']]
           .merge(d5[['Meeting_Index', 'Pred_5d_bp']], on='Meeting_Index')
           .rename(columns={'Pred': 'Score_3d'}))
    lvl = (df_fly[df_fly['Date'] == date][['Meeting_Index', 'Fly_Level']]
           .drop_duplicates()
           .assign(Fly_Level_bp=lambda x: x['Fly_Level'] * 100))
    sig = sig.merge(lvl[['Meeting_Index', 'Fly_Level_bp']], on='Meeting_Index', how='left')
    sig['Label'] = sig['Meeting_Index'].map(FLY_SHORT)
    return sig.sort_values('Score_3d', ascending=False).reset_index(drop=True)


def _m2m5_signal(r3, r5, date):
    """C2+C3+C4 の逆正規化合算 → M2-M5 スティープナー/フラットナー方向"""
    out = {}
    for h, res in [(3, r3), (5, r5)]:
        sc = f'Target_{h}d_std'
        ps, acs = 0.0, 0.0
        for n in [2, 3, 4]:
            row = res[(res['Date'] == date) & (res['Meeting_Index'] == n)]
            if row.empty:
                return None
            sv   = float(row[sc].values[0])
            ps  += float(row['Pred'].values[0])   * sv * 100
            acs += float(row['Actual'].values[0]) * sv * 100
        out[f'Pred_{h}d_bp']   = ps
        out[f'Actual_{h}d_bp'] = acs
    lvl = 0.0
    for n in [2, 3, 4]:
        row_crv = df_crv[(df_crv['Date'] == date) & (df_crv['Meeting_Index'] == n)]
        if not row_crv.empty:
            lvl += float(row_crv['Curve_Level'].values[0]) * 100
    out['M2M5_level_bp'] = lvl
    return out


def _m4_signal(r3, r5, date):
    """M4 アウトライト予測値 (bp)"""
    d3 = r3[(r3['Date'] == date) & (r3['Meeting_Index'] == 4)]
    d5 = r5[(r5['Date'] == date) & (r5['Meeting_Index'] == 4)]
    if d3.empty or d5.empty:
        return None
    m4_row    = df_out[(df_out['Date'] == date) & (df_out['Meeting_Index'] == 4)]
    spread_bp = float(m4_row['M4_spread'].values[0]) * 100 if not m4_row.empty else np.nan
    policy_bp = float(df_raw[df_raw['Date'] == date]['Actual_Policy_Rate'].values[0]) * 100
    return {
        'M4_abs_bp':    spread_bp + policy_bp,
        'M4_spread_bp': spread_bp,
        'Pred_3d_bp':   float(d3['Pred'].values[0]) * float(d3['Target_3d_std'].values[0]) * 100,
        'Pred_5d_bp':   float(d5['Pred'].values[0]) * float(d5['Target_5d_std'].values[0]) * 100,
    }


# 最終フォールドモデルを SIGNAL_DATE の特徴量に直接適用してシグナルを生成
sig_fly_3d = _predict_for_signal(df_fly, mdl_fly_3d, 'Target_3d_norm', SIGNAL_DATE)
sig_fly_5d = _predict_for_signal(df_fly, mdl_fly_5d, 'Target_5d_norm', SIGNAL_DATE)
sig_crv_3d = _predict_for_signal(df_crv, mdl_crv_3d, 'Target_3d_norm', SIGNAL_DATE)
sig_crv_5d = _predict_for_signal(df_crv, mdl_crv_5d, 'Target_5d_norm', SIGNAL_DATE)
sig_out_3d = _predict_for_signal(df_out, mdl_out_3d, 'Target_3d_norm', SIGNAL_DATE)
sig_out_5d = _predict_for_signal(df_out, mdl_out_5d, 'Target_5d_norm', SIGNAL_DATE)

fly_sig  = _fly_signal(sig_fly_3d, sig_fly_5d, SIGNAL_DATE)
m2m5_sig = _m2m5_signal(sig_crv_3d, sig_crv_5d, SIGNAL_DATE)
m4_sig   = _m4_signal(sig_out_3d, sig_out_5d, SIGNAL_DATE)


# ─────────────────────────────────────────────────────────────────────────────
# 3b. M1-M8 vs T12 (1y): correlation & slope over last 50 days
# ─────────────────────────────────────────────────────────────────────────────
_CORR_WINDOW = 50
_raw_w = df_raw.sort_values('Date').tail(_CORR_WINDOW + 1)
_t12_chg = _raw_w['T12'].diff().dropna() if 'T12' in _raw_w.columns else None

_corr_t12  = {}
_slope_t12 = {}
for n in range(1, 9):
    col = f'M{n}'
    if _t12_chg is None or col not in _raw_w.columns:
        _corr_t12[col] = np.nan
        _slope_t12[col] = np.nan
        continue
    _mn_chg = _raw_w[col].diff().dropna()
    _idx    = _t12_chg.index.intersection(_mn_chg.index)
    t12v    = _t12_chg.loc[_idx].values
    mnv     = _mn_chg.loc[_idx].values
    mask    = ~(np.isnan(t12v) | np.isnan(mnv))
    if mask.sum() >= 5:
        slope, _, r_val, _, _ = linregress(t12v[mask], mnv[mask])
        _corr_t12[col]  = r_val
        _slope_t12[col] = slope
    else:
        _corr_t12[col]  = np.nan
        _slope_t12[col] = np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 4. M1-M8 絶対水準 逆算（参考値）
#
#   未知数: x = [ΔM1, ΔM2, ..., ΔM8]  (bp、Policy_Rate は変化しない前提)
#
#   Row 0-5  Butterfly B{n} = 2x[n] - x[n-1] - x[n+1]  (n=2..7)
#   Row 6    Outright  M4:   x[3] = ΔM4_pred
#   Row 7    Curve M2-M5:    x[1] - x[4] = Δ(M2-M5)_pred
#
#   np.linalg.solve(A, b) でフルランク 8×8 系を解く
# ─────────────────────────────────────────────────────────────────────────────
def _reconstruct_delta(fly_sig, m2m5_sig, m4_sig, h):
    """8×8 連立方程式を解いて ΔM1..ΔM8 (bp) を返す"""
    A = np.zeros((8, 8))
    b = np.zeros(8)
    pc = f'Pred_{h}_bp'

    for i, n in enumerate(range(2, 8)):        # 行 0-5: Butterfly
        A[i, n - 2] = -1                       #   M{n-1}
        A[i, n - 1] =  2                       #   M{n}
        A[i, n    ] = -1                       #   M{n+1}
        row = fly_sig[fly_sig['Meeting_Index'] == n]
        b[i] = float(row[pc].values[0]) if not row.empty else 0.0

    A[6, 3] = 1                                # 行 6: M4 outright
    b[6] = m4_sig[pc] if m4_sig else 0.0

    A[7, 1] =  1                               # 行 7: M2-M5 curve
    A[7, 4] = -1
    b[7] = m2m5_sig[pc] if m2m5_sig else 0.0

    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.full(8, np.nan)


raw_row   = df_raw[df_raw['Date'] == SIGNAL_DATE].iloc[0]
curr_bp   = np.array([raw_row[f'M{n}'] * 100 for n in range(1, 9)])
policy_bp = raw_row['Actual_Policy_Rate'] * 100
delta_3d  = _reconstruct_delta(fly_sig, m2m5_sig, m4_sig, '3d')
delta_5d  = _reconstruct_delta(fly_sig, m2m5_sig, m4_sig, '5d')

recon_df = pd.DataFrame({
    'Series':   [f'M{n}' for n in range(1, 9)],
    'Current':  curr_bp,
    'Delta_3d': delta_3d,
    'Pred_3d':  curr_bp + delta_3d,
    'Delta_5d': delta_5d,
    'Pred_5d':  curr_bp + delta_5d,
})


# ─────────────────────────────────────────────────────────────────────────────
# 5. コンソール出力
# ─────────────────────────────────────────────────────────────────────────────
SEP = '=' * 68
print(f'\n{SEP}')
print(f'  BOJ OIS Trade Signal   {SIGNAL_DATE.date()}')
print(SEP)

print('\n[ Butterfly B2-B7  (Score_3d 降順) ]')
print(f"  {'Rk':>2}  {'Sym':<4}  {'Level(bp)':>9}  {'Pred_3d(bp)':>11}  {'Pred_5d(bp)':>11}  Dir")
print('  ' + '-' * 55)
for rk, row in enumerate(fly_sig.itertuples(), 1):
    lvl = f'{row.Fly_Level_bp:+.3f}' if not pd.isna(row.Fly_Level_bp) else '  N/A '
    d   = 'UP  ' if row.Pred_3d_bp > 0 else 'DOWN'
    print(f'  {rk:>2}  {row.Label:<4}  {lvl:>9}  {row.Pred_3d_bp:>+10.3f}  '
          f'{row.Pred_5d_bp:>+10.3f}  {d}')

print('\n[ M2-M5 Curve Signal ]')
if m2m5_sig:
    d3 = 'Steeper' if m2m5_sig['Pred_3d_bp'] > 0 else 'Flatter'
    d5 = 'Steeper' if m2m5_sig['Pred_5d_bp'] > 0 else 'Flatter'
    print(f"  Level: {m2m5_sig['M2M5_level_bp']:+.2f} bp  |  "
          f"Pred 3d: {m2m5_sig['Pred_3d_bp']:+.3f} bp  {d3}  |  "
          f"Pred 5d: {m2m5_sig['Pred_5d_bp']:+.3f} bp  {d5}")

print('\n[ M4 Outright Signal ]')
if m4_sig:
    d3 = 'UP  ' if m4_sig['Pred_3d_bp'] > 0 else 'DOWN'
    d5 = 'UP  ' if m4_sig['Pred_5d_bp'] > 0 else 'DOWN'
    print(f"  M4={m4_sig['M4_abs_bp']:.1f}bp  spread={m4_sig['M4_spread_bp']:+.1f}bp  |  "
          f"Pred 3d: {m4_sig['Pred_3d_bp']:+.3f}bp {d3}  |  "
          f"Pred 5d: {m4_sig['Pred_5d_bp']:+.3f}bp {d5}")

print('\n[ M1-M8 Reconstruction (参考値: 3モデル連立方程式から逆算) ]')
print(f"  {'':>5}  {'Current':>9}  {'Pred 3d':>9}  {'delta_3d':>9}  "
      f"{'Pred 5d':>9}  {'delta_5d':>9}")
print('  ' + '-' * 62)
for _, r in recon_df.iterrows():
    print(f"  {r['Series']:>5}  {r['Current']:>9.2f}  {r['Pred_3d']:>9.2f}  "
          f"{r['Delta_3d']:>+9.3f}  {r['Pred_5d']:>9.2f}  {r['Delta_5d']:>+9.3f}")
print(f'\n{SEP}')


# ─────────────────────────────────────────────────────────────────────────────
# 6. IC サマリー計算
# ─────────────────────────────────────────────────────────────────────────────
ic_fly_3d = summarize_ic(res_fly_3d, instrument_indices=set(range(2, 8)))
ic_fly_5d = summarize_ic(res_fly_5d, instrument_indices=set(range(2, 8)))
ic_crv_3d = summarize_ic(res_crv_3d)
ic_crv_5d = summarize_ic(res_crv_5d)
ic_out_3d = summarize_ic(res_out_3d)
ic_out_5d = summarize_ic(res_out_5d)


# ─────────────────────────────────────────────────────────────────────────────
# 7. 特徴量重要度
# ─────────────────────────────────────────────────────────────────────────────
def _feat_imp(model):
    df_i = pd.DataFrame({'feature': model.feature_name(),
                         'gain':    model.feature_importance(importance_type='gain')})
    return df_i.sort_values('gain', ascending=False).head(TOP_N_FEATURES).reset_index(drop=True)


imp_fly_3d = _feat_imp(mdl_fly_3d)
imp_fly_5d = _feat_imp(mdl_fly_5d)
imp_crv_3d = _feat_imp(mdl_crv_3d)
imp_crv_5d = _feat_imp(mdl_crv_5d)
imp_out_3d = _feat_imp(mdl_out_3d)
imp_out_5d = _feat_imp(mdl_out_5d)


# ─────────────────────────────────────────────────────────────────────────────
# PDF 出力開始
# ─────────────────────────────────────────────────────────────────────────────
PDF_PATH = os.path.join(OUT_DIR, f'trade_signal_{SIGNAL_DATE.strftime("%Y%m%d")}.pdf')
_pdf = PdfPages(PDF_PATH)
print(f'\nGenerating PDF: {os.path.basename(PDF_PATH)}')

# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: 予測値サマリー + M1-M8 逆算
# ═════════════════════════════════════════════════════════════════════════════
fly_plot = fly_sig.sort_values('Meeting_Index').reset_index(drop=True)
fly_lbls = fly_plot['Label'].tolist()
fly_lvls = fly_plot['Fly_Level_bp'].tolist()
fly_p3   = fly_plot['Pred_3d_bp'].tolist()
fly_p5   = fly_plot['Pred_5d_bp'].tolist()
rate_lbls = [f'M{n}' for n in range(1, 9)]
rate_vals = [raw_row[c] * 100 for c in rate_lbls]

fig = plt.figure(figsize=(18, 24))
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)
fig.suptitle(f'BOJ OIS Trade Signal   |   {SIGNAL_DATE.date()}', fontsize=14, fontweight='bold')

# (0,0) Butterfly 現在水準
ax = fig.add_subplot(gs[0, 0])
ax.bar(fly_lbls, fly_lvls,
       color=['#d62728' if v < 0 else '#ff7f0e' for v in fly_lvls], edgecolor='white')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('bp'); ax.set_title('Butterfly Levels (bp)'); ax.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(fly_lvls):
    if not np.isnan(v):
        ax.text(i, v + (0.02 if v >= 0 else -0.04), f'{v:+.2f}',
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=8.5)

# (0,1) Butterfly 予測変化量
ax = fig.add_subplot(gs[0, 1])
x, w = np.arange(len(fly_lbls)), 0.35
ax.bar(x - w / 2, fly_p3, w, label='3d',
       color=['#2ca02c' if v >= 0 else '#d62728' for v in fly_p3], alpha=0.85, edgecolor='white')
ax.bar(x + w / 2, fly_p5, w, label='5d',
       color=['#1f77b4' if v >= 0 else '#e377c2' for v in fly_p5], alpha=0.85, edgecolor='white')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(fly_lbls)
ax.set_ylabel('Delta (bp)'); ax.set_title('Butterfly Predicted Changes (bp)')
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(fly_p3):
    ax.text(i - w / 2, v + (0.01 if v >= 0 else -0.02), f'{v:+.2f}',
            ha='center', va='bottom' if v >= 0 else 'top', fontsize=7.5)

# (1,0) OIS カーブ（絶対水準）
ax = fig.add_subplot(gs[1, 0])
ax.plot(rate_lbls, rate_vals, marker='o', color='#1f77b4', lw=2)
ax.axhline(policy_bp, color='gray', ls='--', alpha=0.6, label=f'Policy {policy_bp:.0f} bp')
ax.scatter(['M4'], [rate_vals[3]], color='red', s=80, zorder=5, label='M4 (Outright)')
ax.scatter(['M2', 'M5'], [rate_vals[1], rate_vals[4]], color='#2ca02c',
           s=60, zorder=5, label='M2, M5 (Curve)')
ax.set_ylabel('bp'); ax.set_title('OIS Curve (bp)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
for i, v in enumerate(rate_vals):
    ax.text(i, v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=7.5)

# (1,1) Curve & Outright シグナルテキスト
ax = fig.add_subplot(gs[1, 1]); ax.axis('off')
lines = []
if m2m5_sig:
    d3 = 'Steeper ^' if m2m5_sig['Pred_3d_bp'] > 0 else 'Flatter v'
    d5 = 'Steeper ^' if m2m5_sig['Pred_5d_bp'] > 0 else 'Flatter v'
    lines += [
        'M2-M5 Curve Signal',
        f"  Level : {m2m5_sig['M2M5_level_bp']:+.2f} bp",
        f"  Pred 3d: {m2m5_sig['Pred_3d_bp']:+.3f} bp  [{d3}]",
        f"  Pred 5d: {m2m5_sig['Pred_5d_bp']:+.3f} bp  [{d5}]",
        '',
    ]
if m4_sig:
    d3 = 'UP ^' if m4_sig['Pred_3d_bp'] > 0 else 'DOWN v'
    d5 = 'UP ^' if m4_sig['Pred_5d_bp'] > 0 else 'DOWN v'
    lines += [
        'M4 Outright Signal',
        f"  Level  : {m4_sig['M4_abs_bp']:.1f} bp",
        f"  Spread : {m4_sig['M4_spread_bp']:+.1f} bp (vs policy)",
        f"  Pred 3d: {m4_sig['Pred_3d_bp']:+.3f} bp  [{d3}]",
        f"  Pred 5d: {m4_sig['Pred_5d_bp']:+.3f} bp  [{d5}]",
    ]
ax.text(0.05, 0.95, '\n'.join(lines), ha='left', va='top', fontsize=11,
        transform=ax.transAxes, family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f4f8', alpha=0.9))

# (2,:) M1-M8 逆算テーブル
ax = fig.add_subplot(gs[2, :])
ax.axis('off')
ax.set_title(
    'M1-M8 Reconstruction  --  '
    'B2-B7 x6 + M4(Outright) + M2-M5(Curve) = 8x8 Linear System',
    fontsize=10, fontweight='bold', pad=8)

col_labels = ['Series', 'Current (bp)', 'Pred 3d (bp)', 'Delta 3d (bp)', 'Pred 5d (bp)', 'Delta 5d (bp)']
cell_data  = []
for _, r in recon_df.iterrows():
    cell_data.append([
        r['Series'],
        f"{r['Current']:.2f}",
        f"{r['Pred_3d']:.2f}",
        f"{r['Delta_3d']:+.3f}",
        f"{r['Pred_5d']:.2f}",
        f"{r['Delta_5d']:+.3f}",
    ])

tbl = ax.table(cellText=cell_data, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 1.8)

for col_i, h_col in [(3, 'Delta_3d'), (5, 'Delta_5d')]:
    for row_i, (_, r) in enumerate(recon_df.iterrows()):
        cell = tbl[row_i + 1, col_i]
        val  = r[h_col]
        cell.set_facecolor('#c8e6c9' if val > 0.005 else '#ffcdd2' if val < -0.005 else '#fffde7')

# (3,:) Delta line chart
ax = fig.add_subplot(gs[3, :])
x_pos  = np.arange(8)
x_lbls = [f'M{n}' for n in range(1, 9)]
ax.plot(x_pos, delta_3d, marker='o', color='#2ca02c', lw=2, label='Delta 3d (bp)')
ax.plot(x_pos, delta_5d, marker='s', color='#1f77b4', lw=2, label='Delta 5d (bp)')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_lbls)
ax.set_ylabel('Delta (bp)')
ax.set_title('Predicted Delta M1-M8 (bp)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
for i, (v3, v5) in enumerate(zip(delta_3d, delta_5d)):
    if not np.isnan(v3):
        ax.text(i - 0.12, v3 + (0.05 if v3 >= 0 else -0.08), f'{v3:+.2f}',
                ha='center', va='bottom' if v3 >= 0 else 'top', fontsize=8, color='#2ca02c')
    if not np.isnan(v5):
        ax.text(i + 0.12, v5 + (0.05 if v5 >= 0 else -0.08), f'{v5:+.2f}',
                ha='center', va='bottom' if v5 >= 0 else 'top', fontsize=8, color='#1f77b4')

# (4,0) M1-M8 vs T12 Correlation (last 50d)
ax = fig.add_subplot(gs[4, 0])
_m_lbls   = [f'M{n}' for n in range(1, 9)]
corr_vals = [_corr_t12.get(l, np.nan) for l in _m_lbls]
ax.bar(_m_lbls, corr_vals,
       color=['#2ca02c' if (not np.isnan(v) and v >= 0) else '#d62728' for v in corr_vals],
       edgecolor='white')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylim(-1.1, 1.1)
ax.set_ylabel('Pearson r')
ax.set_title(f'M1-M8 vs T12 (1y) Correlation  (last {_CORR_WINDOW}d)')
ax.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(corr_vals):
    if not np.isnan(v):
        ax.text(i, v + (0.03 if v >= 0 else -0.05), f'{v:.2f}',
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)

# (4,1) M1-M8 vs T12 Regression Slope (last 50d)
ax = fig.add_subplot(gs[4, 1])
slope_vals = [_slope_t12.get(l, np.nan) for l in _m_lbls]
ax.bar(_m_lbls, slope_vals,
       color=['#1f77b4' if (not np.isnan(v) and v >= 0) else '#e377c2' for v in slope_vals],
       edgecolor='white')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('Slope  dM / dT12')
ax.set_title(f'M1-M8 vs T12 (1y) Regression Slope  (last {_CORR_WINDOW}d)')
ax.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(slope_vals):
    if not np.isnan(v):
        ax.text(i, v + (0.01 if v >= 0 else -0.02), f'{v:.2f}',
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)

_pdf.savefig(fig, bbox_inches='tight')
plt.close()
print('  Page 1: Signal summary')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: IC 診断 (Fold >= FOLD_DIAG_START)
#
#   Row 0: Butterfly 3d  Actual vs Pred scatter (B2-B7 各銘柄, fold5+ のみ)
#   Row 1: Butterfly 5d  Actual vs Pred scatter (B2-B7 各銘柄, fold5+ のみ)
#   Row 2: Butterfly Global IC by fold  (3d / 5d)
#   Row 3: Curve     Global IC by fold  (3d / 5d)
#   Row 4: Outright  Global IC by fold  (3d / 5d)
# ═════════════════════════════════════════════════════════════════════════════
def _scatter_per_inst(ax, results, mi, folds_from, title, color='steelblue'):
    """指定 Meeting_Index × fold >= folds_from の Actual vs Pred 散布図"""
    sub = results[(results['Fold'] >= folds_from) & (results['Meeting_Index'] == mi)]
    if sub.empty:
        ax.set_title(title, fontsize=8.5)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    ic, _ = spearmanr(sub['Actual'], sub['Pred'])
    ax.scatter(sub['Actual'], sub['Pred'], s=8, alpha=0.45, color=color)
    lo = min(sub['Actual'].min(), sub['Pred'].min())
    hi = max(sub['Actual'].max(), sub['Pred'].max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=0.8, alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.35)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.35)
    ax.set_title(f'{title}\nIC={ic:.3f}', fontsize=8.5, fontweight='bold')
    ax.set_xlabel('Actual (norm)', fontsize=7)
    ax.set_ylabel('Pred (norm)', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)


def _ic_fold_bar(ax, ic_dict, label, color_hi, folds_from):
    """fold 別 Global IC 棒グラフ。fold >= folds_from を color_hi で強調"""
    folds = sorted(ic_dict['ic_by_fold'].keys())
    vals  = [ic_dict['ic_by_fold'][f] for f in folds]
    colors = [color_hi if f >= folds_from else '#cccccc' for f in folds]
    bars = ax.bar([str(f) for f in folds], vals, color=colors, edgecolor='white')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('Fold', fontsize=8)
    ax.set_ylabel('Global IC', fontsize=8)
    title_lines = (
        f'{label}  Global IC by Fold\n'
        f'All={ic_dict["ic_all"]:.3f}  '
        f'Train={ic_dict["train_ic"]:.3f}  '
        f'Gap={ic_dict["gap"]:.3f}'
    )
    if 'cs_ic' in ic_dict and not np.isnan(ic_dict['cs_ic']):
        title_lines += f'  CS={ic_dict["cs_ic"]:.3f}'
    ax.set_title(title_lines, fontsize=8.5, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.005 if v >= 0 else -0.018), f'{v:.3f}',
                ha='center', fontsize=7)


fig = plt.figure(figsize=(20, 24))
gs  = gridspec.GridSpec(5, 6, figure=fig, hspace=0.60, wspace=0.42)
fig.suptitle(
    f'IC Diagnostics (Fold >= {FOLD_DIAG_START} highlighted)   |   {SIGNAL_DATE.date()}',
    fontsize=13, fontweight='bold')

# Row 0: Butterfly 3d scatter per instrument
for i, n in enumerate(range(2, 8)):
    ax = fig.add_subplot(gs[0, i])
    _scatter_per_inst(ax, res_fly_3d, n, FOLD_DIAG_START, f'Fly {FLY_SHORT[n]} 3d', color='#ff7f0e')

# Row 1: Butterfly 5d scatter per instrument
for i, n in enumerate(range(2, 8)):
    ax = fig.add_subplot(gs[1, i])
    _scatter_per_inst(ax, res_fly_5d, n, FOLD_DIAG_START, f'Fly {FLY_SHORT[n]} 5d', color='#d62728')

# Row 2: Butterfly Global IC by fold
_ic_fold_bar(fig.add_subplot(gs[2, :3]), ic_fly_3d, 'Butterfly 3d', '#ff7f0e', FOLD_DIAG_START)
_ic_fold_bar(fig.add_subplot(gs[2, 3:]), ic_fly_5d, 'Butterfly 5d', '#d62728', FOLD_DIAG_START)

# Row 3: Curve Global IC by fold
_ic_fold_bar(fig.add_subplot(gs[3, :3]), ic_crv_3d, 'Curve 3d',    '#2ca02c', FOLD_DIAG_START)
_ic_fold_bar(fig.add_subplot(gs[3, 3:]), ic_crv_5d, 'Curve 5d',    '#98df8a', FOLD_DIAG_START)

# Row 4: Outright Global IC by fold
_ic_fold_bar(fig.add_subplot(gs[4, :3]), ic_out_3d, 'Outright 3d', '#1f77b4', FOLD_DIAG_START)
_ic_fold_bar(fig.add_subplot(gs[4, 3:]), ic_out_5d, 'Outright 5d', '#aec7e8', FOLD_DIAG_START)

_pdf.savefig(fig, bbox_inches='tight')
plt.close()
print('  Page 2: IC diagnostics')


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3: 特徴量重要度 (gain, 最終フォールドのモデル)
# ═════════════════════════════════════════════════════════════════════════════
imp_pairs = [
    (imp_fly_3d, 'Butterfly 3d', '#ff7f0e'),
    (imp_fly_5d, 'Butterfly 5d', '#d62728'),
    (imp_crv_3d, 'Curve 3d',     '#2ca02c'),
    (imp_crv_5d, 'Curve 5d',     '#98df8a'),
    (imp_out_3d, 'Outright 3d',  '#1f77b4'),
    (imp_out_5d, 'Outright 5d',  '#aec7e8'),
]

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle(
    f'Feature Importance (gain, last fold, top {TOP_N_FEATURES})   |   {SIGNAL_DATE.date()}',
    fontsize=13, fontweight='bold')

for ax, (df_imp, title, color) in zip(axes.flat, imp_pairs):
    ax.barh(df_imp['feature'][::-1], df_imp['gain'][::-1], color=color, edgecolor='white')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Gain', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
_pdf.savefig(fig, bbox_inches='tight')
plt.close()
print('  Page 3: Feature importance')

_pdf.close()
print(f'\nSaved: {PDF_PATH}')
print('All done.')

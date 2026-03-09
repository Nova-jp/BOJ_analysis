"""
BOJ OIS Swap Analysis Report
自己完結型。data/BOJ_data.xlsx と data/BOJ_meeting_history.csv のみ使用。

実行:
    python boj_report.py

出力:
    boj_analysis_report.pdf（スクリプトと同じディレクトリ）

依存パッケージ:
    pandas, numpy, matplotlib, scipy, scikit-learn, lightgbm, openpyxl
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import lightgbm as lgb

# ------------------------------------------------------------------ #
#  パス設定
# ------------------------------------------------------------------ #
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH       = os.path.join(BASE_DIR, 'data', 'BOJ_data.xlsx')
MEETING_CSV_PATH = os.path.join(BASE_DIR, 'data', 'BOJ_meeting_history.csv')
OUTPUT_PATH      = os.path.join(BASE_DIR, 'boj_analysis_report.pdf')
OOS_START_DATE   = '2024-01-01'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family']  = 'DejaVu Sans'
plt.rcParams['font.size']    = 9


# ================================================================== #
#  1. データ処理
# ================================================================== #
def load_and_clean_data(excel_path, meeting_csv_path):
    df_raw = pd.read_excel(excel_path)
    df = df_raw.iloc[1:].copy()
    df['日付'] = pd.to_datetime(df['日付'], format='%Y年%m月%d日')
    df = df.sort_values('日付').reset_index(drop=True)

    rename = {
        'JPBOJ1ONI=TRDT (MID_PRICE)': 'M1', 'JPBOJ2ONI=TRDT (MID_PRICE)': 'M2',
        'JPBOJ3ONI=TRDT (MID_PRICE)': 'M3', 'JPBOJ4ONI=TRDT (MID_PRICE)': 'M4',
        'JPBOJ5ONI=TRDT (MID_PRICE)': 'M5', 'JPBOJ6ONI=TRDT (MID_PRICE)': 'M6',
        'JPBOJ7ONI=TRDT (MID_PRICE)': 'M7', 'JPBOJ8ONI=TRDT (MID_PRICE)': 'M8',
        'JPY= (MID_PRICE)': 'USDJPY', 'JGBc1 (TRDPRC_1)': 'JGB_Future',
        '.N225 (TRDPRC_1)': 'Nikkei225', '.DXY (TRDPRC_1)': 'DXY',
    }
    df = df.rename(columns=rename)
    keep = ['日付', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8',
            'USDJPY', 'JGB_Future', 'Nikkei225', 'DXY']
    df = df[[c for c in keep if c in df.columns]].copy()
    for col in df.columns.drop('日付'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_mtg = pd.read_csv(meeting_csv_path)
    df_mtg['Date'] = pd.to_datetime(df_mtg['Date'])
    df = pd.merge(df, df_mtg[['Date', 'Policy_Rate', 'Event']],
                  left_on='日付', right_on='Date', how='left')
    df = df.drop(columns=['Date']).rename(columns={'日付': 'Date'})
    df['Is_Meeting_Day']    = df['Event'].notnull().astype(int)
    df['Actual_Policy_Rate'] = df['Policy_Rate'].ffill().bfill()

    boj_cols = [f'M{i}' for i in range(1, 9)]
    for col in boj_cols:
        df[f'{col}_is_imputed'] = df[col].isnull().astype(int)

    impute_cols = boj_cols + ['USDJPY', 'JGB_Future', 'Nikkei225', 'DXY']
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)
    df[impute_cols] = imp.fit_transform(df[impute_cols])

    all_mtg = sorted(df_mtg['Date'].unique())
    def _dtm(date):
        future = [m for m in all_mtg if m >= date]
        return (future[0] - date).days if future else np.nan
    df['Days_to_MPM'] = df['Date'].map({d: _dtm(d) for d in df['Date'].unique()})

    final = (['Date', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8',
              'USDJPY', 'JGB_Future', 'Nikkei225', 'DXY',
              'Actual_Policy_Rate', 'Is_Meeting_Day', 'Days_to_MPM']
             + [f'{c}_is_imputed' for c in boj_cols])
    return df[[c for c in final if c in df.columns]]


# ================================================================== #
#  2. 特徴量生成
# ================================================================== #
def _frac_diff(series, d=0.4, window=50):
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    w = np.array(weights[::-1])

    def _apply(x):
        return np.dot(x, w) if (len(x) >= window and not np.isnan(x).any()) else np.nan

    return series.rolling(window=window).apply(_apply, raw=True)


def generate_features(df, d=0.4, window=50):
    f = df.copy()
    for i in range(1, 9):
        f[f'M{i}_spread']    = f[f'M{i}'] - f['Actual_Policy_Rate']
        f[f'M{i}_frac_diff'] = _frac_diff(f[f'M{i}_spread'], d=d, window=window)
    for col in ['USDJPY', 'JGB_Future', 'Nikkei225']:
        f[f'{col}_frac_diff'] = _frac_diff(f[col], d=d, window=window)
    return f


# ================================================================== #
#  3. プーリング
# ================================================================== #
def pool_boj_data(df):
    boj_rate = [f'M{i}' for i in range(1, 9)]
    id_cols  = [c for c in df.columns
                if not c.startswith('M') or '_is_imputed' in c
                or '_spread' in c or '_frac_diff' in c]
    pooled = df.melt(id_vars=id_cols, value_vars=boj_rate,
                     var_name='M_Label', value_name='M_Value')
    pooled['Meeting_Index'] = pooled['M_Label'].str.extract(r'(\d+)').astype(int)
    pooled = pooled.sort_values(['Meeting_Index', 'Date']).reset_index(drop=True)

    for h in [1, 3, 5]:
        pooled[f'Target_{h}d'] = (pooled.groupby('Meeting_Index')['M_Value'].shift(-h)
                                  - pooled['M_Value'])

    dm = df[['Date', 'Is_Meeting_Day']].copy()
    dm['is_post_mpm'] = (dm['Is_Meeting_Day'].rolling(6, min_periods=1).max() == 1).astype(int)
    pooled = pooled.merge(dm[['Date', 'is_post_mpm']], on='Date', how='left')

    cols = (['Date', 'Meeting_Index', 'is_post_mpm', 'Days_to_MPM', 'Actual_Policy_Rate']
            + [f'M{i}_spread'    for i in range(1, 9)]
            + [f'M{i}_frac_diff' for i in range(1, 9)]
            + [f'M{i}_is_imputed' for i in range(1, 9)]
            + ['USDJPY_frac_diff', 'JGB_Future_frac_diff', 'Nikkei225_frac_diff']
            + ['Target_1d', 'Target_3d', 'Target_5d'])
    return pooled[[c for c in cols if c in pooled.columns]]


# ================================================================== #
#  4. モデリング
# ================================================================== #
_FEATURE_COLS = (
    ['Meeting_Index', 'Days_to_MPM', 'Actual_Policy_Rate']
    + [f'M{i}_spread'     for i in range(1, 9)]
    + [f'M{i}_frac_diff'  for i in range(1, 9)]
    + [f'M{i}_is_imputed' for i in range(1, 9)]
    + ['USDJPY_frac_diff', 'JGB_Future_frac_diff', 'Nikkei225_frac_diff']
)


def _get_xy(df, target_col):
    feat = [c for c in _FEATURE_COLS if c in df.columns]
    data = df[feat + [target_col, 'Date', 'is_post_mpm']].copy()
    data = data[data['is_post_mpm'] == 0].dropna(subset=feat + [target_col])
    return data[feat], data[target_col], data['Date'], data['Meeting_Index']


def _train(X_tr, y_tr, X_val, y_val):
    params = dict(
        objective='regression', metric='rmse', verbosity=-1,
        boosting_type='gbdt', random_state=42, learning_rate=0.05,
        num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=5, min_data_in_leaf=30, lambda_l2=1.0,
    )
    cat = ['Meeting_Index'] if 'Meeting_Index' in X_tr.columns else []
    d_tr  = lgb.Dataset(X_tr,  label=y_tr,  categorical_feature=cat)
    d_val = lgb.Dataset(X_val, label=y_val, reference=d_tr, categorical_feature=cat)
    return lgb.train(params, d_tr, valid_sets=[d_val], valid_names=['valid'],
                     num_boost_round=300, callbacks=[lgb.log_evaluation(period=0)])


def _metrics(y_true, y_pred):
    if len(y_true) < 2:
        return dict(IC=np.nan, Direction_Accuracy=np.nan, Direction_Accuracy_LargeMove=np.nan)
    ic, _  = spearmanr(y_true, y_pred)
    ok     = np.sign(y_true) == np.sign(y_pred)
    thr    = np.percentile(np.abs(y_true), 75)
    lm     = np.abs(y_true) > thr
    return dict(IC=ic,
                Direction_Accuracy=ok.mean(),
                Direction_Accuracy_LargeMove=ok[lm].mean() if lm.any() else np.nan)


def walk_forward(df, target_col, start_date,
                 test_window_days=90, purge_days=5):
    X, y, dates, m_idx = _get_xy(df, target_col)
    uq   = sorted(dates.unique())
    s    = pd.to_datetime(start_date)
    curr = next((i for i, d in enumerate(uq) if d >= s), 1) or 1

    results, last_model, last_Xt = [], None, None
    fold = 0
    while curr < len(uq):
        tr_end  = uq[curr - 1] - pd.Timedelta(days=purge_days)
        ts_s    = uq[curr]
        tr_mask = dates <= tr_end
        ts_mask = (dates >= ts_s) & (dates < ts_s + pd.Timedelta(days=test_window_days))
        if not ts_mask.any():
            break

        tr_uq   = sorted(dates[tr_mask].unique())
        if len(tr_uq) < 10:
            break
        vd      = 60 if len(tr_uq) >= 300 else max(1, int(len(tr_uq) * 0.15))
        v_thr   = tr_uq[-vd]
        v_mask  = dates[tr_mask] >= v_thr

        Xtr = X[tr_mask][~v_mask]; ytr = y[tr_mask][~v_mask]
        Xvl = X[tr_mask][v_mask];  yvl = y[tr_mask][v_mask]
        Xts = X[ts_mask];          yts = y[ts_mask]

        model = _train(Xtr, ytr, Xvl, yvl)
        preds = model.predict(Xts)
        results.append(pd.DataFrame({
            'Fold': fold, 'Date': dates[ts_mask],
            'Meeting_Index': m_idx[ts_mask],
            'Actual': yts, 'Pred': preds,
        }))
        last_model, last_Xt = model, Xts
        fold += 1
        nxt  = ts_s + pd.Timedelta(days=test_window_days)
        curr = next((i for i, d in enumerate(uq) if d >= nxt), len(uq))

    out = pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()
    return out, last_model, last_Xt


# ================================================================== #
#  5. PDF ページ描画
# ================================================================== #
_BLUE   = 'steelblue'
_RED    = 'salmon'
_NAVY   = 'navy'
_CORAL  = 'coral'


def _fold_ic(res):
    return res.groupby('Fold').apply(lambda g: spearmanr(g['Actual'], g['Pred'])[0])


def _summary_row(res, label):
    m    = _metrics(res['Actual'], res['Pred'])
    fics = _fold_ic(res)
    icir = fics.mean() / fics.std() if fics.std() > 0 else np.nan
    pnl  = np.sign(res['Pred']) * res['Actual']
    sh   = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else np.nan
    cum  = pnl.cumsum()
    mdd  = (cum.cummax() - cum).max()
    return [label,
            f"{m['IC']:.4f}",
            f"{icir:.3f}",
            f"{m['Direction_Accuracy']:.1%}",
            f"{m['Direction_Accuracy_LargeMove']:.1%}",
            f"{sh:.3f}",
            f"{mdd:.5f}",
            str(int(res['Fold'].nunique()))]


# ---- Page 1: サマリーテーブル + フォールド別 IC ------------------
def _page1(pdf, res_3d, res_5d):
    fig = plt.figure(figsize=(13, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35)

    # テーブル
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis('off')
    cols = ['Target', 'IC', 'ICIR', 'Dir Acc.', 'Dir Acc.\n(Large)', 'Sharpe', 'Max DD', 'Folds']
    rows = [_summary_row(res_3d, '3d'), _summary_row(res_5d, '5d')]
    tbl  = ax_t.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c5f8a')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#eaf2ff')
    ax_t.set_title('BOJ OIS Swap — Model Performance Summary (OOS Walk-Forward)',
                   fontsize=12, fontweight='bold', pad=20)

    # フォールド別 IC
    for ax, res, lbl in [(fig.add_subplot(gs[1, 0]), res_3d, 'Target_3d'),
                         (fig.add_subplot(gs[1, 1]), res_5d, 'Target_5d')]:
        fics   = _fold_ic(res)
        fdates = res.groupby('Fold')['Date'].min().dt.strftime('%Y-%m')
        colors = [_BLUE if v > 0 else _RED for v in fics]
        ax.bar(range(len(fics)), fics.values, color=colors, edgecolor='black', linewidth=0.4)
        ax.axhline(0, color='black', linewidth=0.7)
        ax.axhline(fics.mean(), color=_NAVY, linestyle='--', linewidth=1.0,
                   label=f'Mean={fics.mean():.3f}  ICIR={fics.mean()/fics.std():.2f}')
        ax.set_xticks(range(len(fics)))
        ax.set_xticklabels(fdates.values, rotation=45, ha='right', fontsize=7)
        ax.set_title(f'IC by Fold — {lbl}', fontsize=10)
        ax.set_ylabel('IC (Spearman)')
        ax.legend(fontsize=8)

    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ---- Page 2: 限月別 IC + MPM 近接別 IC ---------------------------
def _page2(pdf, res_3d, res_5d):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('IC Breakdown — by Tenor (M1–M8) and Days to MPM',
                 fontsize=12, fontweight='bold')

    def _bar(ax, res, grp, xlbl_fn, title):
        ics    = res.groupby(grp).apply(lambda g: spearmanr(g['Actual'], g['Pred'])[0])
        xlbls  = xlbl_fn(ics.index)
        colors = [_BLUE if v > 0 else _RED for v in ics]
        ax.bar(xlbls, ics.values, color=colors, edgecolor='black', linewidth=0.4)
        ax.axhline(0, color='black', linewidth=0.7)
        ax.axhline(ics.mean(), color=_NAVY, linestyle='--', linewidth=1.0,
                   label=f'Mean={ics.mean():.3f}')
        for j, (v, lbl) in enumerate(zip(ics.values, xlbls)):
            ax.text(j, v + 0.004 * (1 if v >= 0 else -1),
                    f'{v:.3f}', ha='center', fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('IC')
        ax.legend(fontsize=8)

    _bar(axes[0, 0], res_3d, 'Meeting_Index',
         lambda idx: [f'M{int(i)}' for i in idx], 'IC by Tenor — Target_3d')
    _bar(axes[0, 1], res_5d, 'Meeting_Index',
         lambda idx: [f'M{int(i)}' for i in idx], 'IC by Tenor — Target_5d')

    # Days_to_MPM バケット
    BKTS = ['<=5', '6-15', '16+']

    def _bkt(d):
        if pd.isna(d): return '16+'
        return '<=5' if d <= 5 else ('6-15' if d <= 15 else '16+')

    for ax, res, lbl in [(axes[1, 0], res_3d, 'Target_3d'),
                         (axes[1, 1], res_5d, 'Target_5d')]:
        r = res.copy(); r['b'] = r['Days_to_MPM'].apply(_bkt)
        bics = r.groupby('b').apply(
            lambda g: spearmanr(g['Actual'], g['Pred'])[0]).reindex(BKTS)
        bn   = r.groupby('b').size().reindex(BKTS)
        colors = [_BLUE if (v or 0) > 0 else _RED for v in bics.fillna(0)]
        ax.bar(BKTS, bics.values, color=colors, edgecolor='black', linewidth=0.4)
        ax.axhline(0, color='black', linewidth=0.7)
        for j, bk in enumerate(BKTS):
            v = bics.get(bk) or 0
            n = bn.get(bk, 0)
            ax.text(j, v + 0.004 * (1 if v >= 0 else -1),
                    f'{v:.3f}\n(n={n})', ha='center', fontsize=8)
        ax.set_title(f'IC by Days to MPM — {lbl}', fontsize=10)
        ax.set_ylabel('IC')
        ax.set_xlabel('Days to Next MPM')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ---- Page 3: 限月別 累積 P&L (5d) --------------------------------
def _page3(pdf, res_5d):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    fig.suptitle(
        'Cumulative P&L by Tenor — 5d Model  (OOS, Sign-Follow Strategy)\n'
        'P&L per trade = sign(Pred) × Actual 5-day OIS spread change',
        fontsize=11, fontweight='bold')

    for i, mi in enumerate(range(1, 9)):
        t = res_5d[res_5d['Meeting_Index'] == mi].sort_values('Date').copy()
        t['PnL']    = np.sign(t['Pred']) * t['Actual']
        t['CumPnL'] = t['PnL'].cumsum()
        total  = t['CumPnL'].iloc[-1]
        sharpe = (t['PnL'].mean() / t['PnL'].std() * np.sqrt(252)
                  if t['PnL'].std() > 0 else np.nan)
        mdd    = (t['CumPnL'].cummax() - t['CumPnL']).max()
        hit    = (t['PnL'] > 0).mean()

        ax = axes[i]
        ax.plot(t['Date'], t['CumPnL'], color=_BLUE, linewidth=1.3)
        ax.fill_between(t['Date'], t['CumPnL'], 0,
                        where=t['CumPnL'] >= 0, alpha=0.15, color=_BLUE)
        ax.fill_between(t['Date'], t['CumPnL'], 0,
                        where=t['CumPnL'] <  0, alpha=0.15, color=_RED)
        ax.axhline(0, color='black', linewidth=0.5)
        txt = f'Total ={total:+.4f}\nSharpe={sharpe:.2f}\nMaxDD ={mdd:.4f}\nHit   ={hit:.1%}'
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', fontsize=7.5,
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75))
        ax.set_title(f'M{mi}', fontsize=10, fontweight='bold')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Cum P&L (rate units)', fontsize=7.5)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ---- Page 4: 最新日予測バーチャート ----------------------------------
def _page4(pdf, model_3d, model_5d, df_pooled):
    latest = df_pooled['Date'].max()
    lrows  = df_pooled[df_pooled['Date'] == latest].sort_values('Meeting_Index').copy()
    tenors = [f'M{int(mi)}' for mi in lrows['Meeting_Index'].values]
    dtm_val = int(lrows['Days_to_MPM'].iloc[0])

    fc3 = model_3d.feature_name()
    fc5 = model_5d.feature_name()
    # % → bps: ×100（OIS レートは % 単位で格納されているため）
    p3  = model_3d.predict(lrows[fc3]) * 100
    p5  = model_5d.predict(lrows[fc5]) * 100

    x     = np.arange(len(tenors))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b3 = ax.bar(x - width/2, p3, width, label='3d prediction',
                color=_BLUE, edgecolor='black', linewidth=0.5, alpha=0.85)
    b5 = ax.bar(x + width/2, p5, width, label='5d prediction',
                color=_CORAL, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)

    for bar in list(b3) + list(b5):
        h  = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + (0.05 if h >= 0 else -0.05),
                f'{h:+.2f}', ha='center', va=va, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(tenors)
    ax.set_xlabel('Tenor')
    ax.set_ylabel('Predicted OIS Spread Change (bps)')
    ax.set_title(
        f'Latest Predictions  as of {latest.date()}\n'
        f'3-day & 5-day predicted OIS spread change per tenor'
        f'   (Days to MPM = {dtm_val})',
        fontsize=11, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ================================================================== #
#  6. メイン
# ================================================================== #
def main():
    print('BOJ OIS Analysis Report Generator')
    print('=' * 45)

    print('[1/4] Loading and processing data...')
    df_raw    = load_and_clean_data(EXCEL_PATH, MEETING_CSV_PATH)
    df_feat   = generate_features(df_raw)
    df_pooled = pool_boj_data(df_feat)
    dtm_map   = df_pooled[['Date', 'Meeting_Index', 'Days_to_MPM']].drop_duplicates()
    print(f'      Date range : {df_pooled["Date"].min().date()} — {df_pooled["Date"].max().date()}')
    print(f'      OOS start  : {OOS_START_DATE}')

    print('[2/4] Walk-forward validation (Target_3d)...')
    res_3d, model_3d, _ = walk_forward(df_pooled, 'Target_3d', OOS_START_DATE)

    print('[3/4] Walk-forward validation (Target_5d)...')
    res_5d, model_5d, _ = walk_forward(df_pooled, 'Target_5d', OOS_START_DATE)

    res_3d = res_3d.merge(dtm_map, on=['Date', 'Meeting_Index'], how='left')
    res_5d = res_5d.merge(dtm_map, on=['Date', 'Meeting_Index'], how='left')

    # コンソールにサマリー出力
    for res, lbl in [(res_3d, '3d'), (res_5d, '5d')]:
        m    = _metrics(res['Actual'], res['Pred'])
        fics = _fold_ic(res)
        icir = fics.mean() / fics.std()
        pnl  = np.sign(res['Pred']) * res['Actual']
        sh   = pnl.mean() / pnl.std() * np.sqrt(252)
        print(f'      [{lbl}] IC={m["IC"]:.4f}  ICIR={icir:.3f}  '
              f'DirAcc={m["Direction_Accuracy"]:.1%}  Sharpe={sh:.3f}  '
              f'Folds={res["Fold"].nunique()}')

    print('[4/4] Generating PDF...')
    with PdfPages(OUTPUT_PATH) as pdf:
        meta = pdf.infodict()
        meta['Title']   = 'BOJ OIS Swap Analysis Report'
        meta['Subject'] = 'Walk-Forward Validation & Trading Analysis'

        _page1(pdf, res_3d, res_5d)   # サマリーテーブル + フォールド別IC
        _page2(pdf, res_3d, res_5d)   # 限月別IC + MPM近接別IC
        _page3(pdf, res_5d)           # 限月別 累積P&L
        _page4(pdf, model_3d, model_5d, df_pooled)  # 最新日予測

    print(f'\nDone. Report saved to:\n  {OUTPUT_PATH}')


if __name__ == '__main__':
    main()

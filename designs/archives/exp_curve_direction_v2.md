# 実験設計書: Curve Direction v2（Exp-CurveDirV2）

作成日：2026-03-22
担当：Gemini CLI（実装・実験）
レビュー：Claude（採否判断）

**ノートブック: `notebooks/exp_curve_direction_v2.ipynb`**

---

## 目的

カーブ方向予測モデルとして、プーリング戦略の異なる2モデルを比較する。

| モデル | プーリング対象 | 系列数 |
|--------|-------------|--------|
| **Model A: Adjacent** | C1〜C7（隣接差分のみ） | 7系列 |
| **Model B: AllPair** | M{i}-M{j}（全ペア） | 28系列 |

**主目的**: M2-M4・M2-M5 スパンについてどちらが方向予測精度が高いかを確認し、採用するプーリング方式を決める。

---

## 予測対象スパン（事前指定）

```
M2-M4 = M2_spread - M4_spread  (= C2 + C3)
M2-M5 = M2_spread - M5_spread  (= C2 + C3 + C4)
```

「28系列のうち最良を選ぶ」という事後選択は行わない。
評価するスパンをここで事前固定することで選択バイアスを排除する。

---

## 特徴量

両モデルで共通の特徴量セットを使う。Model B のみ識別子特徴量を追加。

| # | 特徴量 | Model A | Model B | 備考 |
|---|--------|---------|---------|------|
| 1 | `Span_Level` | ✅ | ✅ | 当該スパンの現在水準（mean reversion起点） |
| 2 | `Span_frac_diff` | ✅ | ✅ | 当該スパン全体で計算したfrac_diff（d=0.4, window=50） |
| 3 | `C1_frac_diff`〜`C7_frac_diff` | ✅ | ✅ | カーブ全体の変化速度のコンテキスト（7本） |
| 4 | `M1_spread` | ✅ | ✅ | 金利水準アンカー |
| 5 | `M1_frac_diff` | ✅ | ✅ | M1_spread の分数階差 |
| 6 | `Days_to_MPM` | ✅ | ✅ | 次回MPMまでの日数 |
| 7 | `Actual_Policy_Rate` | ✅ | ✅ | 現在の政策金利 |
| 8 | `M{n}_is_imputed`（構成銘柄分） | ✅ | ✅ | 補完フラグ（M1〜M8全部） |
| 9 | `USDJPY/JGB_Future/Nikkei225/DXY _frac_diff` | ✅ | ✅ | 外部指標（4本） |
| 10 | `Leg_Near`（カテゴリ） | — | ✅ | 近い限月 i（1〜7） |
| 11 | `Leg_Far`（カテゴリ） | — | ✅ | 遠い限月 j（2〜8） |
| 12 | `Span_Length` | — | ✅ | j - i（スパン幅、1〜7の整数） |

Model A のカテゴリ変数: `Meeting_Index`（=n, 1〜7）
Model B のカテゴリ変数: `Leg_Near`, `Leg_Far`

---

## 目的変数

```
Target_{h}d = Span_{t+h} - Span_t   （h = 3, 5）
Target_{h}d_norm = Target_{h}d / per_span_std
```

正規化はスパンごとの全期間 std で実施（異分散補正、IC評価には影響しない）。

---

## 評価方法

**CS IC は使わない。** 以下の3指標で評価する。

### Global IC（プーリングモデル全体の質）

全系列 × 全OOSの Spearman 相関。モデルが「どのスパンが上がり・下がるか」を予測できているか。

### M2-M4・M2-M5 の個別 IC と方向的中率

**モデルから M2-M4・M2-M5 の予測を取り出す方法：**

- **Model B**: M2-M4・M2-M5 が直接1系列として存在 → そのまま抽出
- **Model A**: M2-M4 は C2・C3 の合算で間接的に取り出す

```python
# Model A から M2-M4 を復元する手順
# 1. 各フォールドで C2・C3 の OOS 予測値を逆正規化（実 bps に戻す）
# 2. 逆正規化した C2_pred + C3_pred = Δ(M2-M4)_pred
# 3. Actual も同様に C2_actual + C3_actual = Δ(M2-M4)_actual
# ※ IC は Spearman なのでスケールは IC 計算に影響しないが、
#    Direction Accuracy のために逆正規化は必須
```

```python
def extract_span_from_adjacent(results, std_map, near, far):
    """
    Model A（隣接差分）の OOS 結果から M{near}-M{far} の予測を復元する。
    std_map: {Meeting_Index: Target_{h}d の per-pair std} を事前に計算して渡す。
    """
    span_pred   = pd.Series(0.0, index=results['Date'].unique())
    span_actual = pd.Series(0.0, index=results['Date'].unique())
    for n in range(near, far):          # C{near}, C{near+1}, ..., C{far-1}
        sub = results[results['Meeting_Index'] == n].set_index('Date')
        std = std_map[n]
        span_pred   += sub['Pred']   * std
        span_actual += sub['Actual'] * std
    return span_pred, span_actual
```

---

## 実装コード骨格

```python
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr

from src.processing import load_and_clean_data
from src.features_rv import generate_rv_features
from src.features import frac_diff
from src.pooling_curve import pool_curve_data
from src.modeling import walk_forward_validation, summarize_ic

START_DATE = '2024-01-01'

df_raw  = load_and_clean_data('../data/BOJ_data.xlsx', '../data/BOJ_meeting_history.csv')
df_feat = generate_rv_features(df_raw)

# ============================================================
# セル 1: RV Butterfly ベースライン再現
# ============================================================
from src.pooling_butterfly import pool_butterfly_data
df_fly = pool_butterfly_data(df_feat)
res_fly_3d = walk_forward_validation(df_fly, 'Target_3d_norm', START_DATE)
ic_fly_3d  = summarize_ic(res_fly_3d, instrument_indices=set(range(2, 8)))
print(f'[Butterfly baseline] CS IC 3d: {ic_fly_3d["cs_ic"]:.4f}')  # 期待値: ~0.304

# ============================================================
# Model A: Adjacent（C1〜C7、7系列）
# ============================================================
df_curve = pool_curve_data(df_feat)
res_a_3d = walk_forward_validation(df_curve, 'Target_3d_norm', START_DATE)
res_a_5d = walk_forward_validation(df_curve, 'Target_5d_norm', START_DATE)

ic_a_3d = summarize_ic(res_a_3d, instrument_indices=set(range(1, 8)))
ic_a_5d = summarize_ic(res_a_5d, instrument_indices=set(range(1, 8)))
print(f'[Model A] Global IC 3d={ic_a_3d["ic_all"]:.4f}  5d={ic_a_5d["ic_all"]:.4f}')

# --- M2-M4・M2-M5 を C{n} から復元 ---
# per-pair std の取得（pool_curve_data の Target_{h}d_std 列から）
std_map_3d = df_curve.groupby('Meeting_Index')['Target_3d_std'].first().to_dict()
std_map_5d = df_curve.groupby('Meeting_Index')['Target_5d_std'].first().to_dict()

def extract_span_from_adjacent(results, std_map, near, far):
    dates = sorted(results['Date'].unique())
    span_pred   = {d: 0.0 for d in dates}
    span_actual = {d: 0.0 for d in dates}
    for n in range(near, far):
        sub = results[results['Meeting_Index'] == n]
        std = std_map.get(n, 1.0)
        for _, row in sub.iterrows():
            span_pred[row['Date']]   += row['Pred']   * std
            span_actual[row['Date']] += row['Actual'] * std
    df_out = pd.DataFrame({'Date': dates,
                           'Pred': [span_pred[d] for d in dates],
                           'Actual': [span_actual[d] for d in dates]})
    return df_out.dropna()

# ============================================================
# Model B: AllPair（28系列）
# ============================================================
def build_allpair_df(df):
    """全ペア M{i}-M{j} (i<j, i,j in 1..8) をプーリングする。"""
    records = []
    pair_id = 0
    pair_info = {}   # pair_id -> (i, j)

    for i, j in combinations(range(1, 9), 2):
        span_col = f'Span_{i}_{j}'
        df[span_col] = df[f'M{i}_spread'] - df[f'M{j}_spread']
        df[f'{span_col}_fd'] = frac_diff(df[span_col], d=0.4, window=50)

        tmp = df[['Date', 'Days_to_MPM', 'Actual_Policy_Rate', 'is_post_mpm',
                  span_col, f'{span_col}_fd',
                  'C1_frac_diff', 'C2_frac_diff', 'C3_frac_diff',
                  'C4_frac_diff', 'C5_frac_diff', 'C6_frac_diff', 'C7_frac_diff',
                  'M1_spread', 'M1_frac_diff',
                  'M1_is_imputed', 'M2_is_imputed', 'M3_is_imputed', 'M4_is_imputed',
                  'M5_is_imputed', 'M6_is_imputed', 'M7_is_imputed', 'M8_is_imputed',
                  'USDJPY_frac_diff', 'JGB_Future_frac_diff',
                  'Nikkei225_frac_diff', 'DXY_frac_diff',
                  ]].copy()

        tmp['Span_Level']    = tmp[span_col]
        tmp['Span_frac_diff'] = tmp[f'{span_col}_fd']
        tmp['Leg_Near']      = i
        tmp['Leg_Far']       = j
        tmp['Span_Length']   = j - i
        tmp['Meeting_Index'] = pair_id   # summarize_ic 用のダミー（CS ICは使わない）
        tmp['Pair_Label']    = f'M{i}M{j}'
        records.append(tmp)
        pair_info[pair_id] = (i, j)
        pair_id += 1

    pooled = pd.concat(records, ignore_index=True)
    pooled = pooled.sort_values(['Pair_Label', 'Date']).reset_index(drop=True)

    for h in [3, 5]:
        pooled[f'Target_{h}d'] = (
            pooled.groupby('Pair_Label')['Span_Level'].shift(-h) - pooled['Span_Level']
        )
        std_vals = pooled.groupby('Pair_Label')[f'Target_{h}d'].transform('std')
        pooled[f'Target_{h}d_std']  = std_vals
        pooled[f'Target_{h}d_norm'] = pooled[f'Target_{h}d'] / std_vals

    # NON_FEATURE_COLS に含まれない列のみ残す
    drop_cols = [c for c in pooled.columns
                 if c.startswith('Span_') and c not in ('Span_Level', 'Span_frac_diff')]
    drop_cols += [c for c in pooled.columns if c.startswith('M') and '_spread' in c
                  and c not in ('M1_spread',)]
    drop_cols += ['Pair_Label']   # Rate_Label 相当（NON_FEATURE_COLS 対象）
    keep_cols = [c for c in pooled.columns if c not in drop_cols]
    return pooled[keep_cols], pair_info

df_allpair, pair_info = build_allpair_df(df_feat.copy())

res_b_3d = walk_forward_validation(df_allpair, 'Target_3d_norm', START_DATE)
res_b_5d = walk_forward_validation(df_allpair, 'Target_5d_norm', START_DATE)

ic_b_3d = summarize_ic(res_b_3d)
ic_b_5d = summarize_ic(res_b_5d)
print(f'[Model B] Global IC 3d={ic_b_3d["ic_all"]:.4f}  5d={ic_b_5d["ic_all"]:.4f}')

# ============================================================
# 評価: M2-M4・M2-M5 の個別 IC と方向的中率
# ============================================================
def direction_accuracy(pred, actual):
    valid = pd.DataFrame({'pred': pred, 'actual': actual}).dropna()
    n = len(valid)
    if n == 0:
        return np.nan, np.nan
    dir_acc = (np.sign(valid['pred']) == np.sign(valid['actual'])).mean()
    thr = valid['actual'].abs().quantile(0.75)
    large = valid[valid['actual'].abs() > thr]
    dir_large = (np.sign(large['pred']) == np.sign(large['actual'])).mean() if len(large) > 0 else np.nan
    return round(dir_acc, 4), round(dir_large, 4)

def global_ic(pred, actual):
    valid = pd.DataFrame({'pred': pred, 'actual': actual}).dropna()
    if len(valid) < 2:
        return np.nan
    return round(spearmanr(valid['pred'], valid['actual'])[0], 4)

for h, res_a, res_b, std_map in [(3, res_a_3d, res_b_3d, std_map_3d),
                                   (5, res_a_5d, res_b_5d, std_map_5d)]:
    print(f'\n=== {h}d ===')
    for (near, far), label in [((2, 4), 'M2-M4'), ((2, 5), 'M2-M5')]:
        # Model A
        sp_a = extract_span_from_adjacent(res_a, std_map, near, far)
        ic_a   = global_ic(sp_a['Pred'], sp_a['Actual'])
        da_a, da_large_a = direction_accuracy(sp_a['Pred'], sp_a['Actual'])

        # Model B（Leg_Near=near, Leg_Far=far の行を抽出）
        pid = [k for k, v in pair_info.items() if v == (near, far)][0]
        sp_b = res_b[res_b['Meeting_Index'] == pid].copy()
        # 逆正規化
        std_b = df_allpair[df_allpair['Meeting_Index'] == pid][f'Target_{h}d_std'].iloc[0]
        sp_b_pred   = sp_b['Pred']   * std_b
        sp_b_actual = sp_b['Actual'] * std_b
        ic_b   = global_ic(sp_b_pred, sp_b_actual)
        da_b, da_large_b = direction_accuracy(sp_b_pred, sp_b_actual)

        print(f'  [{label}] Model A: IC={ic_a:.4f} Dir={da_a:.4f} Dir_Large={da_large_a:.4f}')
        print(f'           Model B: IC={ic_b:.4f} Dir={da_b:.4f} Dir_Large={da_large_b:.4f}')
```

---

## 採用基準

| 指標 | 閾値 | 備考 |
|------|------|------|
| Global IC (全系列) | ≥ 0.10 | プーリングモデルとして有効か |
| M2-M4 or M2-M5 の個別 IC | ≥ 0.08 | 単体方向予測として有効か |
| Direction Accuracy | ≥ 0.53 | ランダム（0.50）を上回るか |

Model A・Model B いずれかが基準を超えた場合、精度が高い方を採用。
両方が基準未達の場合は不採用とし、特徴量・ターゲット定義を再検討する。

---

## 報告フォーマット（Gemini → Claude）

```
=== Exp-CurveDirV2 ===

[全系列 Global IC]
Model A (Adjacent 7): 3d=X.XXX  5d=X.XXX  Train=X.XXX  Gap=X.XXX
Model B (AllPair 28): 3d=X.XXX  5d=X.XXX  Train=X.XXX  Gap=X.XXX

[M2-M4 個別]
        IC 3d   Dir 3d  Dir_L 3d  IC 5d   Dir 5d  Dir_L 5d
Model A  X.XXX  X.XXX   X.XXX     X.XXX   X.XXX   X.XXX
Model B  X.XXX  X.XXX   X.XXX     X.XXX   X.XXX   X.XXX

[M2-M5 個別]
（同上）

フォールド別 IC（全系列, 3d）:
  Model A: {F0: ..., F1: ..., ...}
  Model B: {F0: ..., F1: ..., ...}

採用推奨: Model A / Model B / 不採用
観察:
```

---

## 注意事項

- `start_date='2024-01-01'` 固定
- Model B で `Pair_Label` 列は `Rate_Label` 相当のため `NON_FEATURE_COLS` の除外対象として明示的に除くこと
- `Leg_Near`, `Leg_Far` は LightGBM にカテゴリ変数として渡す
- Model A の M2-M4 復元では逆正規化（×std）を忘れずに実施すること
- セル 1 で Butterfly CS IC=0.304/0.346 を再現してからメイン実験を実行

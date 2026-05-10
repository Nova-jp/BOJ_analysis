# 実験設計書: Curve Direction モデル（Exp-CurveDir）

作成日：2026-03-22
担当：Gemini CLI（実装・実験）
レビュー：Claude（採否判断）

**この設計書を受け取った Gemini は、独立したノートブック `notebooks/exp_curve_direction_model.ipynb` を作成して実験すること。**

---

## 目的

M2-M4 および M2-M5 スパンの方向予測モデルを作り、精度が高い方をカーブモデルとして採用する。

現行のバタフライモデルは「M{n}間の強弱（相対価値）」を捉えているが、
**カーブ全体のスティープネス方向**は別途予測する必要がある。
このモデルはその役割を担う。

---

## なぜ M2-M4 / M2-M5 か

- **純粋にBOJ会合期待のみの露出**：M{n}-T12のようなスポットOIS参照を使わないため、
  グローバルマクロ・Fed要因のノイズが混入しない
- **M2-M4 = C2 + C3、M2-M5 = C2 + C3 + C4**：隣接差分の和として分解可能で、
  バタフライモデルと同じ銘柄空間で一貫性がある
- BOJ OIS市場でトレーダブルかつボラティリティが出やすいスパン

---

## 目的変数

```
Span_M2M4 = M2_spread - M4_spread   （= C2 + C3）
Span_M2M5 = M2_spread - M5_spread   （= C2 + C3 + C4）

Target_{h}d = Span_{t+h} - Span_t   （h = 3, 5）
```

プーリングなし。日付レベルの時系列（1日1行）。
M2-M4 と M2-M5 でそれぞれ独立したモデルを学習・評価する。

---

## 特徴量一覧

| # | カテゴリ | 特徴量 | 備考 |
|---|---------|--------|------|
| 1 | 対象スパン水準 | `Span_Level` | 現在の Span 水準（mean reversion 起点） |
| 2 | | `Span_frac_diff` | Span の分数階差（d=0.4, window=50、Span 全体で計算） |
| 3 | カーブ動態 | `C1_frac_diff`〜`C7_frac_diff` | 全隣接差分の分数階差（カーブ形状の変化速度） |
| 4 | M1 アンカー | `M1_spread` | 金利水準コンテキスト |
| 5 | | `M1_frac_diff` | M1_spread の分数階差 |
| 6 | 会合コンテキスト | `Days_to_MPM` | 次回MPMまでの日数 |
| 7 | | `Actual_Policy_Rate` | 現在の政策金利 |
| 8 | 補完フラグ | `M2_is_imputed`, `M3_is_imputed`, `M4_is_imputed`, `M5_is_imputed` | 構成銘柄の MICE 補完フラグ |
| 9 | 外部指標 | `USDJPY_frac_diff`, `JGB_Future_frac_diff`, `Nikkei225_frac_diff`, `DXY_frac_diff` | |

合計: 約18特徴量（カテゴリ変数なし）

> `C{n}` の絶対水準（レベル値）は今回含めない。Span_Level でカーブ水準は表現済み。
> 採用後に追加実験で検討する。

---

## 評価指標

CS IC は不要。以下の2指標で評価する：

| 指標 | 内容 | 採用判断の軸 |
|------|------|-------------|
| **Global IC** | 全OOS期間の Spearman 相関 | 予測値と実現値の単調な相関 |
| **Direction Accuracy** | 符号一致率（予測 > 0 かつ実現 > 0 など） | トレードとして実際に役立つか |
| Direction Accuracy (Large) | 上位25%の大動き限定の方向的中率 | 大きな動きで勝てるか |

ベースライン（ランダム）: Direction Accuracy ≈ 0.50

---

## パイプライン実装

```python
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from src.processing import load_and_clean_data
from src.features_rv import generate_rv_features
from src.features import frac_diff
from src.modeling import walk_forward_validation, summarize_ic

START_DATE = '2024-01-01'

# --- データ・特徴量生成 ---
df_raw  = load_and_clean_data('../data/BOJ_data.xlsx', '../data/BOJ_meeting_history.csv')
df_feat = generate_rv_features(df_raw)

# --- Span の構築 ---
df_feat['Span_M2M4'] = df_feat['M2_spread'] - df_feat['M4_spread']   # = C2 + C3
df_feat['Span_M2M5'] = df_feat['M2_spread'] - df_feat['M5_spread']   # = C2 + C3 + C4

df_feat['Span_M2M4_frac_diff'] = frac_diff(df_feat['Span_M2M4'], d=0.4, window=50)
df_feat['Span_M2M5_frac_diff'] = frac_diff(df_feat['Span_M2M5'], d=0.4, window=50)


def build_df(df, span_col):
    """指定スパンの日付レベル DataFrame を構築する。"""
    d = df.copy()
    d['Span_Level']    = d[span_col]
    d['Span_frac_diff'] = d[f'{span_col}_frac_diff']

    # 目的変数（h日後の変化）
    for h in [3, 5]:
        d[f'Target_{h}d'] = d['Span_Level'].shift(-h) - d['Span_Level']

    # walk_forward_validation が Meeting_Index を必要とするためダミー付与
    # （単一系列モデルのため CS IC は使わない）
    d['Meeting_Index'] = 1

    feature_cols = [
        'Span_Level', 'Span_frac_diff',
        'C1_frac_diff', 'C2_frac_diff', 'C3_frac_diff', 'C4_frac_diff',
        'C5_frac_diff', 'C6_frac_diff', 'C7_frac_diff',
        'M1_spread', 'M1_frac_diff',
        'Days_to_MPM', 'Actual_Policy_Rate',
        'M2_is_imputed', 'M3_is_imputed', 'M4_is_imputed', 'M5_is_imputed',
        'USDJPY_frac_diff', 'JGB_Future_frac_diff',
        'Nikkei225_frac_diff', 'DXY_frac_diff',
        'Meeting_Index',   # ダミー（モデルには定数として無視される）
    ]
    keep = feature_cols + ['Date', 'is_post_mpm', 'Target_3d', 'Target_5d']
    return d[[c for c in keep if c in d.columns]]


df_m2m4 = build_df(df_feat, 'Span_M2M4')
df_m2m5 = build_df(df_feat, 'Span_M2M5')


def evaluate_direction(results):
    """方向的中率を計算する。"""
    valid = results.dropna(subset=['Actual', 'Pred'])
    n = len(valid)
    if n == 0:
        return {}
    correct = (np.sign(valid['Actual']) == np.sign(valid['Pred'])).sum()
    dir_acc = correct / n

    threshold = valid['Actual'].abs().quantile(0.75)
    large = valid[valid['Actual'].abs() > threshold]
    dir_large = (np.sign(large['Actual']) == np.sign(large['Pred'])).mean() if len(large) > 0 else np.nan

    return {'Direction_Accuracy': round(dir_acc, 4), 'Direction_Accuracy_Large': round(dir_large, 4)}


# --- walk-forward 評価 ---
results = {}
for label, df in [('M2M4', df_m2m4), ('M2M5', df_m2m5)]:
    for h in [3, 5]:
        target = f'Target_{h}d'
        res = walk_forward_validation(df, target, START_DATE)
        ic  = summarize_ic(res)
        dir_metrics = evaluate_direction(res)
        fold_ics = ic['ic_by_fold']
        results[f'{label}_{h}d'] = {
            'Global IC':  ic['ic_all'],
            'IC Recent':  ic['ic_recent'],
            'Train IC':   ic['train_ic'],
            'Gap':        ic['gap'],
            **dir_metrics,
            'fold_ics':   fold_ics,
        }
        print(f"[{label} {h}d] Global IC={ic['ic_all']:.4f}  "
              f"Train IC={ic['train_ic']:.4f}  Gap={ic['gap']:.4f}  "
              f"Dir={dir_metrics.get('Direction_Accuracy', 'N/A'):.4f}  "
              f"Dir_Large={dir_metrics.get('Direction_Accuracy_Large', 'N/A'):.4f}")
        print(f"          fold ICs: {fold_ics}")
```

---

## ベースライン比較

| 指標 | 参考値 | 備考 |
|------|--------|------|
| Global IC 3d | RV Butterfly: 0.373 | スケールが異なる（プーリング vs 単体）ため直接比較は注意 |
| Direction Accuracy | 0.50 | ランダム予測のベースライン |

単体系列モデルのため、プーリングモデルとの直接IC比較は意味を持たない点に注意。
**方向的中率 > 0.54 程度を実用的な目標とする。**

---

## 採用基準

以下を満たした方を採用：

- Global IC 3d または 5d ≥ 0.10（単体系列として有意）
- Direction Accuracy ≥ 0.53（ランダムを有意に上回る）
- フォールド別 IC に著しいばらつきがない

M2-M4・M2-M5 双方が基準未達の場合は不採用とし、別アプローチを検討する。

---

## 報告フォーマット（Gemini → Claude）

```
=== Exp-CurveDir 結果 ===

[M2M4 3d] Global IC=X.XXX, Train IC=X.XXX, Gap=X.XXX
           Dir=X.XXX, Dir_Large=X.XXX
           fold ICs: {F0: ..., F1: ..., ...}

[M2M4 5d] ...
[M2M5 3d] ...
[M2M5 5d] ...

採用推奨: M2M4 / M2M5 / 不採用（理由: ）
観察:
```

---

## 注意事項

- `start_date='2024-01-01'` 固定
- MPM 直後除外: `is_post_mpm == 1` は walk_forward_validation 内で自動除外
- セル 1 で RV Butterfly CS IC=0.304/0.346 を再現してからメイン実験を実行
- `Meeting_Index=1` はダミー。`summarize_ic` の CS IC は無視してよい

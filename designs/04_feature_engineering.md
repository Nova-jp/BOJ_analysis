# 特徴量エンジニアリング実験設計書 04

Gemini CLI への実装指示書。
設計方針・パイプライン全体は `designs/01_model_design.md` を先に読むこと。

---

## 背景と目的

現在のベースライン特徴量（36個）で OOS IC は 3d=0.389、5d=0.633。
以下の3種類の特徴量グループを**個別に追加**してベースラインと比較し、有効な特徴量を特定する。

---

## ベースライン特徴量（現行）

| カテゴリ | 特徴量 | 説明 |
|---------|--------|------|
| コンテキスト | `Meeting_Index`（カテゴリ変数） | 限月 1〜8 |
| | `Days_to_MPM`（数値・整数） | 次回会合までの日数 |
| | `Actual_Policy_Rate`（数値） | 現在の政策金利 |
| スプレッド水準 | `M1_spread`〜`M8_spread` | `M{n} - Actual_Policy_Rate` |
| 分数階差 | `M1_frac_diff`〜`M8_frac_diff` | スプレッドの分数階差 (d=0.4) |
| 補完フラグ | `M1_is_imputed`〜`M8_is_imputed` | MICE補完された行 |
| 外部指標 | `USDJPY_frac_diff` | ドル円の分数階差 |
| | `JGB_Future_frac_diff` | JGB先物の分数階差 |
| | `Nikkei225_frac_diff` | 日経225の分数階差 |

合計 36 特徴量。

---

## 実験一覧

| 実験ID | 追加する特徴量グループ | 追加数 |
|--------|---------------------|--------|
| Baseline | なし（現行のまま） | — |
| Exp-A | 曜日（sin/cos 円環） | +2 |
| Exp-B | Days_to_MPM（sin/cos 円環・数値は残す） | +2 |
| Exp-C | カーブ形状（Slope・Butterfly の水準＋分数階差） | +14 |
| Exp-ABC | A + B + C 全部追加 | +18 |

各実験は `walk_forward_validation` をそのまま使い、**特徴量リストだけを変える**ことで比較する。

---

## Exp-A：曜日の円環エンコーディング

### 動機

週の曜日は市場の流動性・ポジション調整パターンに影響する。
整数（月=0〜金=4）では「金曜と月曜が距離4」になってしまうため、
循環構造を持つ sin/cos で表現する。

> **GBDT での有効性について**
> GBDT はしきい値分割で曜日の非線形効果を学習できるため、整数のままでも表現力は十分。
> sin/cos の効果は限定的かもしれないが、実験で確認する。

### 実装内容（`src/features.py` の `generate_features` に追加）

```python
# 曜日（月=0, 火=1, ..., 金=4）
feat_df['DayOfWeek'] = feat_df['Date'].dt.dayofweek  # 0〜4

# 円環エンコーディング（週5日サイクル）
feat_df['DayOfWeek_sin'] = np.sin(2 * np.pi * feat_df['DayOfWeek'] / 5)
feat_df['DayOfWeek_cos'] = np.cos(2 * np.pi * feat_df['DayOfWeek'] / 5)
```

`DayOfWeek`（整数）は特徴量に**含めない**（sin/cos のみ使う）。

---

## Exp-B：Days_to_MPM の円環エンコーディング

### 重要な前提

**現状**：`Days_to_MPM` は単純な整数として LightGBM に渡されている。
`Meeting_Index` だけがカテゴリ変数として扱われており、`Days_to_MPM` は数値のまま。

> **GBDT での有効性について**
> GBDT はしきい値分割（`Days_to_MPM ≤ 5` など）で非線形関係を学習できるため、
> 整数のまま渡しても十分な表現力がある。
> また「会合直後（Days ≈ 40）」と「会合直前（Days ≈ 0）」を「近い」とみなす
> 円環表現は市場実態に合わない可能性がある。
> ただし「会合サイクル内の相対位置」として有用かどうかは実験で確認する。

### 実装内容（`src/features.py` の `generate_features` に追加）

```python
# BOJ 会合は概ね 35〜45 日間隔。最大サイクルとして 45 日を使用。
CYCLE = 45

feat_df['Days_to_MPM_sin'] = np.sin(2 * np.pi * feat_df['Days_to_MPM'] / CYCLE)
feat_df['Days_to_MPM_cos'] = np.cos(2 * np.pi * feat_df['Days_to_MPM'] / CYCLE)
```

既存の `Days_to_MPM`（数値）は**そのまま残す**（置き換えではなく追加）。
この実験では `Days_to_MPM` 関連が合計 3 特徴量になる。

---

## Exp-C：カーブ形状特徴量

### 動機

現在の特徴量は各限月のスプレッドを独立に扱っている。
OIS カーブ全体の形状（傾き・曲率）はより直接的に「次回以降の会合への期待の変化」を捉える。
`Days_to_MPM` の次に重要な説明変数になる可能性がある。

### 実装内容（`src/features.py` の `generate_features` に追加）

```python
# 1. スロープ：長端 - 短端（カーブ全体の傾き）
feat_df['Curve_Slope'] = feat_df['M8_spread'] - feat_df['M1_spread']

# 2. バタフライ：n=2〜7（各限月における カーブの局所的な曲率）
for n in range(2, 8):
    feat_df[f'Butterfly_M{n}'] = (
        2 * feat_df[f'M{n}_spread']
        - feat_df[f'M{n-1}_spread']
        - feat_df[f'M{n+1}_spread']
    )

# 3. 上記の分数階差（変化速度）
for col in ['Curve_Slope'] + [f'Butterfly_M{n}' for n in range(2, 8)]:
    feat_df[f'{col}_frac_diff'] = frac_diff(feat_df[col], d=0.4, window=50)
```

追加される特徴量：

| 特徴量 | 説明 |
|--------|------|
| `Curve_Slope` | M8_spread - M1_spread（水準） |
| `Butterfly_M2`〜`Butterfly_M7` | 局所曲率（水準）×6個 |
| `Curve_Slope_frac_diff` | スロープの変化速度 |
| `Butterfly_M2_frac_diff`〜`Butterfly_M7_frac_diff` | 曲率の変化速度 ×6個 |

合計 **14 個**。

---

## 実装手順

### Step 1：`src/features.py` の更新

`generate_features` にフラグ引数を追加する：

```python
def generate_features(
    df,
    d=0.4,
    window=50,
    add_weekday_cyclic=False,     # Exp-A
    add_days_to_mpm_cyclic=False, # Exp-B
    add_curve_features=False,     # Exp-C
):
```

各フラグが `True` のときのみ対応する特徴量を生成する。
既存の処理は**変更しない**（後方互換を保つ）。

### Step 2：`src/pooling.py` の更新

`pool_boj_data` の `final_cols` に追加特徴量を含める。
**既存の `available_cols` フィルタ（存在する列のみ選択）を活用すればよい**ので、
`final_cols` のリストに以下を追加する：

```python
# カーブ特徴量
curve_cols = ['Curve_Slope'] + [f'Butterfly_M{n}' for n in range(2, 8)]
curve_fd_cols = [f'{c}_frac_diff' for c in curve_cols]

# 円環特徴量
cyclic_cols = ['DayOfWeek_sin', 'DayOfWeek_cos',
               'Days_to_MPM_sin', 'Days_to_MPM_cos']

final_cols = basic_cols + ... + curve_cols + curve_fd_cols + cyclic_cols + target_cols
```

### Step 3：`src/modeling.py` の更新

`get_features_and_target` の特徴量リスト構築を**「存在する列のみ使う」方式**に変更する：

```python
# ベース特徴量（常に使う）
base_cols = context_cols + boj_spread_cols + boj_fd_cols + boj_imp_cols + ext_fd_cols

# 追加特徴量（列が存在する場合のみ使う）
optional_cols = (
    ['DayOfWeek_sin', 'DayOfWeek_cos',
     'Days_to_MPM_sin', 'Days_to_MPM_cos',
     'Curve_Slope'] +
    [f'Butterfly_M{n}' for n in range(2, 8)] +
    ['Curve_Slope_frac_diff'] +
    [f'Butterfly_M{n}_frac_diff' for n in range(2, 8)]
)

feature_cols = base_cols + [c for c in optional_cols if c in df.columns]
```

### Step 4：実験ノートブック `notebooks/07_feature_engineering_experiments.ipynb` の作成

`.ipynb` は Python の `json` ライブラリ経由で生成すること。

#### セル構成

**セル 1：セットアップ**

```python
import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
from src.processing import load_and_clean_data
from src.features import generate_features
from src.pooling import pool_boj_data
from src.modeling import walk_forward_validation, calculate_metrics

EXCEL_PATH = '../data/BOJ_data.xlsx'
MEETING_CSV_PATH = '../data/BOJ_meeting_history.csv'
START_DATE = '2024-01-01'

df_raw = load_and_clean_data(EXCEL_PATH, MEETING_CSV_PATH)
```

**セル 2〜6：各実験の実行**

各実験を以下のパターンで実行し、OOS IC を記録する：

```python
def run_experiment(df_raw, label, **feat_kwargs):
    df_feat = generate_features(df_raw, **feat_kwargs)
    df_pooled = pool_boj_data(df_feat)
    res_3d = walk_forward_validation(df_pooled, 'Target_3d', START_DATE)
    res_5d = walk_forward_validation(df_pooled, 'Target_5d', START_DATE)
    ic_3d = calculate_metrics(res_3d['Actual'], res_3d['Pred'])['IC']
    ic_5d = calculate_metrics(res_5d['Actual'], res_5d['Pred'])['IC']
    print(f"{label}: IC_3d={ic_3d:.4f}, IC_5d={ic_5d:.4f}")
    return ic_3d, ic_5d

baseline     = run_experiment(df_raw, 'Baseline')
exp_a        = run_experiment(df_raw, 'Exp-A (Weekday)',      add_weekday_cyclic=True)
exp_b        = run_experiment(df_raw, 'Exp-B (DTM cyclic)',   add_days_to_mpm_cyclic=True)
exp_c        = run_experiment(df_raw, 'Exp-C (Curve)',        add_curve_features=True)
exp_abc      = run_experiment(df_raw, 'Exp-ABC (All)',
                              add_weekday_cyclic=True,
                              add_days_to_mpm_cyclic=True,
                              add_curve_features=True)
```

**セル 7：比較表の出力**

```python
results = {
    'Experiment': ['Baseline', 'Exp-A', 'Exp-B', 'Exp-C', 'Exp-ABC'],
    'IC_3d': [baseline[0], exp_a[0], exp_b[0], exp_c[0], exp_abc[0]],
    'IC_5d': [baseline[1], exp_a[1], exp_b[1], exp_c[1], exp_abc[1]],
}
df_results = pd.DataFrame(results)
df_results['Delta_IC_3d'] = df_results['IC_3d'] - baseline[0]
df_results['Delta_IC_5d'] = df_results['IC_5d'] - baseline[1]
print(df_results.to_string(index=False))
```

**セル 8：Exp-C の特徴量重要度**

`walk_forward_validation` を `return_model=True` で呼び出し、
最終フォールドのモデルで Gain importance を可視化する（上位 25 特徴量を横棒グラフ）。
3d・5d を並べて比較する。

---

## 評価基準

| 判定 | 条件 |
|------|------|
| **採用** | OOS IC（3d または 5d）が +0.01 以上改善かつ全フォールドで安定 |
| **要検討** | +0.005〜+0.01 の改善（他の根拠と合わせて判断） |
| **不採用** | 改善なし または 悪化 |

---

## Gemini CLI への報告内容

実験完了後に以下を報告すること：

1. 各実験の OOS IC（3d・5d）と Baseline との差（Delta IC）の比較表
2. Exp-B（Days_to_MPM 円環）の結果に対するコメント：数値の Days_to_MPM に対して改善があったか
3. Exp-C のカーブ特徴量の Gain importance 上位 5 特徴量とその値
4. 採用推奨の特徴量グループ（上記評価基準に基づく）

---

## 注意事項

- **データリーク厳禁**：カーブ特徴量は横持ちデータ（同一時点の M1〜M8）から計算するため将来データの参照は発生しない（問題なし）
- **NaN 処理**：分数階差（window=50）によりデータ先頭 50 行は NaN になる。既存の `dropna` 処理で自動除外される
- `pool_boj_data` は `available_cols` フィルタが既に実装されているので、`final_cols` に列を追加するだけでよい

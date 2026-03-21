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

## 注意事項（Exp-A/B/C 共通）

- **データリーク厳禁**：カーブ特徴量は横持ちデータ（同一時点の M1〜M8）から計算するため将来データの参照は発生しない（問題なし）
- **NaN 処理**：分数階差（window=50）によりデータ先頭 50 行は NaN になる。既存の `dropna` 処理で自動除外される
- `pool_boj_data` は `available_cols` フィルタが既に実装されているので、`final_cols` に列を追加するだけでよい

---

---

# 第2フェーズ実験：会合識別子とバタフライ目的変数変換

実験追加日：2026-03-20

---

## 実験一覧（第2フェーズ）

| 実験ID | 概要 | 変更箇所 |
|--------|------|---------|
| Exp-D | 絶対会合番号（Absolute_Meeting_ID）の追加 | `src/pooling.py` |
| Exp-E | 初観測からの経過日数（Days_since_first_seen）の追加 | `src/pooling.py` |
| Exp-DE | D + E 両方追加 | `src/pooling.py` |
| Exp-F | バタフライ空間での目的変数変換（新関数 `pool_boj_butterfly`） | `src/pooling.py` |

各実験は第1フェーズと同様、`walk_forward_validation` を使い **特徴量/目的変数リストのみを変える**ことでベースラインと比較する。

---

## Exp-D：絶対会合番号（Absolute_Meeting_ID）

### 動機

現行の `Meeting_Index`（1〜8）は「次から何番目の会合か」という**相対インデックス**である。
MPMが開催されるたびにロールするため、例えば「6月会合」を指す列は
- 4月時点では `M2`（Meeting_Index=2）
- 3月時点では `M3`（Meeting_Index=3）

と変化する。モデルは「これが6月会合である」という情報を持てず、会合固有の市場コンテキスト（その会合への累積的な織り込み）が失われている。

絶対会合番号（データ中の最初の会合=0として連番）を追加することで、
モデルが会合固有のパターンを学習できる余地を与える。

### 実装内容（`src/pooling.py` の `pool_boj_data` に追加）

melt後のDataFrameに以下の計算を追加する。`Days_to_MPM` は processing.py で計算済みのため、ここで `next_meeting_date` を再構成できる。

```python
# --- Absolute_Meeting_ID の計算 ---
# 1. df の各行から「次の会合日」を再構成
df_dates = df[['Date', 'Days_to_MPM']].drop_duplicates()
df_dates = df_dates.dropna(subset=['Days_to_MPM'])
df_dates['Next_Meeting_Date'] = df_dates['Date'] + pd.to_timedelta(
    df_dates['Days_to_MPM'].astype(int), unit='D'
)

# 2. 全会合日をソートしてランク辞書を作る
all_meeting_dates = sorted(df_dates['Next_Meeting_Date'].unique())
meeting_date_to_rank = {m: i for i, m in enumerate(all_meeting_dates)}

# 3. 各日付の「次の会合」のランクを引く
date_to_next_rank = dict(zip(df_dates['Date'], df_dates['Next_Meeting_Date'].map(meeting_date_to_rank)))

# 4. pooled DataFrame に適用（melt後なので Meeting_Index 列が使える）
#    Absolute_Meeting_ID = rank(次の会合) + Meeting_Index - 1
pooled['_next_rank'] = pooled['Date'].map(date_to_next_rank)
pooled['Absolute_Meeting_ID'] = (pooled['_next_rank'] + pooled['Meeting_Index'] - 1).astype('Int64')
pooled = pooled.drop(columns=['_next_rank'])
```

`Is_Tenor_OIS=1` の行（T12/T18/T24）は `Absolute_Meeting_ID` が不定義（会合先でないため）。
それらの行は NaN のままにする（モデルへは整数/NaN で渡してよい）。

### 特徴量扱い

- `Absolute_Meeting_ID` は**整数型カテゴリ変数**として LightGBM に渡す
- `Meeting_Index` は引き続き残す（ロール前後の相対情報として補完的）

### 実験手順

```python
# pooling.py 側でフラグを追加するのではなく、pool_boj_data の戻り値に常に
# Absolute_Meeting_ID を含める形にする（後から drop すれば Baseline と比較できる）

def run_exp_d(df_raw, label):
    df_feat = generate_features(df_raw)
    df_pooled = pool_boj_data(df_feat)   # Absolute_Meeting_ID が含まれた状態

    # Baseline: Absolute_Meeting_ID を除外したモデル
    # Exp-D:   Absolute_Meeting_ID を含めたモデル
    # → walk_forward_validation の feature_cols 引数で切り替える
```

---

## Exp-E：初観測からの経過日数（Days_since_first_seen）

### 動機

M_n の絶対会合番号 K が分かれば、その会合が「初めて M8 として観測された日」も計算できる。
K 番目の会合が M8 として初登場するのは、「K-8 番目の会合が開催された日」（その翌営業日）である。

`Days_since_first_seen` = (当日 - 初観測日).days

これは「市場がその会合をどれだけ長く折り込んできたか」を定量化し、
会合の生成・消滅ライフサイクル上の位置を表す。

> **Days_to_MPM との線形相関について**
> `Days_since_first_seen ≈ C - Days_to_MPM`（C は会合サイクル長）となる傾向があり、
> 新たな情報量は限定的かもしれない。ただし会合サイクル長の変動（6〜8週のばらつき）や、
> データ開始時点の打ち切り効果があるため、完全な線形変換にはならない。
> 実験でSHAPを確認し、`Days_to_MPM` と独立した寄与があるかを検証する。

### 実装内容（Exp-D の処理の後に続けて `pool_boj_data` に追加）

```python
# --- Days_since_first_seen の計算 ---
# 会合 K が初めて観測される日 = all_meeting_dates[K-8] が開催された日
# （K < 8 の場合はデータ開始日を使う）
data_start_date = df['Date'].min()

def get_first_seen_date(abs_id):
    if pd.isna(abs_id):
        return pd.NaT
    k = int(abs_id)
    predecessor_idx = k - 8
    if predecessor_idx < 0:
        return data_start_date
    elif predecessor_idx < len(all_meeting_dates):
        return all_meeting_dates[predecessor_idx]
    else:
        return pd.NaT

abs_id_to_first_seen = {
    k: get_first_seen_date(k) for k in pooled['Absolute_Meeting_ID'].dropna().unique()
}

pooled['First_Seen_Date'] = pooled['Absolute_Meeting_ID'].map(abs_id_to_first_seen)
pooled['Days_since_first_seen'] = (pooled['Date'] - pooled['First_Seen_Date']).dt.days
pooled = pooled.drop(columns=['First_Seen_Date'])
```

### 実験手順

Exp-D と同様に、`walk_forward_validation` の `feature_cols` で切り替える：

| 実験 | feature_cols への追加 |
|------|---------------------|
| Baseline | なし |
| Exp-D | `Absolute_Meeting_ID` |
| Exp-E | `Days_since_first_seen` |
| Exp-DE | `Absolute_Meeting_ID` + `Days_since_first_seen` |

---

## Exp-D/E の比較表出力

```python
results = {
    'Experiment': ['Baseline', 'Exp-D', 'Exp-E', 'Exp-DE'],
    'IC_3d': [...],
    'IC_5d': [...],
}
df_results = pd.DataFrame(results)
df_results['Delta_IC_3d'] = df_results['IC_3d'] - baseline_ic_3d
df_results['Delta_IC_5d'] = df_results['IC_5d'] - baseline_ic_5d
print(df_results.to_string(index=False))
```

Exp-DE で SHAP を確認し、`Absolute_Meeting_ID` と `Days_since_first_seen` それぞれの Gain importance および SHAP 寄与を報告すること。

---

## Exp-F：バタフライ空間での目的変数変換

### 動機

現行モデルは「M_n のレート変化（ΔM_n）」を予測する。
一方、OISカーブの形状変化を **水準（Level）・スロープ（Slope）・曲率（Butterfly）** に分解すると、
各成分の変化は相互に独立性が高く、予測しやすい可能性がある。

また、同じGBDTモデルが「レート全体が上がるか」と「カーブの曲率がどう変わるか」を
混在した目的変数で学習するより、独立した成分ごとに最適化した方が精度が上がる可能性がある。

### バタフライ変換の定義（M_n → B_k の線形変換）

M_n のスプレッド S_n = M_n - Actual_Policy_Rate を入力とする（Policy_Rate は定数として消去される）。

```
B1 = S1                          （水準：M1スプレッドをアンカーとする）
B2 = 2*S2 - S1 - S3             （M2の局所曲率）
B3 = 2*S3 - S2 - S4
B4 = 2*S4 - S3 - S5
B5 = 2*S5 - S4 - S6
B6 = 2*S6 - S5 - S7
B7 = 2*S7 - S6 - S8             （M7の局所曲率）
B8 = S8 - S1                     （スロープ：長端 - 短端）
```

目的変数は `ΔB_k(h) = B_k(t+h) - B_k(t)`（h日後のバタフライ成分変化量）。

### 逆変換（B_k → M_n）の数学的導出

B1, B8 から境界条件を復元：

```
S1 = B1
S8 = B1 + B8
```

S2〜S7 は次のトリディアゴナル方程式を解いて得る：

```
A * [S2, S3, S4, S5, S6, S7]^T = [B2 + S1, B3, B4, B5, B6, B7 + S8]^T

    [ 2 -1  0  0  0  0 ]
    [-1  2 -1  0  0  0 ]
A = [ 0 -1  2 -1  0  0 ]   （6×6 対称正定値行列 → 一意解が存在）
    [ 0  0 -1  2 -1  0 ]
    [ 0  0  0 -1  2 -1 ]
    [ 0  0  0  0 -1  2 ]
```

Python実装：

```python
import numpy as np

def butterfly_to_spread(B):
    """
    B: array-like, shape (8,) [B1, B2, ..., B8]
    Returns S: array-like, shape (8,) [S1, S2, ..., S8]
    """
    B = np.asarray(B, dtype=float)
    S = np.zeros(8)
    S[0] = B[0]           # S1 = B1
    S[7] = B[0] + B[7]   # S8 = B1 + B8

    A = (2 * np.eye(6)
         - np.eye(6, k=1)
         - np.eye(6, k=-1))
    rhs = B[1:7].copy()
    rhs[0]  += S[0]   # B2 + S1
    rhs[-1] += S[7]   # B7 + S8

    S[1:7] = np.linalg.solve(A, rhs)
    return S
```

変化量の逆変換も同一の線形変換で適用できる（`butterfly_to_spread(ΔB)` = ΔS）。

### 実装内容：`src/pooling.py` に新関数 `pool_boj_butterfly` を追加

既存の `pool_boj_data` は変更しない。新関数として実装する。

```python
def pool_boj_butterfly(df):
    """
    M1-M8 スプレッドをバタフライ空間（B1-B8）に変換し、
    バタフライ成分の h 日変化量を目的変数としてプーリングする。

    Returns:
        pooled_b: バタフライ空間のプーリング済み DataFrame
                  Target_{h}d_B: Bk の h 日変化量
                  Target_{h}d_B_norm: per-instrument std で正規化
    """
    spread_cols = [f'M{i}_spread' for i in range(1, 9)]

    # 1. バタフライ変換（行ごとに適用）
    S = df[spread_cols].values  # shape: (N, 8)
    B = np.zeros_like(S)
    B[:, 0] = S[:, 0]                                    # B1
    for k in range(1, 7):                                # B2-B7
        B[:, k] = 2 * S[:, k] - S[:, k-1] - S[:, k+1]
    B[:, 7] = S[:, 7] - S[:, 0]                         # B8 (slope)

    df_b = df.copy()
    for k in range(8):
        df_b[f'B{k+1}'] = B[:, k]

    # 2. プーリング（B1-B8 を melt）
    boj_rate_cols = [f'M{i}' for i in range(1, 9)]
    b_cols = [f'B{k}' for k in range(1, 9)]
    id_cols = [col for col in df_b.columns if col not in boj_rate_cols + b_cols]

    pooled_b = df_b.melt(
        id_vars=id_cols,
        value_vars=b_cols,
        var_name='Rate_Label',
        value_name='Rate_Value',
    )
    pooled_b['Butterfly_Index'] = pooled_b['Rate_Label'].str.extract(r'(\d+)').astype(int)

    # 3. 目的変数の生成（バタフライ成分の h 日変化量）
    pooled_b = pooled_b.sort_values(['Rate_Label', 'Date']).reset_index(drop=True)
    for h in [3, 5]:
        pooled_b[f'Target_{h}d_B'] = (
            pooled_b.groupby('Rate_Label')['Rate_Value'].shift(-h)
            - pooled_b['Rate_Value']
        )
        instr_std = pooled_b.groupby('Rate_Label')[f'Target_{h}d_B'].transform('std')
        pooled_b[f'Target_{h}d_B_std']  = instr_std
        pooled_b[f'Target_{h}d_B_norm'] = pooled_b[f'Target_{h}d_B'] / instr_std

    return pooled_b
```

### 評価の2通り

#### 評価①：バタフライ空間での IC

`walk_forward_validation` の target_col に `Target_{h}d_B_norm` を指定してそのまま IC を計算する。

#### 評価②：M_n 空間への逆変換後の IC（ベースラインとの直接比較）

```python
# 1. pooled_b の予測値（Pred_B）を日付×Butterfly_Index のピボットに変換
pred_pivot = results_b.pivot(index='Date', columns='Butterfly_Index', values='Pred_B')
# shape: (N_dates, 8)

# 2. バタフライ→スプレッドの逆変換
pred_spread = np.array([butterfly_to_spread(row) for row in pred_pivot.values])
# shape: (N_dates, 8)

# 3. M_n 空間でのIC計算
# Actual は pool_boj_data の Target_{h}d（元の M_n 変化量）
# → 逆変換した pred_spread と突き合わせて IC を計算
```

逆変換後の IC が Baseline（`pool_boj_data` + M_n 目的変数のモデル）の IC と直接比較できる。

### 実験ノートブック構成の指定

第2フェーズ実験は `notebooks/08_advanced_feature_experiments.ipynb` として新規作成すること（`.ipynb` は Python `json` ライブラリ経由で生成）。

#### セル構成

| セル | 内容 |
|------|------|
| 1 | セットアップ（import・データ読み込み） |
| 2 | Baseline 再確認（Exp-A〜C 最良構成のIC） |
| 3 | Exp-D（Absolute_Meeting_ID）実行・IC比較 |
| 4 | Exp-E（Days_since_first_seen）実行・IC比較 |
| 5 | Exp-DE（D+E）実行・IC比較 + SHAP importance確認 |
| 6 | D/E/DE 比較表の出力 |
| 7 | Exp-F（バタフライ目的変数）実行・評価① バタフライ空間IC |
| 8 | Exp-F 評価② 逆変換→M_n空間IC（ベースラインと比較） |
| 9 | 全実験の最終比較表 |

---

## 評価基準（第2フェーズ共通）

| 判定 | 条件 |
|------|------|
| **採用** | OOS IC が +0.01 以上改善かつ全フォールドで安定 |
| **要検討** | +0.005〜+0.01 の改善 |
| **不採用** | 改善なし または 悪化 |

---

## Gemini CLI への報告内容（第2フェーズ）

実験完了後に以下を報告すること：

1. **Exp-D/E/DE 比較表**：Baseline との Delta IC（3d・5d）
2. **Exp-DE の SHAP**：`Absolute_Meeting_ID` と `Days_since_first_seen` それぞれの Gain importance の値と、Baseline に対して他の特徴量の順位が変わったか
3. **Exp-E の独立性確認**：`Days_since_first_seen` と `Days_to_MPM` の Pearson 相関係数を報告
4. **Exp-F 評価①**：バタフライ空間での IC（3d・5d）
5. **Exp-F 評価②**：逆変換後の M_n 空間での IC（3d・5d）と Baseline との比較
6. 採用推奨の実験（上記評価基準に基づく）

---

## 注意事項（第2フェーズ）

- **Absolute_Meeting_ID の外挿リスク**：walk-forward 検証では、テスト期間の会合番号が学習データに存在しない場合がある。LightGBM はカテゴリ変数の未知値を特別扱いするため実害は小さいが、フォールドごとの IC の安定性に注意すること
- **バタフライ逆変換の数値誤差**：`np.linalg.solve` の解は浮動小数点誤差を含む。逆変換後に元の M_n 変化量と最大誤差（`np.abs(reconstructed - original).max()`）を確認し、`1e-8` 以下であることを確かめること
- **pool_boj_butterfly の独立性**：既存の `pool_boj_data` を変更しないこと。バタフライ実験は完全に独立した関数として実装する

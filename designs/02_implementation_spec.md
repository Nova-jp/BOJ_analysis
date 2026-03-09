# 実装仕様書 02：Gemini CLI向け実装指示

設計の詳細は `designs/01_model_design.md` を先に読むこと。

---

## タスク概要

`src/` 配下のパイプラインを設計書に基づいて再構築し、`notebooks/` に分析ノートブックを作成する。

---

## Task 1：`src/processing.py` の更新

既存の `load_and_clean_data(excel_path, meeting_csv_path)` を以下の仕様に合わせて確認・修正する。

### 出力DataFrameに必要な列

```
Date                    : datetime
M1 〜 M8               : float（MICE補完済み）
USDJPY, JGB_Future,
Nikkei225, DXY         : float（MICE補完済み）
Actual_Policy_Rate     : float（政策金利を前方補完）
Is_Meeting_Day         : int（0 or 1）
M1_is_imputed 〜
M8_is_imputed          : int（0 or 1、MICE補完前の欠損フラグ）
```

現行コードで上記が揃っていれば変更不要。

---

## Task 2：`src/features.py` の全面書き直し

`generate_features(df, d=0.4, window=50)` を以下の仕様で実装する。

### 入力

`load_and_clean_data` の出力DataFrame（横持ち）

### 処理

```python
# 1. M{n}_spread = M{n} - Actual_Policy_Rate  (n=1〜8)
# 2. M{n}_frac_diff = frac_diff(M{n}_spread, d=d, window=window)  (n=1〜8)
# 3. 外部指標の分数階差
#    USDJPY_frac_diff, JGB_Future_frac_diff, Nikkei225_frac_diff
# 4. Days_to_MPM：次回MPM（Is_Meeting_Day=1の将来の日付）までの営業日数
#    注意：MPM当日はDays_to_MPM=0とする
```

### 分数階差の実装

```python
def frac_diff(series, d, window=50):
    """
    分数階差。重みは w_k = (-1)^k * C(d, k)
    windowサイズ分の過去データが揃わない先頭行はNaNとする
    """
```

### 出力

入力DataFrameに上記の列を追加して返す（横持ちのまま）。

---

## Task 3：`src/pooling.py` の全面書き直し

`pool_boj_data(df)` を以下の仕様で実装する。

### 入力

`generate_features` の出力DataFrame（横持ち）

### 処理

```python
# 1. 横持ち → 縦持ちに変換（melt）
#    各行 = (Date, Meeting_Index, その限月の値)

# 2. 全限月の特徴量を全行に結合
#    各行に M1_spread〜M8_spread, M1_frac_diff〜M8_frac_diff を全て持たせる

# 3. 目的変数の生成（Meeting_Indexグループ内でshift）
#    Target_1d = M{n}_{t+1} - M{n}_t
#    Target_3d = M{n}_{t+3} - M{n}_t
#    Target_5d = M{n}_{t+5} - M{n}_t

# 4. MPM当日・直後5日間の除外フラグを追加
#    is_post_mpm = 1 if 過去5日以内にIs_Meeting_Day=1があった日
```

### 出力DataFrameの列

```
Date, Meeting_Index
M1_spread 〜 M8_spread       （水準特徴量・全行共通）
M1_frac_diff 〜 M8_frac_diff （分数階差特徴量・全行共通）
M1_is_imputed 〜 M8_is_imputed
USDJPY_frac_diff, JGB_Future_frac_diff, Nikkei225_frac_diff
Days_to_MPM, Actual_Policy_Rate
is_post_mpm                  （除外フラグ）
Target_1d, Target_3d, Target_5d
```

---

## Task 4：分析ノートブック `notebooks/01_eda.ipynb` の作成

### 目的

データの基本的な性質を確認し、モデル設計の前提を検証する。

### セル構成

1. **データ読み込み**：`load_and_clean_data` → `generate_features` → `pool_boj_data` を実行

2. **M{n}スプレッドの時系列プロット**：M1〜M8スプレッドを1枚のグラフに重ねて描画。MPM日付を縦線で表示

3. **分数階差の確認**：M1_spread と M1_frac_diff を並べてプロット。ADF検定の結果（p値）を表示

4. **目的変数の分布**：Target_1d・Target_3d・Target_5d のヒストグラム。限月別・レジーム別（政策金利水準で色分け）に確認

5. **欠損・補完状況**：M{n}_is_imputedの時系列ヒートマップ

### ノートブック作成の注意

`.ipynb` を新規作成する際はPythonの `json` ライブラリ経由で生成すること（直接 write_file で書かない）。

---

## 確認事項

実装後に以下を報告すること：

- ADF検定結果（各M{n}_frac_diffのp値）→ d=0.4で定常化できているか
- Target_1d / Target_3d / Target_5d のサンプル数（is_post_mpm除外後）
- 各Targetの平均・標準偏差・最大最小値

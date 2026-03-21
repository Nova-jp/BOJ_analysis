# RV Curve モデル仕様

最終更新：2026-03-21（初版）

**新規会話でこのモデルの実験を始める際は必ずこのファイルを確認すること。**

---

## モデル概要

| 項目 | 内容 |
|------|------|
| 予測対象 | 隣接差分 C{n} = M{n}_spread − M{n+1}_spread の h 日後変化 |
| 経済的意味 | BOJ OIS カーブのスティープナー/フラットナー予測 |
| プーリング | C1〜C7（7ペア）を縦積みして 1 モデルで学習 |
| 学習アルゴリズム | LightGBM GBDT（outright と同設定） |
| 予測ホライズン | 3日・5日 |
| 検証方法 | walk-forward（拡張窓、purge=5日、test_window=90日） |
| **主評価指標** | **CS IC**（各日付で C1〜C7 をランク付け、日付平均） |
| 補助指標 | Global IC（outright との比較用）、TS IC（診断用） |
| テスト開始日 | 2024-01-01 |

---

## パイプライン

```python
from src.processing import load_and_clean_data
from src.features_rv import generate_rv_features
from src.pooling_curve import pool_curve_data
from src.modeling import walk_forward_validation, summarize_ic

df_raw    = load_and_clean_data('../data/BOJ_data.xlsx', '../data/BOJ_meeting_history.csv')
df_feat   = generate_rv_features(df_raw)
df_pooled = pool_curve_data(df_feat)

res_3d = walk_forward_validation(df_pooled, 'Target_3d_norm', '2024-01-01')
res_5d = walk_forward_validation(df_pooled, 'Target_5d_norm', '2024-01-01')

# CS IC: Pair_Index 1〜7 全て対象（T12/T18/T24 は含まれないため）
ic_3d = summarize_ic(res_3d, instrument_indices=set(range(1, 8)))
ic_5d = summarize_ic(res_5d, instrument_indices=set(range(1, 8)))
```

---

## 特徴量一覧

| # | カテゴリ | 特徴量 | 備考 |
|---|---------|--------|------|
| 1 | コンテキスト | `Meeting_Index`（Pair_Index、カテゴリ） | 1=C1(M1-M2), ..., 7=C7(M7-M8) |
| 2 | | `Days_to_MPM` | **次回MPMまでの日数**（全銘柄共通、日付レベルの値） |
| 3 | | `Actual_Policy_Rate` | 現在の政策金利 |
| 4 | カーブ水準 | `Curve_Level` | C{n} の現在水準（mean reversion の起点） |
| 5 | M1 アンカー | `M1_spread` | 金利水準のコンテキスト（カーブ全体の位置） |
| 6 | | `M1_frac_diff` | M1_spread の分数階差 |
| 7-13 | カーブ分数階差 | `C1_frac_diff`〜`C7_frac_diff` | 各隣接差分の分数階差（d=0.4, window=50） |
| 14-20 | 補完フラグ | `M1_is_imputed`〜`M8_is_imputed` | 各構成銘柄の MICE 補完フラグ |
| 21-24 | 外部指標 | `USDJPY_frac_diff`, `JGB_Future_frac_diff`, `Nikkei225_frac_diff`, `DXY_frac_diff` | |

合計: 24特徴量

**カテゴリ変数**（LightGBMに明示）: `Meeting_Index`

---

## 目的変数

```
Target_{h}d_norm = (C{n}_{t+h} - C{n}_t) / per_pair_std
```

- `per_pair_std`: 各ペアの全期間 std で正規化（異分散補正）
- `Target_{h}d_std`: 逆正規化用（P&L 計算時に使用）

---

## ベースライン IC（2026-03-21 計測、Days_to_MPM バグ修正後）

| ホライズン | Global IC | CS IC | TS IC | Train IC | Gap |
|-----------|----------|-------|-------|----------|-----|
| 3d | 0.222 | **0.202** | 0.224 | 0.469 | 0.246 |
| 5d | 0.278 | **0.249** | 0.286 | 0.525 | 0.248 |

### Outright モデルとの比較

| 指標 | Outright 3d | RV Curve 3d | 解釈 |
|------|------------|-------------|------|
| Global IC | 0.227 | 0.222 | 同程度 |
| **CS IC** | -0.005 | **0.202** | RV 予測力が大幅改善 |
| TS IC | 0.229 | 0.224 | 同程度 |

**CS IC が -0.005 → 0.202 に改善**：カーブ空間への変換により、モデルがM1〜M8間の相対的なランク付けを学習できるようになった。

---

## Days_to_MPM の設計根拠

処理の単純性と一貫性のため、全ペア共通で「次回MPMまでの日数」（入力DataFrame由来）を使用する。
C{n} の後ろ脚会合までの日数に変換するアプローチも検討したが、
同じ日付のデータでは全銘柄が同一の次回MPMまでの日数を持つため、
日付レベルの共通値をそのまま使うことでシンプルかつ正確な実装になる。

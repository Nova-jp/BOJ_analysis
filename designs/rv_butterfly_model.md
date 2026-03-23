# RV Butterfly モデル仕様

最終更新：2026-03-23（Model B昇格: post-MPM除外廃止）

**新規会話でこのモデルの実験を始める際は必ずこのファイルを確認すること。**

---

## モデル概要

| 項目 | 内容 |
|------|------|
| 予測対象 | バタフライ B{n} = 2*M{n}_spread − M{n-1}_spread − M{n+1}_spread の h 日後変化 |
| 経済的意味 | カーブの曲率変化（M{n} のリッチ/チープ）予測 |
| プーリング | B2〜B7（6系列）を縦積みして 1 モデルで学習 |
| 学習アルゴリズム | LightGBM GBDT（outright と同設定） |
| 予測ホライズン | 3日・5日 |
| 検証方法 | walk-forward（拡張窓、purge=5日、test_window=90日） |
| **post-MPM除外** | **なし**（MPM直後5日間も訓練・テストに含める） |
| **主評価指標** | **CS IC**（各日付で B2〜B7 をランク付け、日付平均） |
| 補助指標 | Global IC（outright との比較用）、TS IC（診断用） |
| テスト開始日 | 2024-01-01 |

---

## パイプライン

```python
from src.processing import load_and_clean_data
from src.features_rv import generate_rv_features
from src.pooling_butterfly import pool_butterfly_data
from src.modeling import walk_forward_validation, summarize_ic

EXCEL_PATH   = '../data/BOJ_data.xlsx'
MEETING_PATH = '../data/BOJ_meeting_history.csv'

df_raw    = load_and_clean_data(EXCEL_PATH, MEETING_PATH)
df_feat   = generate_rv_features(df_raw)
df_pooled = pool_butterfly_data(df_feat)

res_3d = walk_forward_validation(df_pooled, 'Target_3d_norm', '2024-01-01')
res_5d = walk_forward_validation(df_pooled, 'Target_5d_norm', '2024-01-01')

# CS IC: Butterfly_Index 2〜7 全て対象
ic_3d = summarize_ic(res_3d, instrument_indices=set(range(2, 8)))
ic_5d = summarize_ic(res_5d, instrument_indices=set(range(2, 8)))
```

---

## 特徴量一覧

| # | カテゴリ | 特徴量 | 備考 |
|---|---------|--------|------|
| 1 | コンテキスト | `Meeting_Index`（Butterfly_Index、カテゴリ） | 2=B2(2M2-M1-M3), ..., 7=B7(2M7-M6-M8) |
| 2 | | `Days_to_MPM` | **次回MPMまでの日数**（全銘柄共通、日付レベルの値） |
| 3 | | `Actual_Policy_Rate` | 現在の政策金利 |
| 4 | バタフライ水準 | `Fly_Level` | B{n} の現在水準（mean reversion の起点） |
| 5 | M1 アンカー | `M1_spread` | 金利水準のコンテキスト |
| 6 | | `M1_frac_diff` | M1_spread の分数階差 |
| 7 | M1-M8 スロープ | `Slope_M1M8` | カーブのスロープ水準（バタフライはスロープと連動する） |
| 8 | | `Slope_M1M8_frac_diff` | スロープの分数階差 |
| 9-14 | バタフライ分数階差 | `B2_frac_diff`〜`B7_frac_diff` | 各バタフライの分数階差（d=0.4, window=50） |
| 15-22 | 補完フラグ | `M1_is_imputed`〜`M8_is_imputed` | 各構成銘柄の MICE 補完フラグ |
| 23-26 | 外部指標 | `USDJPY_frac_diff`, `JGB_Future_frac_diff`, `Nikkei225_frac_diff`, `DXY_frac_diff` | |

合計: 26特徴量

**カテゴリ変数**（LightGBMに明示）: `Meeting_Index`

---

## 目的変数

```
Target_{h}d_norm = (B{n}_{t+h} - B{n}_t) / per_butterfly_std
```

- `per_butterfly_std`: 各バタフライ系列の全期間 std で正規化
- `Target_{h}d_std`: 逆正規化用

---

## ベースライン IC（2026-03-23 更新、post-MPM除外なし=Model B）

| ホライズン | Global IC | CS IC | TS IC | Train IC | Gap |
|-----------|----------|-------|-------|----------|-----|
| 3d | 0.378 | **0.296** | — | 0.509 | 0.132 |
| 5d | 0.434 | **0.355** | — | 0.564 | 0.130 |

### 旧ベースライン IC（Model A: post-MPM除外あり、2026-03-21計測）

| ホライズン | Global IC | CS IC | Train IC | Gap |
|-----------|----------|-------|----------|-----|
| 3d | 0.373 | 0.304 | 0.521 | 0.148 |
| 5d | 0.418 | 0.346 | 0.573 | 0.154 |

### Model A→B 変更の根拠（2026-03-23、Exp: exp_post_mpm_diagnostic.ipynb）

- post-MPM 5日間は予想外に高いCSICを示した（Fold≥5: Post合計 3d=0.367, 5d=0.373）
- D+0（会合当日）・D+1が特に高IC → 会合直後のバタフライ歪みの急速な解消をモデルが捉えている
- 最弱ゾーンは **6-10日前（3d CS IC ≈ 0.05）**で、会合中間地点がランダムウォーク的
- 実用上、分析空白期間をなくすことが目的の一つ
- 3d CS ICは微減（0.304→0.296）だが、5d CS ICは改善（0.346→0.355）

### Outright・RV Curve との比較（参考、旧Model A値）

| 指標 | Outright 3d | RV Curve 3d | RV Butterfly 3d |
|------|------------|-------------|-----------------|
| Global IC | 0.227 | 0.222 | **0.373** |
| **CS IC** | -0.005 | 0.202 | **0.304** |
| Train IC | 0.519 | 0.469 | 0.521 |
| Gap | 0.292 | 0.246 | **0.148** |

---

## Days_to_MPM の設計根拠

処理の単純性と一貫性のため、全系列共通で「次回MPMまでの日数」（入力DataFrame由来）を使用する。
同じ日付のデータでは全銘柄が同一の次回MPMまでの日数を持つため、
日付レベルの共通値をそのまま使うことでシンプルかつ正確な実装になる。

---

## バタフライの分散に関する注記

事前の懸念とは異なり、バタフライの 3d 変化 std（0.91〜1.14bp）は
隣接差分（0.63〜0.90bp）を**上回る**。予測対象として十分な信号強度がある。
Exp-F（不採用）との違いは特徴量も同空間（バタフライ空間）に変換したことで、
特徴量と目的変数の空間的整合性が確保されている。

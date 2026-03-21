# 現在のメインモデル仕様

最終更新：2026-03-21（BOJ会合CSV修正・ベースライン再計測）

**新規会話でのスタート時はこのファイルを必ず確認すること。**

---

## モデル概要

| 項目 | 内容 |
|------|------|
| アルゴリズム | LightGBM GBDT（regression, RMSE） |
| プーリング | M1〜M8を縦積み（1モデルで全限月を学習） |
| 予測ホライズン | 3日・5日（1日は精度不足で不採用） |
| 検証方法 | walk-forward（拡張窓、purge=5日、test_window=90日） |
| 評価指標 | OOS IC（Spearman相関） |
| 学習データ開始 | 全期間（MICE補完後の先頭50行はNaNで自動除外） |
| テスト開始日 | 2024-01-01（固定） |

## 特徴量一覧（現行42特徴量）

| # | カテゴリ | 特徴量 | 実装場所 | 備考 |
|---|---------|--------|---------|------|
| 1 | コンテキスト | `Meeting_Index` | `pooling.py` | カテゴリ変数。1〜8=BOJ会合先、10/11/12=T12/T18/T24 |
| 2 | | `Is_Tenor_OIS` | `pooling.py` | 0=M1-M8、1=T12/T18/T24 |
| 3 | | `Days_to_MPM` | `processing.py` | 次回会合までの日数（0〜65） |
| 4 | | `Actual_Policy_Rate` | `processing.py` | 現在の政策金利（ffill+bfill） |
| 5-12 | BOJ スプレッド水準 | `M1_spread`〜`M8_spread` | `features.py` | M{n} − Actual_Policy_Rate |
| 13-20 | BOJ 分数階差 | `M1_frac_diff`〜`M8_frac_diff` | `features.py` | スプレッドの分数階差（d=0.4, window=50） |
| 21-28 | BOJ 補完フラグ | `M1_is_imputed`〜`M8_is_imputed` | `processing.py` | MICE補完された行を示すバイナリ |
| 29 | 外部指標 | `USDJPY_frac_diff` | `features.py` | ドル円の分数階差 |
| 30 | | `JGB_Future_frac_diff` | `features.py` | JGB先物の分数階差 |
| 31 | | `Nikkei225_frac_diff` | `features.py` | 日経225の分数階差 |
| 32-34 | テナーOIS スプレッド | `T12_spread`, `T18_spread`, `T24_spread` | `features.py` | 12m/18m/24m OIS − 政策金利 |
| 35-37 | テナーOIS 分数階差 | `T12_frac_diff`, `T18_frac_diff`, `T24_frac_diff` | `features.py` | 同上の分数階差 |
| 38-40 | テナーOIS 補完フラグ | `T12_is_imputed`, `T18_is_imputed`, `T24_is_imputed` | `processing.py` | 同上 |
| 41 | 会合識別子 | `Absolute_Meeting_ID` | `pooling.py` | カテゴリ変数。データ開始から通し番号（Exp-D） |
| 42 | | `Days_since_first_seen` | `pooling.py` | その会合の初観測からの日数（Exp-E） |

**カテゴリ変数**（LightGBMに明示）: `Meeting_Index`, `Is_Tenor_OIS`, `Absolute_Meeting_ID`

**実験候補**（未採用）: 曜日 sin/cos (Exp-A)、Days_to_MPM sin/cos (Exp-B)、カーブ形状 (Exp-C)

## OOS IC（テスト期間 2024-01-01〜、rounds=100）

### IC の種類（2026-03-21 追加）

| 指標 | 説明 | 用途 |
|------|------|------|
| **CS IC**（主指標） | 各日付で M1〜M8 をランク付け → 日付平均 | RV 戦略の直接評価。トレンドの影響なし |
| **Global IC**（参照） | 全 (日付×銘柄) の Spearman 相関。時系列+横断面の混合 | 他研究との比較用 |
| **TS IC**（診断） | 各銘柄の時系列 IC → 銘柄平均。トレンド相場で高くなりやすい | Global - CS の差でベータ混入を診断 |

### 現行ベースライン（2026-03-21 再計測、DXY追加後）

| ホライズン | Global IC | CS IC | TS IC | Train IC | Gap |
|-----------|----------|-------|-------|----------|-----|
| 3d | 0.227 | **-0.005** | 0.229 | 0.519 | 0.292 |
| 5d | 0.228 | **0.074** | 0.234 | 0.564 | 0.336 |

**フォールド別 Global IC（3d）**: {0: 0.257, 1: -0.051, 2: 0.166, 3: 0.163, 4: 0.266, 5: 0.118, 6: 0.332, 7: 0.426, 8: 0.552}
**フォールド別 Global IC（5d）**: {0: 0.412, 1: 0.040, 2: -0.193, 3: 0.318, 4: 0.313, 5: -0.051, 6: 0.423, 7: 0.528, 8: 0.638}

**重要な発見（2026-03-21）**: CS IC ≈ 0（3d: -0.005、5d: 0.074）に対して Global IC ≈ 0.23。
Global IC のほぼ全てが TS IC（時系列方向性）から来ており、横断面的な相対価値予測力はほぼゼロ。
詳細は下記「CS IC の発見と今後の方針」を参照。

- rounds=100：感度分析で最適と確認（`designs/archives/06_rounds_sensitivity.md`）
- 直近3フォールド（Fold 6-8 = 2025年以降、利上げサイクル）は IC が高い
- 初期フォールド（Fold 0-2 = NIRP解除直後）は分布シフトで IC が低い

**注意**: 旧ノートブック(03_model_analysis_executed)の「0.39/0.63」は
最終フォールドのみのICであり、全OOS ICではない。

## データ

| ファイル | 内容 | 期間 |
|---------|------|------|
| `data/BOJ_data.xlsx` | Reuters OISスワップ・外部指標 | 2007-03-05〜 |
| `data/BOJ_meeting_history.csv` | MPM日程・政策金利 | 2007-01-18〜（217会合） |

## LightGBMハイパーパラメータ

```python
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 30,
    'lambda_l2': 1.0,
    'num_boost_round': 100,  # 感度分析（06_rounds_sensitivity）で100が最適と判明
}
categorical_feature = ['Meeting_Index', 'Is_Tenor_OIS', 'Absolute_Meeting_ID']
```

## 実装ファイル

- `src/processing.py` : `load_and_clean_data(excel_path, meeting_csv_path)`
- `src/features.py` : `generate_features(df, d=0.4, window=50)`
- `src/pooling.py` : `pool_boj_data(df)`
- `src/modeling.py` : `walk_forward_validation(df, target_col, start_date)`

## 標準的な実行コード

```python
from src.processing import load_and_clean_data
from src.features import generate_features
from src.pooling import pool_boj_data
from src.modeling import walk_forward_validation, calculate_metrics

df_raw    = load_and_clean_data('../data/BOJ_data.xlsx', '../data/BOJ_meeting_history.csv')
df_feat   = generate_features(df_raw)
df_pooled = pool_boj_data(df_feat)

res_3d = walk_forward_validation(df_pooled, 'Target_3d_norm', '2024-01-01')
res_5d = walk_forward_validation(df_pooled, 'Target_5d_norm', '2024-01-01')
```

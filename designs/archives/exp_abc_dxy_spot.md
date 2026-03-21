# 実験設計書: Exp-A/B/C/DXY/Spot 比較実験

作成日: 2026-03-21
担当: Gemini CLI
ノートブック: `notebooks/exp_feature_comparison.ipynb`

---

## 目的

以下の5実験を1ノートブックで実施し、ベースラインとのIC差分を比較する。

---

## ベースライン（セル1で必ず再現すること）

```python
from src.processing import load_and_clean_data
from src.features import generate_features
from src.pooling import pool_boj_data
from src.modeling import walk_forward_validation, summarize_ic

df_raw    = load_and_clean_data('../data/BOJ_data.xlsx', '../data/BOJ_meeting_history.csv')
df_feat   = generate_features(df_raw)  # デフォルト引数（全フラグFalse）
df_pooled = pool_boj_data(df_feat)

res_3d_base = walk_forward_validation(df_pooled, 'Target_3d_norm', '2024-01-01')
res_5d_base = walk_forward_validation(df_pooled, 'Target_5d_norm', '2024-01-01')
ic_base = summarize_ic(res_3d_base)
# 期待値: 3d OOS IC ≒ 0.224, 5d ≒ 0.228（一致しなければ環境を確認）
```

**注意**: `generate_features()` のデフォルト引数は
- `add_weekday_cyclic=False`
- `add_days_to_mpm_cyclic=False`
- `add_curve_features=False`

なのでベースラインには Exp-A/B/C の特徴量は含まれていない。

---

## 各実験の実装

### Exp-A: 曜日の円環エンコーディング

```python
df_feat_a = generate_features(df_raw, add_weekday_cyclic=True)
df_pooled_a = pool_boj_data(df_feat_a)
res_3d_a = walk_forward_validation(df_pooled_a, 'Target_3d_norm', '2024-01-01')
res_5d_a = walk_forward_validation(df_pooled_a, 'Target_5d_norm', '2024-01-01')
```

追加される特徴量: `DayOfWeek_sin`, `DayOfWeek_cos`（+2特徴量）

---

### Exp-B: Days_to_MPM の円環エンコーディング

```python
df_feat_b = generate_features(df_raw, add_days_to_mpm_cyclic=True)
df_pooled_b = pool_boj_data(df_feat_b)
res_3d_b = walk_forward_validation(df_pooled_b, 'Target_3d_norm', '2024-01-01')
res_5d_b = walk_forward_validation(df_pooled_b, 'Target_5d_norm', '2024-01-01')
```

追加される特徴量: `Days_to_MPM_sin`, `Days_to_MPM_cos`（+2特徴量）
**実装メモ**: `features.py` では CYCLE=45 で正規化している。BOJの会合間隔は概ね28〜70日（中央値45日程度）。

---

### Exp-C: カーブ形状特徴量

```python
df_feat_c = generate_features(df_raw, add_curve_features=True)
df_pooled_c = pool_boj_data(df_feat_c)
res_3d_c = walk_forward_validation(df_pooled_c, 'Target_3d_norm', '2024-01-01')
res_5d_c = walk_forward_validation(df_pooled_c, 'Target_5d_norm', '2024-01-01')
```

追加される特徴量: `Curve_Slope`, `Butterfly_M2`〜`Butterfly_M7`, 各`_frac_diff`（計14特徴量）

- `Curve_Slope = M8_spread - M1_spread`
- `Butterfly_Mn = 2*Mn_spread - M(n-1)_spread - M(n+1)_spread`

---

### Exp-DXY: DXY_frac_diff の追加

**背景**: DXYは `processing.py` でロード・MICE補完済みだが、`features.py` の `ext_cols` リストに含まれていない見落とし。現行ベースラインではDXYは一切特徴量として使われていない。

このノートブック内でのみ `generate_features` をパッチして試す（src/は変更しない）:

```python
from src.features import frac_diff

def generate_features_with_dxy(df, d=0.4, window=50):
    """DXY_frac_diffを追加したgenerate_features"""
    feat_df = generate_features(df, d=d, window=window)  # ベースライン特徴量
    # DXY_frac_diff を追加
    if 'DXY' in feat_df.columns:
        feat_df['DXY_frac_diff'] = frac_diff(feat_df['DXY'], d=d, window=window)
    return feat_df

df_feat_dxy = generate_features_with_dxy(df_raw)
df_pooled_dxy = pool_boj_data(df_feat_dxy)
# pool_boj_data は final_cols に存在する列のみを選択するため、
# DXY_frac_diff が pooled に残っているか確認すること。
# → 残っていない場合は pooling.py の final_cols に追加が必要（Claudeに相談）

res_3d_dxy = walk_forward_validation(df_pooled_dxy, 'Target_3d_norm', '2024-01-01')
res_5d_dxy = walk_forward_validation(df_pooled_dxy, 'Target_5d_norm', '2024-01-01')
```

**実装上の注意**: `pool_boj_data` の `final_cols`（pooling.py line 147-155）に `DXY_frac_diff` が含まれていないため、特徴量が落ちる。その場合は以下の方法で確認:

```python
# DXY_frac_diff が pooled に含まれているか確認
print('DXY_frac_diff' in df_pooled_dxy.columns)
# Falseなら → pool_boj_data の呼び出し後に手動で結合する
df_pooled_dxy2 = df_pooled_dxy.copy()
# 日付でマージしてDXY_frac_diffを補完
dxy_map = df_feat_dxy[['Date', 'DXY_frac_diff']].drop_duplicates()
df_pooled_dxy2 = df_pooled_dxy2.merge(dxy_map, on='Date', how='left')
```

---

### Exp-Spot: スポット金利（T12/T18/T24）の寄与確認

**背景**: T12/T18/T24は現行ベースラインに既に組み込まれているが、「入れる/入れない」の比較が未実施。

T12/T18/T24を除外するには、プーリング前に列を落とす:

```python
def generate_features_no_spot(df, d=0.4, window=50):
    """T12/T18/T24を除外したバージョン"""
    # T12/T18/T24 を drop
    df_no_spot = df.drop(columns=[c for c in ['T12', 'T18', 'T24'] if c in df.columns])
    return generate_features(df_no_spot, d=d, window=window)

df_feat_no_spot = generate_features_no_spot(df_raw)
df_pooled_no_spot = pool_boj_data(df_feat_no_spot)
# この場合 pool_boj_data は T12/T18/T24 なしで動作する（Is_Tenor_OIS=1 の行が生成されない）

res_3d_no_spot = walk_forward_validation(df_pooled_no_spot, 'Target_3d_norm', '2024-01-01')
res_5d_no_spot = walk_forward_validation(df_pooled_no_spot, 'Target_5d_norm', '2024-01-01')
```

---

## 結果まとめ（最後のセルで表として出力）

```python
import pandas as pd

results = {
    'Baseline (42feat)': {
        '3d_ic_all': summarize_ic(res_3d_base)['ic_all'],
        '3d_ic_recent': summarize_ic(res_3d_base)['ic_recent'],
        '5d_ic_all': summarize_ic(res_5d_base)['ic_all'],
        '5d_ic_recent': summarize_ic(res_5d_base)['ic_recent'],
    },
    'Exp-A (+weekday sin/cos)': {
        '3d_ic_all': summarize_ic(res_3d_a)['ic_all'],
        '3d_ic_recent': summarize_ic(res_3d_a)['ic_recent'],
        '5d_ic_all': summarize_ic(res_5d_a)['ic_all'],
        '5d_ic_recent': summarize_ic(res_5d_a)['ic_recent'],
    },
    'Exp-B (+Days_to_MPM sin/cos)': {
        '3d_ic_all': summarize_ic(res_3d_b)['ic_all'],
        '3d_ic_recent': summarize_ic(res_3d_b)['ic_recent'],
        '5d_ic_all': summarize_ic(res_5d_b)['ic_all'],
        '5d_ic_recent': summarize_ic(res_5d_b)['ic_recent'],
    },
    'Exp-C (+curve features)': {
        '3d_ic_all': summarize_ic(res_3d_c)['ic_all'],
        '3d_ic_recent': summarize_ic(res_3d_c)['ic_recent'],
        '5d_ic_all': summarize_ic(res_5d_c)['ic_all'],
        '5d_ic_recent': summarize_ic(res_5d_c)['ic_recent'],
    },
    'Exp-DXY (+DXY_frac_diff)': {
        '3d_ic_all': summarize_ic(res_3d_dxy)['ic_all'],
        '3d_ic_recent': summarize_ic(res_3d_dxy)['ic_recent'],
        '5d_ic_all': summarize_ic(res_5d_dxy)['ic_all'],
        '5d_ic_recent': summarize_ic(res_5d_dxy)['ic_recent'],
    },
    'Exp-Spot (no T12/T18/T24)': {
        '3d_ic_all': summarize_ic(res_3d_no_spot)['ic_all'],
        '3d_ic_recent': summarize_ic(res_3d_no_spot)['ic_recent'],
        '5d_ic_all': summarize_ic(res_5d_no_spot)['ic_all'],
        '5d_ic_recent': summarize_ic(res_5d_no_spot)['ic_recent'],
    },
}

df_results = pd.DataFrame(results).T
df_results['3d_delta'] = df_results['3d_ic_all'] - df_results.loc['Baseline (42feat)', '3d_ic_all']
df_results['5d_delta'] = df_results['5d_ic_all'] - df_results.loc['Baseline (42feat)', '5d_ic_all']
print(df_results.round(4).to_string())
```

---

## 採用基準（Claudeがレビュー）

- OOS IC（全体）が +0.01 以上かつ全フォールドで安定 → 採用
- IC が低下 or 不安定 → 不採用
- スポット除外で IC が下がる → T12/T18/T24 は寄与あり（現状維持）
- スポット除外でも IC が変わらない → 除外を検討（特徴量削減によるノイズ低減）

---

## 結果の報告先

実験完了後、以下を Gemini → Claude に報告する:
- 各実験の `3d_ic_all`, `3d_ic_recent`, `5d_ic_all`, `5d_ic_recent`
- DXYが pool_boj_data に渡ったかどうかの確認（`DXY_frac_diff in df_pooled.columns`）
- フォールド別 IC（`ic_by_fold`）に異常値があれば報告

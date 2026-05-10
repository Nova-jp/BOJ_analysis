# 実験設計書: Spot Spread モデル（Exp-SpotSpread）

作成日：2026-03-21
担当：Gemini CLI（実装・実験）
レビュー：Claude（採否判断）

**この設計書を受け取った Gemini は、独立したノートブック `notebooks/exp_spot_spread_model.ipynb` を作成して実験すること。**

---

## 目的

会合付きOISとスポットテナーOIS（1年物）の差分 `S{n} = M{n} - T12` の変化を予測する新しいカーブモデルを検証する。

現行 RV Curve モデル（C{n} = 隣接限月差分）が捉えていない「会合付きOIS vs 連続テナー曲線」の相対的な割高/割安を予測できるか確かめる。

---

## 経済的直観

- `T12`（12ヶ月スポットOIS）はインターポレートされた連続テナー曲線上の1点
- `M{n}`（会合付きOIS）は特定会合の政策金利期待を直接反映
- **差分 `S{n} = M{n} - T12` はその会合が連続テナー曲線に対して割高/割安かを示す**

`S{n}` が大きい → 当該会合は T12 より上に乗っている（タイト）
`S{n}` が小さい → 当該会合は T12 より下に乗っている（チープ）

このスプレッドの平均回帰・モメンタムを予測することで、
隣接差分（C{n}）とは独立した相対価値シグナルが得られる可能性がある。

---

## 目的変数

```
S{n} = M{n} - T12   （= M{n}_spread - T12_spread、政策金利は相殺される）

Target_{h}d      = S{n}_{t+h} - S{n}_t
Target_{h}d_norm = Target_{h}d / per_series_std   （系列 n ごとの全期間 std で正規化）
```

プーリング対象: S1〜S8（8系列）

---

## 特徴量一覧

| # | カテゴリ | 特徴量 | 備考 |
|---|---------|--------|------|
| 1 | コンテキスト | `Meeting_Index`（カテゴリ） | 1〜8（S{n} の n に対応） |
| 2 | | `Days_to_MPM` | 次回MPMまでの日数 |
| 3 | | `Actual_Policy_Rate` | 現在の政策金利 |
| 4 | スプレッド水準 | `Spread_Level` | S{n}_t の現在水準（mean reversion 起点） |
| 5 | スプレッド分数階差 | `Spread_frac_diff` | S{n} の分数階差（d=0.4, window=50） |
| 6 | T12文脈 | `T12_spread` | T12 - Actual_Policy_Rate（T12 水準のコンテキスト） |
| 7 | | `T12_frac_diff` | T12_spread の分数階差 |
| 8 | | `T18_frac_diff` | T18_spread の分数階差（スポットカーブ高次モメンタム） |
| 9 | | `T24_frac_diff` | T24_spread の分数階差 |
| 10 | 補完フラグ | `Instrument_is_imputed` | 当該 M{n} の MICE 補完フラグ |
| 11 | | `T12_is_imputed` | T12 の MICE 補完フラグ |
| 12-15 | 外部指標 | `USDJPY_frac_diff`, `JGB_Future_frac_diff`, `Nikkei225_frac_diff`, `DXY_frac_diff` | |

合計: 15特徴量（`Meeting_Index` はカテゴリ変数としてLightGBMに明示）

> T18/T24 の絶対水準は今回含めない。効果を単離したい場合は第2弾実験で追加する。

---

## パイプライン

`src/` モジュールを使って実装する。

```python
import sys
sys.path.insert(0, '..')

from src.processing import load_and_clean_data
from src.features_rv import generate_rv_features
from src.pooling_spot_spread import pool_spot_spread_data
from src.modeling import walk_forward_validation, summarize_ic

df_raw    = load_and_clean_data('../data/BOJ_data.xlsx', '../data/BOJ_meeting_history.csv')
df_feat   = generate_rv_features(df_raw)
df_pooled = pool_spot_spread_data(df_feat)

res_3d = walk_forward_validation(df_pooled, 'Target_3d_norm', '2024-01-01')
res_5d = walk_forward_validation(df_pooled, 'Target_5d_norm', '2024-01-01')

# CS IC: Meeting_Index 1〜8 全て対象
ic_3d = summarize_ic(res_3d, instrument_indices=set(range(1, 9)))
ic_5d = summarize_ic(res_5d, instrument_indices=set(range(1, 9)))

print("=== Spot Spread モデル ===")
for label, ic in [('3d', ic_3d), ('5d', ic_5d)]:
    print(f"[{label}] Global IC={ic['ic_all']:.4f}  CS IC={ic['cs_ic_all']:.4f}"
          f"  TS IC={ic['ts_ic_all']:.4f}  Train IC={ic['train_ic']:.4f}"
          f"  Gap={ic['gap']:.4f}")
```

---

## ベースライン比較

| モデル | CS IC 3d | CS IC 5d |
|--------|----------|----------|
| RV Curve | 0.202 | 0.249 |
| **RV Butterfly ★** | **0.304** | **0.346** |
| Spot Spread（今回） | ? | ? |

目標: CS IC 3d > 0.21（RV Curve を上回ること）

---

## 採用基準

- CS IC 3d ≥ 0.21 かつ CS IC 5d ≥ 0.26
- フォールド別 CS IC が全体的に安定していること（特定フォールドのみ高い状態でないこと）

---

## 報告フォーマット（Gemini → Claude）

```
=== Exp-SpotSpread 結果 ===
3d: Global IC=X.XXX, CS IC=X.XXX, TS IC=X.XXX, Train IC=X.XXX, Gap=X.XXX
5d: Global IC=X.XXX, CS IC=X.XXX, TS IC=X.XXX, Train IC=X.XXX, Gap=X.XXX

フォールド別 CS IC (3d): [F0=..., F1=..., ...]
フォールド別 CS IC (5d): [F0=..., F1=..., ...]

特徴量重要度 Top5（任意）:
観察（気づいた点）:
```

---

## 将来の拡張候補（本実験が採用された場合）

1. **T18/T24 絶対水準の追加**: `T18_spread`, `T24_spread`（スポットカーブスロープ）
2. **T18/T24 を参照点にした変形**: `S{n}_18 = M{n} - T18` など複数テナーとの差分
3. **Butterfly・Curve との合成**: 各モデルの予測シグナルを線形合成

---

## 注意事項

- `start_date='2024-01-01'` で固定（全モデル共通）
- MPM 直後除外: `is_post_mpm == 1` の行は walk_forward_validation 内で自動除外される
- セル 1 で RV Butterfly のベースライン IC（CS IC 3d=0.304, 5d=0.346）を再現してから本実験を実行すること

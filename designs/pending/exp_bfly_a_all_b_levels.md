# 実験設計書: Exp-BFly-A（全バタフライ水準追加）

**作成日**: 2026-03-22
**担当**: Gemini CLI
**対象モデル**: RV Butterfly
**ノートブック**: `notebooks/feature_engineering_base.ipynb` をコピーして実験

---

## 背景・目的

現状の RV Butterfly モデルには特徴量に**非対称性**がある：

| カテゴリ | 現状 | 問題 |
|---------|------|------|
| 水準（Level） | `Fly_Level`（自分の B{n} の水準のみ・1列） | B3〜B7 の水準が B2 の行に入っていない |
| 分数階差 | `B2_frac_diff`〜`B7_frac_diff`（全6列・全行共通） | 水準と非対称 |

設計意図は「各バタフライ B2〜B7 の水準と分数階差を全部特徴量にする」だったが、
水準だけ `Fly_Level`（自分の B{n}のみ）になっていた。

**目的**: B2〜B7 全ての水準を全行に追加した場合に CS IC が改善するか確認する。

---

## ベースライン IC（必ず再現してから実験）

| Horizon | Global IC | CS IC | Train IC | Gap |
|---------|-----------|-------|----------|-----|
| 3d | 0.373 | **0.304** | 0.521 | 0.148 |
| 5d | 0.418 | **0.346** | 0.573 | 0.154 |

設定: `start_date="2024-01-01"`, `rounds=100`
評価: `summarize_ic(results, instrument_indices=set(range(2, 8)))`
**主指標は CS IC**（採用基準: +0.01 以上改善かつ全フォールドで安定）

---

## 実験内容

### 変更点（ノートブック内のみ。`src/` は変更しない）

`pool_butterfly_data()` の代わりに、ノートブック内でインライン実装したプーリングを使う。

**変更箇所 1: melt 前に B2〜B7 水準を列として保持**

```python
# melt 前に B2_level〜B7_level を列として追加
for n in range(2, 8):
    feat_df[f'B{n}_level'] = feat_df[f'B{n}']

# melt（この時点で B{n}_level は id_cols に含まれるため全行に引き継がれる）
fly_raw_cols = [f'B{n}' for n in range(2, 8)]
id_cols = [c for c in feat_df.columns if c not in fly_raw_cols]
pooled = feat_df.melt(
    id_vars=id_cols,
    value_vars=fly_raw_cols,
    var_name='Rate_Label',
    value_name='Rate_Value',
)
```

**変更箇所 2: Fly_Level を削除し、B2_level〜B7_level を追加**

```python
# Fly_Level は使わない（Rate_Value は NON_FEATURE_COLS で除外される）
# pooled['Fly_Level'] = pooled['Rate_Value']  ← この行を削除

# final_cols の変更
b_level_cols = [f'B{n}_level' for n in range(2, 8)]  # ← 追加
# 旧: ['Fly_Level'] を final_cols から削除し、b_level_cols を追加
```

**変更箇所 3: final_cols の修正**

```python
final_cols = (
    basic_cols
    + b_level_cols       # B2_level〜B7_level（6列）← Fly_Level の代替
    + anchor_cols
    + fly_fd_cols
    + imputed_cols_fly
    + ext_fd_cols
    + target_cols
)
```

---

## 実験手順

1. **セル1**: `src/` のコードをそのまま使ってベースライン IC を再現
   → Global IC 3d≈0.373, CS IC 3d≈0.304 であることを確認

2. **セル2以降**: 上記変更を加えたインライン実装で Exp-BFly-A を実行

3. **IC 比較表を出力**:

```python
print("=== Baseline ===")
ic_all_b, ic_recent_b, cs_ic_b, ts_ic_b, train_ic_b, gap_b, by_fold_b = summarize_ic(results_baseline, instrument_indices=set(range(2, 8)))

print("=== Exp-BFly-A (全B水準追加) ===")
ic_all_a, ic_recent_a, cs_ic_a, ts_ic_a, train_ic_a, gap_a, by_fold_a = summarize_ic(results_a, instrument_indices=set(range(2, 8)))
```

4. **フォールド別 IC を棒グラフで可視化**（Baseline vs Exp-A の両方）

---

## 報告フォーマット

実験完了後、以下の表を Claude に共有すること：

| | Baseline | Exp-BFly-A | Δ |
|-|----------|------------|---|
| 3d Global IC | 0.373 | | |
| 3d CS IC | 0.304 | | |
| 3d Train IC | 0.521 | | |
| 3d Gap | 0.148 | | |
| 5d Global IC | 0.418 | | |
| 5d CS IC | 0.346 | | |
| 5d Train IC | 0.573 | | |
| 5d Gap | 0.154 | | |

加えて、フォールド別 IC の棒グラフ画像を共有すること。

---

## 注意事項

- `src/pooling_butterfly.py` は変更しない（ノートブック内のインライン実装で完結させる）
- 比較は同一 walk-forward split（`start_date="2024-01-01"`）で行う
- 特徴量の数が変わる（26特徴量 → `Fly_Level` 削除 + `B2_level`〜`B7_level` 追加 = 31特徴量）ことを確認する

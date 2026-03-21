# 分析タスク仕様書 03：追加分析

設計方針は `designs/01_model_design.md` を参照。

---

## Task A：iteration=1問題の修正（`src/modeling.py`）

### 問題

`walk_forward_validation` 内の `train_model` で early stopping が round 1 で止まるフォールドがある。訓練末尾20%をバリデーションセットにしているため、フォールドによってはゼロ金利期と利上げ期で分布が乖離し、即座にスコアが悪化して止まる。

### 修正内容

以下の2点を `train_model` に適用する：

```python
# 1. early_stopping_rounds を 50 → 100 に増やす
# 2. num_boost_round の実績値に下限を設ける
#    → 少なくとも 30 round は学習する（early stopping で 1 round で止まらないように）
callbacks=[
    lgb.early_stopping(stopping_rounds=100, min_delta=1e-6),
    lgb.log_evaluation(period=0)
]
```

また `walk_forward_validation` 内のバリデーションセット生成を以下に変更する：

```python
# 現行：訓練データの末尾20%
# 変更後：訓練データの末尾から一定日数（60日）をバリデーションに使う
#         ただし訓練データが300日未満の場合は末尾15%を使う
VAL_DAYS = 60
```

修正後、03_model_analysis を再実行して iteration=1 のフォールドが消えたか確認し報告すること。

---

## Task B：会合直前シグナルの逆張り vs 順張り分析（`notebooks/04_mpm_signal_analysis.ipynb`）

### 目的

会合直前（Days_to_MPM ≤ 5）において、モデルが予測する方向に対して順張りと逆張りのどちらが期待リターンが高いかを検証する。

### 分析手順

#### Step 1：会合直前データの抽出

3d・5dモデルの予測結果から Days_to_MPM ≤ 5 の行を抽出。

#### Step 2：4象限の集計

| | 実際：上昇 | 実際：下落 |
|---|---|---|
| **予測：上昇** | 的中（順張り利益） | 外れ（逆張り利益） |
| **予測：下落** | 外れ（逆張り利益） | 的中（順張り利益） |

各象限の**度数・実際の変化幅の平均・標準偏差・中央値**を集計する。

#### Step 3：戦略別の期待リターン計算

```
順張り期待リターン = Σ(的中時の実際変化幅) / 全サンプル数
逆張り期待リターン = Σ(外れ時の実際変化幅の絶対値) / 全サンプル数
```

※ 取引コストは現段階では無視する。

#### Step 4：MPM別の可視化

各MPMイベント（会合）について：
- 会合前5日間の予測方向の推移
- 会合後の実際のレート変化
- 順張り vs 逆張りの仮想PnL累積グラフ

#### Step 5：限月別の比較

M1（次回会合）は直接的な政策変更を反映するため最も大きく動く。M3〜M6との比較も行う。

#### 出力として欲しいもの

- 4象限の集計表（3d・5d別、M1・M3・M5別）
- 順張り vs 逆張りの累積仮想PnLチャート
- MPM回別の予測方向と実際の変化幅の散布図

---

## Task C：特徴量重要度とSHAP分析（`notebooks/05_feature_importance.ipynb`）

### 目的

3d・5dモデルが何を根拠に予測しているかを理解し、今後の特徴量エンジニアリングの方針を立てる。

### 事前準備

```bash
pip install shap
```

### 分析手順

#### Step 1：最終フォールドのモデルを再利用

`walk_forward_validation` を修正して**最終フォールドのモデルオブジェクトを返す**ようにする。あるいは最終フォールドと同じ訓練期間でモデルを単独学習してもよい。

#### Step 2：Gain-based Feature Importance

```python
importance = pd.Series(
    model.feature_importance(importance_type='gain'),
    index=feature_cols
).sort_values(ascending=False)
```

上位20特徴量を横棒グラフで可視化。3d・5dを並べて比較する。

#### Step 3：SHAP Summary Plot

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, max_display=20)
```

#### Step 4：SHAP Dependence Plot（上位3特徴量）

Gain importanceの上位3特徴量について `shap.dependence_plot` を作成する。特徴量値とSHAP値の関係（非線形性・閾値効果）を確認する。

#### 確認したい仮説

| 仮説 | 確認方法 |
|---|---|
| Days_to_MPM が最重要特徴量のひとつ | Gain importance の上位に入るか |
| 自限月（Meeting_Indexに対応するM{n}）より他限月の情報も効いている | M1〜M8の各spread/frac_diffの重要度分布 |
| 外部指標（USDJPY・JGB・日経）は有効か | これらのGain importanceが有意に正か |
| M{n}_is_imputedフラグが重要なら補完の質に問題あり | フラグの重要度が高い場合は注意 |

#### 報告内容

- Gain importance 上位20特徴量（3d・5d）
- SHAP summary plot の画像（スクリーンショットでも可）
- 上記仮説のそれぞれについての観察結果コメント

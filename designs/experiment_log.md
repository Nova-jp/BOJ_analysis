# 実験結果ログ

新しい実験を行うたびに追記する。採否の根拠も必ず記録する。

**結果確認後の手順：**
1. `notebooks/main_model.ipynb` を再実行して最新ICを確認
2. このファイル（experiment_log.md）の表を更新
3. `designs/current_model.md` の IC 欄を更新

---

## 実験一覧

| 実験ID | 内容 | 3d IC | 5d IC | 3d Δ | 5d Δ | 採否 | 日付 |
|--------|------|-------|-------|------|------|------|------|
| **Baseline** | 現行42特徴量、rounds=100、全訓練データ使用 | **0.224** | **0.228** | — | — | — | 2026-03-21 |
| Exp-A | 曜日 sin/cos 追加 | 0.2465 | 0.2252 | +0.023 | -0.003 | **保留**（フォールド別確認待ち） | 2026-03-21 |
| Exp-B | Days_to_MPM sin/cos 追加 | 0.2453 | 0.2037 | +0.021 | -0.024 | **不採用** | 2026-03-21 |
| Exp-C | カーブ形状（Slope・Butterfly_M{2-7}）追加 | 0.2000 | 0.1700 | -0.024 | -0.058 | **不採用** | 2026-03-21 |
| Exp-D | Absolute_Meeting_ID 追加 | — | — | — | — | **採用** | 2026-03 |
| Exp-E | Days_since_first_seen 追加 | — | — | — | — | **採用** | 2026-03 |
| Exp-F | バタフライ目的変数変換 | 大幅低下 | 大幅低下 | — | — | **不採用** | 2026-03 |
| Exp-DXY | DXY_frac_diff 追加 | 再実験待ち | 再実験待ち | — | — | **再実験** | 2026-03-21 |
| Exp-Spot | T12/T18/T24 除外（寄与確認） | 0.1835 | 0.2306 | -0.041 | +0.003 | **現状維持** | 2026-03-21 |

> **注意**：旧Baselineの「0.389/0.633」は最終フォールドのみのIC（選択バイアス）。
> 正しい全OOS ICは上表の 3d=0.186、5d=0.221（rounds=300、start_date=2024-01-01、9フォールド）。

---

## 詳細記録

### Baseline（2026-03-21 再計測）
- **3d OOS IC: 0.224（直近3F: 0.414）、5d OOS IC: 0.228（直近3F: 0.449）**
- Train IC: 3d=0.517、5d=0.559、Gap: 3d=0.293、5d=0.331
- rounds=100：感度分析（06_rounds_sensitivity）で最適確認
- val_data（旧: 訓練の15%を除外）を削除し全訓練データで学習 → OOS IC改善
- 直近フォールド（Fold 6-8 = 2025以降）は IC が高い（利上げサイクルで安定）
- 初期フォールド（Fold 0-2 = NIRP解除直後）は分布シフトでICが低い（3d Fold1=-0.03）
- 特徴量：Exp-D（Absolute_Meeting_ID）、Exp-E（Days_since_first_seen）込みの現行特徴量セット
- データ修正：BOJ_meeting_history.csv を 34会合（2022〜）→ 217会合（2007〜）に再構築
  - 旧CSVの欠損で I4 が訓練データの80%を誤って除外していた問題を解消
- 旧ノートブック(03_model_analysis_executed)の「0.389/0.633」は最終フォールドのみのICであり無効

### Exp-A/B/C/DXY/Spot（2026-03-21 比較実験）

ノートブック: `notebooks/exp_feature_comparison.ipynb`

| 実験 | 3d IC | 3d Recent | 5d IC | 5d Recent | 判断 |
|------|-------|-----------|-------|-----------|------|
| Baseline | 0.2240 | 0.4140 | 0.2277 | 0.4494 | — |
| Exp-A (+曜日 sin/cos) | 0.2465 | 0.4234 | 0.2252 | 0.4128 | 保留 |
| Exp-B (+Days_to_MPM sin/cos) | 0.2453 | 0.4440 | 0.2037 | 0.3965 | 不採用 |
| Exp-C (+カーブ形状) | 0.2000 | 0.3470 | 0.1700 | 0.3660 | 不採用 |
| Exp-DXY (+DXY_frac_diff) | 0.2240 | 0.4140 | 0.2277 | 0.4494 | 再実験（バグ修正後） |
| Exp-Spot (T12/T18/T24 除外) | 0.1835 | 0.4004 | 0.2306 | 0.4604 | 現状維持 |

**Exp-B 不採用理由**: 5d が -0.024 と明確に悪化。Days_to_MPM は線形のまま使う方が安定。

**Exp-C 不採用理由**: 14特徴量追加によるオーバーフィット。カーブ情報は M_spread 水準特徴量にすでに含まれている。

**Exp-Spot 結論**: T12/T18/T24 は 3d 予測に大きく貢献（-0.041）。除外しない。

**Exp-A 保留**: 3d は +0.023 改善だが 5d は -0.003（誤差範囲）。フォールド別 IC を確認してから最終判断。

**Exp-DXY の問題**: `features.py`・`pooling.py`・`modeling.py` の全3箇所で DXY_frac_diff が抜けていた（ソースバグ）。2026-03-21 修正済み。再実験が必要。

---

### Exp-F（バタフライ目的変数変換）
- 手法：M1-M8スプレッドをB1-B8（水準・バタフライ・スロープ）に変換してからh日差分を予測
- 結果：IC大幅低下
- **不採用理由**：BOJ OISの主要な動きは並行シフト（全限月が同方向に動く）。バタフライ成分B2-B7は並行シフトで打ち消し合い、ほぼゼロになる。モデルは残差曲率を予測させられることになり信号が弱くなる。これはバグではなく変換の本質的な性質。
- 逆変換の数学は正しかったが（三対角行列、最大誤差1e-8以下）、評価②（M_n空間に戻したIC）も確認推奨
- ノートブック：`notebooks/08_advanced_feature_experiments.ipynb`

---

## 次の実験予定

| 実験ID | 内容 | 設計書 | ステータス |
|--------|------|--------|----------|
| **Exp-BFly-A** | **[RV Butterfly] Fly_Level削除 → B2_level〜B7_level（全6本水準）追加** | `designs/pending/exp_bfly_a_all_b_levels.md` | **Gemini 実行待ち** |
| Exp-DXY（再） | DXY_frac_diff 追加（src/ 3箇所修正済み） | `designs/pending/exp_abc_dxy_spot.md` | **Gemini 実行待ち** |
| Exp-A（確認） | 曜日 sin/cos のフォールド別IC確認 | 同上 | Gemini 確認待ち |
| Exp-SpotSpread | S{n}=M{n}-T12 の変化予測（新カーブモデル） | `designs/archives/exp_spot_spread_model.md` | **不採用**（2026-03-21） |
| Exp-CurveDir | M2-M4・M2-M5 単体方向予測（プーリングなし） | `designs/archives/exp_curve_direction_model.md` | **基礎確認**（v2に移行） |
| Exp-CurveDirV2 | Adjacent vs AllPair プーリング比較 | `designs/archives/exp_curve_direction_v2.md` | **Adjacent採用・M2-M5主軸**（2026-03-22） |

### Exp-SpotSpread（2026-03-21）

ノートブック: `notebooks/exp_spot_spread_model_executed.ipynb`

| ホライズン | Global IC | CS IC | Gap |
|-----------|----------|-------|-----|
| 3d | 0.2475 | **0.0414** | — |
| 5d | 0.3964 | **0.0909** | — |

**不採用理由**: CS IC が採用基準（3d ≥ 0.21）を大幅に下回った。

根本的な失敗要因: `ΔS{n} = ΔM{n} - ΔT12` において T12 は全 S1〜S8 の共通成分であり、
T12 の動きがモデルの Global IC に寄与する一方、CS IC（日次クロスセクションのランク予測）は
「S1〜S8 が同日に一斉に同方向へ動く」ために完全にノイズになる。
RV Curve（隣接差分）や RV Butterfly（バタフライ）が CS IC で高くなれるのは、
隣接差分・バタフライ変換によって共通の T12/全体方向成分を打ち消しているためであり、
T12 を基準点にした Spot Spread はその性質を持たない。

特徴量重要度（5d）Top5: S4_frac_diff, Days_to_MPM, T12_spread, Spread_Level, S5_frac_diff

---

### Exp-CurveDirV2（2026-03-22）

ノートブック: `notebooks/archives/exp_curve_direction_v2_executed.ipynb`

**比較: Model A（Adjacent C1-C7, 7系列）vs Model B（AllPair 全28系列）**

| モデル | Global IC 3d | Train IC 3d | Gap 3d | Global IC 5d | Train IC 5d | Gap 5d |
|--------|-------------|-------------|--------|-------------|-------------|--------|
| Model A (Adjacent) | 0.2228 | 0.4681 | 0.2453 | 0.2717 | 0.5257 | 0.2540 |
| Model B (AllPair)  | 0.2022 | 0.5141 | 0.3119 | 0.1825 | 0.5559 | 0.3735 |

**M2-M5 個別評価（採用スパン）:**

| モデル | IC 3d | Dir 3d | Dir_Large 3d | IC 5d | Dir 5d | Dir_Large 5d |
|--------|-------|--------|--------------|-------|--------|--------------|
| Model A | 0.2604 | 0.5421 | 0.6667 | **0.3297** | **0.5986** | 0.6887 |
| Model B | 0.1853 | 0.5467 | 0.6569 | 0.2263 | 0.5915 | 0.6132 |

**採用: Model A（Adjacent、既存 `pool_curve_data` と同一実装）、主軸スパン M2-M5**

採用理由:
- Model A が全指標で Model B を上回る
- Model B は Gap が大きく（3d=0.31, 5d=0.37）過学習が顕著。28系列の高相関によりノイズを反復学習
- M2-M5 は M2-M4 より IC・方向的中率ともに高い（5d IC: 0.330 vs 0.318）
- 既存 `pooling_curve.py` を流用可能。新規 src/ モジュール不要

---

### 修正済みの src/ バグ（2026-03-21）
DXY が `features.py`・`pooling.py`・`modeling.py` の全3箇所で特徴量から漏れていた。
修正内容: 各ファイルの `ext_cols` / `ext_fd_cols` に `DXY` / `DXY_frac_diff` を追加。

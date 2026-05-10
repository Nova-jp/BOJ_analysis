# CLAUDE.md

このリポジトリで作業する Claude Code へのガイダンス。プロジェクト概要・使い方は `README.md` を参照。

---

## 役割分担

| 担当 | 作業内容 |
|------|---------|
| **Claude** | `src/` 設計・実装、設計書作成、分析レビュー |
| **Gemini CLI** | 特徴量実験（ノートブック）、可視化 |

根本的な構成・アーキテクチャの変更は Claude が行う。

---

## モデルファミリー

| ファミリー | 主指標 | 設計書 | ノートブック |
|-----------|--------|--------|------------|
| **RV Butterfly** ★メイン | CS IC | `designs/rv_butterfly_model.md` | `notebooks/model_comparison.ipynb` |
| **RV Curve** | CS IC | `designs/rv_curve_model.md` | `notebooks/model_comparison.ipynb` |
| **Outright** | Global IC | `designs/current_model.md` | `notebooks/main_model.ipynb` |

---

## src/ モジュール構成

| ファイル | 役割 |
|---------|------|
| `processing.py` | データ読み込み・MICE補完 |
| `modeling.py` | walk_forward・IC評価（除外リスト方式） |
| `features.py` | Outright特徴量 + `frac_diff`（`features_rv.py` からもインポート） |
| `features_rv.py` | RV Curve / Butterfly 共通特徴量 |
| `pooling.py` / `pooling_curve.py` / `pooling_butterfly.py` | 各モデルのプーリング |

`modeling.py` の `NON_FEATURE_COLS` 除外リスト方式：新特徴量追加時に `modeling.py` の変更不要。

---

## 開発方針

- **実験フロー**: `designs/pending/` に設計書作成 → Gemini が実装 → レビュー → `designs/experiment_log.md` に記録
- **採用基準**: CS IC +0.01 以上改善かつ全フォールド安定、start_date=2024-01-01 固定
- **実験前**: `notebooks/feature_engineering_base.ipynb` でベースラインを確認
- 1実験が完了したら会話を切り新規会話で次の実験へ（コンテキスト分離）

---

## 設計上の重要制約

| 場所 | 制約・注意点 |
|------|------------|
| `processing.py` | MICE補完・`bfill()` は全期間データ（lookahead）。意図的。`{col}_is_imputed` で識別 |
| `pooling*.py` | 正規化 std も全期間。IC = Spearman のため定数倍に不変 |
| `modeling.py` | early stopping なし（固定100ラウンド）。非定常時系列で val loss が即発散するため |
| `modeling.py` | post-MPM 除外なし（廃止済み）。Fold≥5 の CS IC = 0.37 と高く除外不要（2026-03-23 確認） |
| `features_rv.py` | `features.py` から `frac_diff` をインポート。`features.py` 削除時は移動すること |
| `trade_signal.py` | `_m2m5_signal` がグローバル `df_crv` に暗黙依存。関数化時は明示的引数渡しに変更 |

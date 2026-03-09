# GEMINI.md

このファイルは、このリポジトリで作業する Gemini CLI へのガイダンスを提供します。

---

## プロジェクトの目的

日銀（BOJ）OISスワップレート（M1〜M8）の翌日・3日後・5日後の変化量を予測し、アウトライト・カーブスプレッドのトレーディングシグナルを生成する。GBDTをベースとした機械学習モデルを使用する。

---

## 役割分担

| 担当 | 作業内容 |
|------|---------|
| **Gemini CLI** | `src/` の実装、ノートブックの作成・実験、可視化 |
| **Claude** | 設計方針の決定、`src/` 実装のレビュー、分析結果の解釈 |

**実装前に必ず `designs/` の設計書を読むこと。** 特に `designs/01_model_design.md`（設計方針）と `designs/02_implementation_spec.md`（実装仕様）が最重要。

`src/` のアーキテクチャに関わる変更は Claude に確認してから行うこと。

---

## 環境セットアップ

```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## ディレクトリ構成

```
BOJ_analysis/
├── data/                    # 生データのみ（中間ファイルはgitignore済み）
│   ├── BOJ_data.xlsx        # Reuters生データ
│   └── BOJ_meeting_history.csv
├── notebooks/               # 分析ノートブック（Gemini CLIが作成・実験）
├── src/                     # 共通ライブラリ（Gemini CLIが実装）
├── designs/                 # 設計書（Claudeが作成・Gemini CLIが参照）
└── archives/                # 旧ファイル（参照用のみ、編集不要）
```

---

## 開発ルール

### Jupyter Notebook の作成
- `.ipynb` を新規作成する際は `write_file` ツールを直接使わず、**Python の `json` ライブラリを使ってプログラム経由で生成**すること（コントロールキャラクターのエラー回避のため）
- 既存ノートブックの編集は直接行ってよい

### `src/` の実装方針
- 各モジュールは単独でインポートできるよう設計する
- ノートブックからは `src/` の関数を `import` して使う
- `src/` を書き換えた場合は変更内容を簡潔に報告すること（Claudeがレビューする）

### 実験の記録
- 試した特徴量・パラメータとその結果（IC・方向的中率への影響）をノートブックのMarkdownセルに残す

---

## 技術仕様（設計書の要約）

詳細は `designs/01_model_design.md` を参照。

### データパイプライン

```
data/BOJ_data.xlsx + data/BOJ_meeting_history.csv
    ↓  src/processing.py  load_and_clean_data()
    - MICE補完（BayesianRidge）
    - Is_Meeting_Day フラグ
    - Actual_Policy_Rate（前方補完）
    - M{n}_is_imputed フラグ
    ↓  src/features.py  generate_features()
    - M{n}_spread = M{n} - Actual_Policy_Rate
    - M{n}_frac_diff = frac_diff(M{n}_spread, d=0.4)
    - 外部指標の分数階差
    - Days_to_MPM
    ↓  src/pooling.py  pool_boj_data()
    - 横持ち → 縦持ち変換
    - 全限月の特徴量を全行に結合（各行がM1〜M8全ての値を持つ）
    - 目的変数生成：Target_1d / Target_3d / Target_5d
    - MPM直後5日間の除外フラグ（is_post_mpm）
```

### モデル

- **プーリング**：M1〜M8を縦積み、`Meeting_Index`（1〜8）を特徴量として使用
- **ホライゾン別に独立したモデル**：1日後・3日後・5日後それぞれ別モデル
- **損失関数**：MSE（MAEとの比較実験あり）

### 評価指標

- IC（Spearman相関）・ICIR
- 方向的中率（全体 + |実ΔM| > 上位25%の大動き限定）
- Meeting_Index別・Days_to_MPM別に層別して確認

### バリデーション

- ウォークフォワード検証（拡張ウィンドウ）
- 訓練末とテスト開始の間に5日のパージ
- ゼロ金利期（〜2024年3月）と利上げ期（2024年3月〜）で評価を分けて報告

---

## 現在のタスク

`designs/02_implementation_spec.md` に詳細な実装仕様を記載している。以下の順で進めること：

1. `src/processing.py`：仕様確認・必要なら修正
2. `src/features.py`：全面書き直し
3. `src/pooling.py`：全面書き直し
4. `notebooks/01_eda.ipynb`：EDA・基本検証ノートブックの作成

各タスク完了後に実装内容と確認事項の結果を報告すること。

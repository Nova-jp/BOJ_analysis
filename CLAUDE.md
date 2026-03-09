# CLAUDE.md

このファイルは、このリポジトリで作業する Claude Code (claude.ai/code) へのガイダンスを提供します。

---

## プロジェクトの目的

日銀（BOJ）OISスワップレート（M1〜M8：第1〜第8会合先）を分析し、MPM（金融政策決定会合）前後のレート変動を予測するトレーディング戦略の構築。機械学習（現在はGBDT/CatBoost）を積極活用する。

---

## 役割分担

| 担当 | 作業内容 |
|------|---------|
| **Claude** | ディレクトリ構成・`src/`コードの設計と実装、分析方針の策定、特徴量の精査・解釈、分析結果のレビュー |
| **Gemini CLI** | 特徴量の試行錯誤（書いては消すサイクル）、可視化の実装、ノートブック上での実験 |

根本的な構成・アーキテクチャに関わる変更は Claude が行う。特徴量エンジニアリングの実験的な部分は Gemini CLI が担当する。

---

## ディレクトリ構成

```
BOJ_analysis/
├── data/                    # 生データ・中間データ
│   ├── BOJ_data.xlsx        # Reuters生データ（日付列は日本語形式 `日付`）
│   └── BOJ_meeting_history.csv  # MPM履歴（Date, Event, Policy_Rate）
├── notebooks/               # 番号付き分析ノートブック（正規の分析フロー）
├── src/                     # 共通Pythonライブラリ（Claudeが管理）
├── designs/                 # 分析方針・設計メモ（Gemini CLIへの指示書を含む）
└── archives/                # 旧ノートブック・旧スクリプト（参照用のみ）
```

---

## 環境

```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## データとモデルの構造

### データの列名対応

Reutersティッカーは `src/processing.py` でリネームされる：

| 元のティッカー | 変換後 | 意味 |
|--------------|--------|------|
| `JPBOJ{n}ONI=TRDT (MID_PRICE)` | `M1`〜`M8` | BOJ OISスワップ（第n回会合先） |
| `JPY= (MID_PRICE)` | `USDJPY` | ドル円 |
| `JGBc1 (TRDPRC_1)` | `JGB_Future` | JGB先物 |
| `.N225 (TRDPRC_1)` | `Nikkei225` | 日経225 |
| `.DXY (TRDPRC_1)` | `DXY` | ドルインデックス |

### パイプライン概要

```
data/BOJ_data.xlsx + data/BOJ_meeting_history.csv
        ↓  src/processing.py  (load_and_clean_data)
        - MICE補完（BayesianRidge）
        - Is_Meeting_Day フラグ、Actual_Policy_Rate（前方補完）
        - {col}_is_imputed フラグ
        ↓  src/features.py  (generate_features)
        - スプレッド = M{n} - Actual_Policy_Rate
        - 分数階差（d=0.4）による定常化
        - 外部指標の20日MA乖離率
        ↓  src/pooling.py  (pool_boj_data)
        - ワイド（M1-M8）→ロング形式への変換（プーリング）
        - n日後ターゲット生成
        - Days_to_MPM 特徴量
        ↓  CatBoost MultiRMSE  (src/modeling.py)
```

### 主要な設計判断

- **スプレッドベース**：M{n} の絶対水準でなく政策金利差（スプレッド）をターゲットにする
- **プーリング**：M1〜M8を縦積みして1モデルで学習することで限月間の共通パターンを捉える
- **MPM直後除外**：会合翌日から5日間はターゲットとして不安定なため学習データから除外
- **補完フラグ**：欠損補完箇所をモデルに伝えるため `{col}_is_imputed` を特徴量に含める

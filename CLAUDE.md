# CLAUDE.md

このファイルは、このリポジトリで作業する Claude Code (claude.ai/code) へのガイダンスを提供します。

---

## プロジェクトの目的

日銀（BOJ）OISスワップレート（M1〜M8：第1〜第8会合先）を分析し、MPM（金融政策決定会合）前後のレート変動を予測するトレーディング戦略の構築。機械学習（GBDT/LightGBM）を積極活用する。

---

## 役割分担

| 担当 | 作業内容 |
|------|---------|
| **Claude** | ディレクトリ構成・`src/`コードの設計と実装、分析方針の策定・設計書作成、特徴量の精査・解釈、分析結果のレビュー |
| **Gemini CLI** | 特徴量の試行錯誤（書いては消すサイクル）、可視化の実装、ノートブック上での実験 |

根本的な構成・アーキテクチャに関わる変更は Claude が行う。特徴量エンジニアリングの実験的な部分は Gemini CLI が担当する。

---

## モデルファミリー

| ファミリー | 予測対象 | 主指標 | 設計書 | ノートブック |
|-----------|---------|--------|--------|------------|
| **Outright** | M1〜M8 の絶対水準変化 | Global IC / TS IC | `designs/current_model.md` | `notebooks/main_model.ipynb` |
| **RV Curve** | 隣接差分 C{n}=M{n}−M{n+1} の変化 | CS IC | `designs/rv_curve_model.md` | `notebooks/model_comparison.ipynb` |
| **RV Butterfly** ★メイン | バタフライ B{n}=2M{n}−M{n-1}−M{n+1} の変化 | CS IC | `designs/rv_butterfly_model.md` | `notebooks/model_comparison.ipynb` |

★ RV Butterfly が最終採用モデル（CS IC 3d=0.304, 5d=0.346, Gap=0.148）。

---

## ディレクトリ構成

```
BOJ_analysis/
├── data/                         # 生データ・中間データ
│   ├── BOJ_data.xlsx             # Reuters生データ（日付列は日本語形式 `日付`）
│   └── BOJ_meeting_history.csv   # MPM履歴（Date, Event, Policy_Rate）
├── notebooks/
│   ├── main_model.ipynb          # ★ Outright モデル（単一の真実源）
│   ├── model_comparison.ipynb    # ★ 3モデル比較（RV Curve・Butterfly の真実源）
│   ├── trade_signal.ipynb        # ★ トレードシグナル生成（日次参照資料）
│   ├── 01_eda.ipynb              # EDA（参照用）
│   ├── 10_hedge_ratio.ipynb      # ヘッジレシオ分析
│   └── archives/                 # 実験・旧ノートブック（参照用のみ）
├── src/                          # 共通Pythonライブラリ（Claudeが管理）
│   ├── processing.py             # 【共有】データ読み込み・MICE補完
│   ├── modeling.py               # 【共有】walk_forward・train・IC評価
│   ├── features.py               # Outright モデル専用
│   ├── pooling.py                # Outright モデル専用
│   ├── features_rv.py            # RV Curve / Butterfly 共通特徴量
│   ├── pooling_curve.py          # RV Curve モデル専用プーリング
│   └── pooling_butterfly.py      # RV Butterfly モデル専用プーリング
├── designs/
│   ├── current_model.md          # ★ Outright モデル仕様（常に最新）
│   ├── rv_curve_model.md         # ★ RV Curve モデル仕様（常に最新）
│   ├── rv_butterfly_model.md     # ★ RV Butterfly モデル仕様（常に最新）
│   ├── experiment_log.md         # ★ 全モデルファミリーの実験記録
│   ├── pending/                  # 次に試す実験の設計書（Geminiへの指示書）
│   └── archives/                 # 不採用になった設計書（削除せずここに移動）
├── scripts/
│   └── run_all_experiments.py    # 実験一括実行スクリプト
└── archives/                     # 旧スクリプト・旧ノートブック（参照用のみ）
```

★ の付いたファイルは常に最新状態を保つこと。

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
| `JP12MONI=TRDT (BID)` | `T12` | 12ヶ月テナーOIS（スポット） |
| `JP18MONI=TRDT (BID)` | `T18` | 18ヶ月テナーOIS（スポット） |
| `JP24MONI=TRDT (BID)` | `T24` | 24ヶ月テナーOIS（スポット） |

### パイプライン概要

**【共通前処理】**
```
data/BOJ_data.xlsx + data/BOJ_meeting_history.csv
        ↓  src/processing.py  (load_and_clean_data)
        - MICE補完（BayesianRidge）
        - Is_Meeting_Day フラグ、Actual_Policy_Rate（前方補完）
        - {col}_is_imputed フラグ
        - Days_to_MPM（次回会合=M1の会合まで、全会合CSVから計算）
```

**【Outright モデル】**
```
        ↓  src/features.py  (generate_features)
        - スプレッド = M{n} - Actual_Policy_Rate
        - 分数階差（d=0.4, window=50）
        ↓  src/pooling.py  (pool_boj_data)
        - M1〜M8をプーリング（T12/T18/T24含む）
        - Absolute_Meeting_ID・Days_since_first_seen
```

**【RV Curve モデル】**
```
        ↓  src/features_rv.py  (generate_rv_features)
        - C{n} = M{n}_spread - M{n+1}_spread（n=1..7）
        - C{n}_frac_diff、M1アンカー、外部指標
        ↓  src/pooling_curve.py  (pool_curve_data)
        - C1〜C7（7ペア）をプーリング
        - Days_to_MPM = 次回MPMまでの日数（全銘柄共通）
```

**【RV Butterfly モデル】**
```
        ↓  src/features_rv.py  (generate_rv_features)   ← Curve と共有
        - B{n} = 2*M{n}_spread - M{n-1}_spread - M{n+1}_spread（n=2..7）
        - B{n}_frac_diff、M1アンカー、M1-M8スロープ、外部指標
        ↓  src/pooling_butterfly.py  (pool_butterfly_data)
        - B2〜B7（6系列）をプーリング
        - Days_to_MPM = 次回MPMまでの日数（全銘柄共通）
```

**【共通モデル学習・評価】**
```
        ↓  src/modeling.py  (walk_forward_validation)
        - 拡張窓、purge=5日、test_window=90日
        - 固定100ラウンド、early stopping なし
        - summarize_ic: Global IC / CS IC / TS IC
```

### 主要な設計判断

- **スプレッドベース**：M{n} の絶対水準でなく政策金利差（スプレッド）を特徴量ベースにする
- **プーリング**：M1〜M8を縦積みして1モデルで学習することで限月間の共通パターンを捉える
- **MPM直後除外**：会合翌日から5日間はターゲットとして不安定なため学習データから除外
- **正規化**：銘柄ごとの全期間stdで正規化（Target_{h}d_norm）し異分散を補正。逆正規化用にstdも保持
- **補完フラグ**：欠損補完箇所をモデルに伝えるため `{col}_is_imputed` を特徴量に含める
- **early stopping なし**：金融時系列の非定常性でval lossが即発散するフォールドが生じるため固定ラウンド数

---

## 現在のメインモデル（2026-03-21確定）

新しい会話で特徴量エンジニアリングを始める際は、以下をベースラインとして使うこと。

### 特徴量一覧

| カテゴリ | 特徴量 | 備考 |
|---------|--------|------|
| コンテキスト | `Meeting_Index`（カテゴリ） | 限月 1〜8 |
| | `Is_Tenor_OIS` | M1-M8=0、スポットOIS=1 |
| | `Days_to_MPM`（整数） | 次回会合までの日数 |
| | `Actual_Policy_Rate` | 現在の政策金利 |
| スプレッド水準 | `M1_spread`〜`M8_spread` | M{n} - Actual_Policy_Rate |
| 分数階差 | `M1_frac_diff`〜`M8_frac_diff` | スプレッドの分数階差（d=0.4） |
| 補完フラグ | `M1_is_imputed`〜`M8_is_imputed` | MICE補完された行 |
| 外部指標 | `USDJPY_frac_diff` | ドル円の分数階差 |
| | `JGB_Future_frac_diff` | JGB先物の分数階差 |
| | `Nikkei225_frac_diff` | 日経225の分数階差 |
| 会合識別子 | `Absolute_Meeting_ID`（カテゴリ） | データ先頭から連番（Exp-D採用） |
| | `Days_since_first_seen` | その会合の初観測から何日目か（Exp-E採用） |

### ベースラインIC（OOS, walk-forward, start_date=2024-01-01, rounds=100）

- **3d**: IC = 0.224（全フォールド）/ 0.414（直近3フォールド）
- **5d**: IC = 0.228（全フォールド）/ 0.449（直近3フォールド）

> **注意**：旧値「0.389/0.633」は最終フォールドのみのIC（選択バイアス）であり無効。
> 正しい全OOS ICは上記（2026-03-21再計測）。

特徴量実験の評価には `src/modeling.py` の `summarize_ic(results)` を使うこと。
返り値: `ic_all`, `ic_recent`（直近3F）, `train_ic`, `gap`, `ic_by_fold`

### 不採用となった特徴量・手法

| 手法 | 理由 |
|------|------|
| バタフライ目的変数変換（Exp-F） | 並行シフトでB2-B7の信号が消えICが大幅低下 |

---

## 特徴量エンジニアリングの開発方針

### 基本ルール

1. **メインモデルを固定してから実験する**：1度に1つの変更のみ行い、ICの改善・悪化を単離する
2. **同一の walk-forward split で比較する**：start_date=2024-01-01 で固定
3. **採用基準**：OOS IC が +0.01 以上改善かつ全フォールドで安定
4. **実験は独立したノートブックに**：Gemini が担当し、必ずセル1でメインモデルのICを再現してから差分を計算する

### ファイル管理

| ファイル | 役割 |
|---------|------|
| `designs/current_model.md` | 現在のメインモデル仕様（常に最新に保つ） |
| `designs/experiment_log.md` | 全実験の結果記録（IC・採否・コメント・日付） |
| `designs/pending/` | 次に試す実験の設計書（Geminiへの指示書） |
| `designs/archives/` | 不採用になった設計書（削除せずここに移動） |

### 実験のライフサイクル

```
Claude が designs/pending/ に設計書を作成
        ↓
Gemini CLI が独立ノートブックで実装・実験
        ↓
Gemini が結果（IC等）を報告
        ↓
Claude がレビュー → 採用 or 不採用を判断
        ↓
採用：src/ に組み込み、current_model.md を更新
不採用：設計書を designs/archives/ に移動
        ↓
experiment_log.md に記録して次の実験へ
```

### 実験間のコンテキスト分離

- 1実験が完了したら会話を切り、新規会話で次の実験を開始する
- 新規会話では `designs/current_model.md` と `designs/experiment_log.md` を参照すればすぐに現状を把握できる

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
| **RV Curve** | 隣接差分 C{n}=M{n}−M{n+1} の変化・M2-M5カーブ方向シグナル（C2+C3+C4合算） | CS IC | `designs/rv_curve_model.md` | `notebooks/model_comparison.ipynb` |
| **RV Butterfly** ★メイン | バタフライ B{n}=2M{n}−M{n-1}−M{n+1} の変化 | CS IC | `designs/rv_butterfly_model.md` | `notebooks/model_comparison.ipynb` |

★ RV Butterfly が最終採用モデル（CS IC 3d=0.304, 5d=0.346, Gap=0.148）。
RV Curve は Adjacent C1-C7（7系列）プーリング採用（Exp-CurveDirV2）。C2+C3+C4 の合算で M2-M5スティープナー/フラットナー方向を抽出。

---

## ディレクトリ構成

```
BOJ_analysis/
├── data/                              # 生データ
│   ├── BOJ_data.xlsx                  # Reuters生データ（日付列は日本語形式 `日付`）
│   └── BOJ_meeting_history.csv        # MPM履歴（Date, Event, Policy_Rate）
├── notebooks/
│   ├── feature_engineering_base.ipynb # ★ FE基盤（ベースライン確認・残差分析・方針策定）
│   ├── main_model.ipynb               # ★ Outright モデル（単一の真実源）
│   ├── model_comparison.ipynb         # ★ 3モデル比較（RV Curve・Butterfly の真実源）
│   ├── 01_eda.ipynb                   # EDA（参照用）
│   ├── 10_hedge_ratio.ipynb           # ヘッジレシオ分析
│   └── archives/                      # 実験・旧ノートブック（参照用のみ）
├── src/                               # 共通Pythonライブラリ（Claudeが管理）
│   ├── processing.py                  # 【共有】データ読み込み・MICE補完
│   ├── modeling.py                    # 【共有】walk_forward・train・IC評価
│   ├── features.py                    # Outright モデル専用（frac_diff 関数を含む）
│   ├── pooling.py                     # Outright モデル専用
│   ├── features_rv.py                 # RV Curve / Butterfly 共通特徴量
│   ├── pooling_curve.py               # RV Curve モデル専用プーリング
│   └── pooling_butterfly.py           # RV Butterfly モデル専用プーリング
├── designs/
│   ├── current_model.md               # ★ Outright モデル仕様（常に最新）
│   ├── rv_curve_model.md              # ★ RV Curve モデル仕様（常に最新）
│   ├── rv_butterfly_model.md          # ★ RV Butterfly モデル仕様（常に最新）
│   ├── experiment_log.md              # ★ 全モデルファミリーの実験記録
│   ├── pending/                       # 次に試す実験の設計書（Geminiへの指示書）
│   └── archives/                      # 不採用になった設計書（削除せずここに移動）
├── trade_signal.py                    # ★ 日次シグナル生成スクリプト（会社PC用）
├── outputs/                           # 生成物（PDF）
└── archives/                          # 旧スクリプト・旧ノートブック（参照用のみ）
```

★ の付いたファイルは常に最新状態を保つこと。

### trade_signal.py について

`python trade_signal.py` で実行。3ページのPDFを `outputs/trade_signal_YYYYMMDD.pdf` に出力する。

- Page 1: 予測値サマリー（Butterfly/Curve/Outright）+ M1-M8逆算テーブル
- Page 2: IC診断（Fold5以降の散布図・fold別Global IC棒グラフ）
- Page 3: 特徴量重要度（全6モデル）

M1-M8逆算: B2-B7（6式）+ M4 Outright（1式）+ M2-M5 Curve（1式）= 8×8連立方程式を `np.linalg.solve` で解く。

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
        ↓  src/pooling.py  (pool_boj_data(df, meeting_dates))  ← meeting_dates 必須
        - M1〜M8をプーリング（T12/T18/T24含む）
        - Absolute_Meeting_ID・Days_since_first_seen
```

**【RV Curve モデル】**
```
        ↓  src/features_rv.py  (generate_rv_features)
        - C{n} = M{n}_spread - M{n+1}_spread（n=1..7）
        - C{n}_frac_diff、M1アンカー、外部指標
        ↓  src/pooling_curve.py  (pool_curve_data)
        - C1〜C7（7ペア）Adjacent プーリング（Exp-CurveDirV2 採用）
        - Days_to_MPM = 次回MPMまでの日数（全銘柄共通）
        ↓  M2-M5 方向シグナル抽出（trade_signal.ipynb 内 _get_m2m5_signal）
        - C2・C3・C4 の予測値を逆正規化して合算 → M2-M5 スプレッド変化量（bp）
        - 正値 = Steeper（M2-M5 スプレッド縮小）、負値 = Flatter
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
        ↓  src/modeling.py
        - walk_forward_validation(df, target, ...) → DataFrame のみ返却
        - walk_forward_with_model(df, target, ...) → (results, model, X_test, y_test, X_train) を返却
        - 拡張窓、purge=5日、test_window=90日
        - 固定100ラウンド、early stopping なし
        - summarize_ic(results) → ic_all, ic_recent, cs_ic, ts_ic, train_ic, gap, ic_by_fold
```

### 主要な設計判断

- **スプレッドベース**：M{n} の絶対水準でなく政策金利差（スプレッド）を特徴量ベースにする
- **プーリング**：M1〜M8を縦積みして1モデルで学習することで限月間の共通パターンを捉える
- **MPM直後除外**：会合翌日から5日間はターゲットとして不安定なため学習データから除外
- **正規化**：銘柄ごとの全期間stdで正規化（Target_{h}d_norm）し異分散を補正。逆正規化用にstdも保持
- **補完フラグ**：欠損補完箇所をモデルに伝えるため `{col}_is_imputed` を特徴量に含める
- **early stopping なし**：金融時系列の非定常性でval lossが即発散するフォールドが生じるため固定ラウンド数

---

## 現在のメインモデル（RV Butterfly, 2026-03-23更新: post-MPM除外廃止）

新しい会話で特徴量エンジニアリングを始める際は、以下をベースラインとして使うこと。
**`notebooks/feature_engineering_base.ipynb` を実行して現状を確認してから実験を始める。**

### RV Butterfly 特徴量一覧（26特徴量）

| カテゴリ | 特徴量 | 備考 |
|---------|--------|------|
| コンテキスト | `Meeting_Index`（カテゴリ） | Butterfly_Index 2〜7 |
| | `Days_to_MPM`（整数） | 次回会合までの日数 |
| | `Actual_Policy_Rate` | 現在の政策金利 |
| バタフライ水準 | `Fly_Level` | B{n} の現在水準（mean reversion の起点） |
| M1 アンカー | `M1_spread` | 金利水準のコンテキスト |
| | `M1_frac_diff` | M1_spread の分数階差 |
| M1-M8 スロープ | `Slope_M1M8` | カーブのスロープ水準 |
| | `Slope_M1M8_frac_diff` | スロープの分数階差 |
| バタフライ分数階差 | `B2_frac_diff`〜`B7_frac_diff` | 各バタフライの分数階差（d=0.4, window=50） |
| 補完フラグ | `M1_is_imputed`〜`M8_is_imputed` | MICE補完フラグ |
| 外部指標 | `USDJPY_frac_diff`, `JGB_Future_frac_diff` | 分数階差 |
| | `Nikkei225_frac_diff`, `DXY_frac_diff` | 分数階差 |

### ベースライン IC（RV Butterfly, OOS, start_date=2024-01-01, rounds=100, post-MPM除外なし）

| Horizon | Global IC | CS IC | Train IC | Gap |
|---------|-----------|-------|----------|-----|
| **3d** | 0.378 | **0.296** | 0.509 | 0.132 |
| **5d** | 0.434 | **0.355** | 0.564 | 0.130 |

**CS IC が主指標**（クロスセクション相対価値戦略の評価に最適）。
採用基準: CS IC が +0.01 以上改善 かつ全フォールドで安定。

post-MPM除外廃止の根拠: MPM直後5日間のCS ICはFold≥5で合計0.37と高く、除外する理由がない（2026-03-23）。

特徴量実験の評価には `src/modeling.py` の `summarize_ic(results, instrument_indices=set(range(2, 8)))` を使うこと。
返り値: `ic_all`, `ic_recent`（直近3F）, `cs_ic`, `ts_ic`, `train_ic`, `gap`, `ic_by_fold`

### Outright モデルのベースライン IC（参考）

- **3d**: Global IC = 0.224 / CS IC = -0.005（相対価値には使えない）
- 特徴量の詳細は `designs/current_model.md` を参照

### 不採用となった特徴量・手法

| 手法 | 理由 |
|------|------|
| バタフライ目的変数変換（Exp-F） | 並行シフトでB2-B7の信号が消えICが大幅低下 |
| Days_to_MPM sin/cos（Exp-B） | 5d が -0.024 と悪化 |
| カーブ形状特徴量（Exp-C） | 14特徴量追加によるオーバーフィット、IC低下 |

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
- 新規会話では `designs/rv_butterfly_model.md` と `designs/experiment_log.md` を参照すればすぐに現状を把握できる

---

## コードレビュー知見（Senior MLE, 2026-03-22）

### 既知の設計上の制約（意図的な選択）

| 場所 | 制約 | 理由・注意点 |
|------|------|-------------|
| `src/processing.py` | MICE補完を全期間データで fit_transform（lookahead） | 欠損率が低く・金利間相関が安定しているため許容。`{col}_is_imputed` フラグで補完箇所を学習時に識別可能 |
| `src/pooling*.py` | 正規化 std も全期間 std（lookahead） | IC は Spearman ランク相関のため定数倍に不変。影響ゼロ |
| `src/modeling.py` | early stopping なし（固定100ラウンド） | 非定常な金融時系列でval lossが即発散するフォールドが多発するため固定ラウンドが安定 |
| `src/modeling.py` | post-MPM5日間の除外なし（is_post_mpm フィルタ廃止） | MPM直後のCS ICがFold≥5で0.37と高く、除外する理由がない（2026-03-23実験確認） |
| `src/processing.py` | `bfill()` で政策金利を後方補完 | CSV が 2007年まで整備されているため安全。CSV を短縮すると先頭に未来の金利が混入するリスクあり |

### 注意が必要な依存関係

| 依存元 | 依存先 | 内容 |
|--------|--------|------|
| `src/features_rv.py` | `src/features.py` | `frac_diff` 関数をインポートしている。`features.py` を削除する場合は `frac_diff` を別モジュールに移動すること |
| `trade_signal.py` | グローバル変数 `df_crv` | `_m2m5_signal` 関数が暗黙的にグローバルスコープの `df_crv` を参照している。スタンドアロンスクリプトとして問題ないが、関数化する場合は明示的な引数渡しに変更すること |

### `src/modeling.py` の除外リスト方式

```python
NON_FEATURE_COLS = {'Date', 'Rate_Label', 'Rate_Value', 'is_post_mpm', ...}
```

特徴量は「除外リスト以外の全列を自動採用」する方式。新しい列を追加する際は **このリストを変更不要**。除外したい列だけここに追加する。サイレントバグを防ぐ設計。

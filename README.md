# BOJ OIS Analysis

日銀（BOJ）の金融政策決定会合（MPM）前後における OIS スワップレートの変動を予測する、
機械学習ベースのトレーディングシグナル生成システム。

## 概要

日銀は年8回の MPM（金融政策決定会合）で政策金利を決定する。
本プロジェクトは Reuters から取得した M1〜M8（第1〜第8会合先フォワード OIS レート）を
LightGBM（GBDT）で分析し、MPM を跨いだレート変動の方向・大きさを予測する。

### モデルファミリー

| モデル | 予測対象 | 主指標 | 設計書 |
|--------|---------|--------|--------|
| **RV Butterfly**（メイン） | バタフライ B{n} = 2M{n} − M{n-1} − M{n+1} の変化（カーブの曲率） | CS IC | `designs/rv_butterfly_model.md` |
| **RV Curve** | 隣接スプレッド C{n} = M{n} − M{n+1} の変化（M2-M5 スティープ/フラット方向） | CS IC | `designs/rv_curve_model.md` |
| **Outright** | M1〜M8 絶対水準の変化（方向参考） | Global IC | `designs/current_model.md` |

**CS IC**（クロスセクション情報係数）が主評価指標。
各日付で B2〜B7 または C1〜C7 をランク付けし、Spearman 相関の時系列平均を取る。

### ベースライン性能（OOS, 2024-01-01〜）

| モデル | Horizon | CS IC |
|--------|---------|-------|
| RV Butterfly | 3d | 0.296 |
| RV Butterfly | 5d | 0.355 |

---

## セットアップ

```bash
git clone <repo>
cd BOJ_analysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### データ配置

`data/` に以下の2ファイルを配置する（Reuters から取得・非公開）。

| ファイル | 内容 |
|---------|------|
| `data/BOJ_data.xlsx` | OIS・為替・株価等の日次データ（日付列: `日付`） |
| `data/BOJ_meeting_history.csv` | MPM 開催日・政策金利履歴（`Date`, `Event`, `Policy_Rate`） |

---

## スクリプト

### 日次シグナル生成

```bash
python trade_signal.py
# → outputs/trade_signal_YYYYMMDD.pdf
```

3ページ構成の PDF を出力。

| ページ | 内容 |
|--------|------|
| Page 1 | 予測値サマリー（Butterfly / Curve / Outright）+ M1-M8 逆算テーブル |
| Page 2 | IC 診断（Fold5 以降の散布図・fold 別 Global IC 棒グラフ） |
| Page 3 | 特徴量重要度（全6モデル） |

M1-M8 逆算は B2-B7（6式）+ M4 Outright（1式）+ M2-M5 Curve（1式）= 8×8 連立方程式を `np.linalg.solve` で解く。

### OIS 計算ツール

```bash
python tools/ois_calc.py [--date YYYY-MM-DD]
# → outputs/ois_calculator_YYYYMMDD.xlsx
```

グリッドレート（M1-M8）とスポット OIS の相互変換、および利上げパスシミュレーションを行う Excel を生成。

| シート | 内容 |
|--------|------|
| `設定` | 本日日付・政策金利・MPM 日程（黄色セル = 入力欄） |
| `BOJ→スポット` | M1-M8 → スポット OIS 変換（複利計算） |
| `スポット→BOJ` | スポット OIS → M1-M8 ブートストラップ（逐次逆算） |
| `利上げパス` | 会合ごとの利上げ幅 → 理論グリッドレート・スポット OIS |

**日付コンベンション**: ACT/365 Fixed、スポット T+2 東京営業日、MPM 発効日 = 発表日翌営業日、Modified Following。

デフォルト値の変更: `tools/ois_calc.py` 末尾の `DEFAULT_*` 変数を編集して再実行。

---

## ディレクトリ構成

```
BOJ_analysis/
├── data/                               # 生データ（非公開）
├── notebooks/
│   ├── feature_engineering_base.ipynb  # FE ベースライン確認・残差分析
│   ├── main_model.ipynb                # Outright モデル
│   └── model_comparison.ipynb          # RV Curve / Butterfly 比較
├── src/                                # 共通 Python ライブラリ
│   ├── processing.py                   # データ読み込み・MICE 補完
│   ├── modeling.py                     # walk-forward 検証・IC 評価
│   ├── features.py                     # Outright 特徴量・frac_diff
│   ├── features_rv.py                  # RV Curve / Butterfly 共通特徴量
│   ├── pooling.py                      # Outright プーリング
│   ├── pooling_curve.py                # RV Curve プーリング
│   └── pooling_butterfly.py            # RV Butterfly プーリング
├── designs/
│   ├── rv_butterfly_model.md           # メインモデル仕様（常に最新）
│   ├── rv_curve_model.md               # RV Curve 仕様
│   ├── current_model.md                # Outright 仕様
│   ├── experiment_log.md               # 全実験記録
│   └── pending/                        # 次回実験の設計書
├── tools/
│   └── ois_calc.py                     # OIS 計算ツール
├── trade_signal.py                     # 日次シグナル生成（メイン）
├── trade_signal_standalone.py          # スタンドアロン版（src/ 不要）
└── outputs/                            # 生成物（PDF / Excel）
```

---

## データスキーマ

`src/processing.py` が Reuters ティッカーを以下の列名にリネームする。

| Reuters ティッカー | 変換後 | 意味 |
|-------------------|--------|------|
| `JPBOJ{n}ONI=TRDT (MID_PRICE)` | `M1`〜`M8` | BOJ OIS（第 n 回会合先フォワードレート） |
| `JPY= (MID_PRICE)` | `USDJPY` | ドル円 |
| `JGBc1 (TRDPRC_1)` | `JGB_Future` | JGB 先物 |
| `.N225 (TRDPRC_1)` | `Nikkei225` | 日経 225 |
| `.DXY (TRDPRC_1)` | `DXY` | ドルインデックス |
| `JP12MONI=TRDT (BID)` | `T12` | 12M スポット OIS |
| `JP18MONI=TRDT (BID)` | `T18` | 18M スポット OIS |
| `JP24MONI=TRDT (BID)` | `T24` | 24M スポット OIS |

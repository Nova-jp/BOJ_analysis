"""
boj_distortion_standalone.xlsx に 2 シートを追加
  ③ スポット→BOJ  : 市場スポット OIS (1M-15M) → グリッドレート M1-M8 逆算
  ④ 利上げパス    : 会合ごとの利上げ幅入力 → 理論グリッド＋スポット曲線（2Y まで）

動作: Excel 単体で全セルが再計算される（Python 再実行不要）
使い方:
    source venv/bin/activate
    python tools/add_boj_sheets.py
"""

import os
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

# ─── ファイルパス ────────────────────────────────────────────────────────────
ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPATH = os.path.join(ROOT, "boj_distortion_standalone.xlsx")

# ─── スタイル定数 ────────────────────────────────────────────────────────────
_W = Font(name="Meiryo UI", size=10, bold=True,  color="FFFFFF")
_B = Font(name="Meiryo UI", size=10, bold=True)
_N = Font(name="Meiryo UI", size=10)
_S = Font(name="Meiryo UI", size= 9, color="595959")
_T = Font(name="Meiryo UI", size=13, bold=True,  color="FFFFFF")

_FI = PatternFill("solid", fgColor="FFF2CC")   # 黄: 入力
_FO = PatternFill("solid", fgColor="E2EFDA")   # 緑: 計算
_FH = PatternFill("solid", fgColor="2F5496")   # 濃紺: ヘッダー
_FS = PatternFill("solid", fgColor="D6E4F0")   # 薄青: セクション

_TH = Side(style="thin", color="BDD7EE")
_BR = Border(left=_TH, right=_TH, top=_TH, bottom=_TH)
_C  = Alignment(horizontal="center", vertical="center", wrap_text=True)
_L  = Alignment(horizontal="left",   vertical="center", wrap_text=False)

FMT_D   = "YYYY/MM/DD"
FMT_R   = "0.000%"
FMT_R4  = "0.0000%"
FMT_CF  = "0.000000"
FMT_I   = "0"
FMT_BP  = "0.0"

# ─── セル書き込みヘルパー ───────────────────────────────────────────────────
def _put(ws, r, c, val=None, font=None, fill=None,
         fmt=None, align=None, brd=True):
    cell = ws.cell(row=r, column=c)
    if val is not None:
        cell.value = val
    if font:  cell.font   = font
    if fill:  cell.fill   = fill
    if fmt:   cell.number_format = fmt
    cell.alignment = align or _L
    if brd:   cell.border = _BR
    return cell

def _inp(ws, r, c, val, fmt=FMT_R):
    return _put(ws, r, c, val, fill=_FI, fmt=fmt, align=_C)

def _out(ws, r, c, val, fmt=None):
    return _put(ws, r, c, val, fill=_FO, fmt=fmt, align=_C)

def _hdr(ws, r, labels, c=1, height=30):
    ws.row_dimensions[r].height = height
    for i, lbl in enumerate(labels):
        _put(ws, r, c+i, lbl, font=_W, fill=_FH, align=_C)

def _sec(ws, r, c, text, ncols=1):
    _put(ws, r, c, text, font=_B, fill=_FS, align=_L)
    if ncols > 1:
        ws.merge_cells(start_row=r, start_column=c,
                       end_row=r,   end_column=c+ncols-1)

def _lbl(ws, r, c, text, bold=False, align=None):
    return _put(ws, r, c, text, font=_B if bold else _N,
                align=align or _C)

def _note(ws, r, c, text):
    _put(ws, r, c, text, font=_S, brd=False)


# ─── 補間・計算式 ─────────────────────────────────────────────────────────────

def _interp(days_cell: str) -> str:
    """
    区分線形補間 (piecewise linear)
    data!$C$20:$C$32 の実日数 vs data!$B$20:$B$32 のスポット OIS から補間。
    LET 関数使用 (Excel 365 / 2021+)。
    """
    return (
        f"=LET("
        f"xs,data!$C$20:$C$32,"
        f"ys,data!$B$20:$B$32,"
        f"n,13,"
        f"Dk,{days_cell},"
        f"idx,MAX(1,MIN(n-1,IF(Dk>=INDEX(xs,n),n-1,MATCH(Dk,xs,1)))),"
        f"ylo,INDEX(ys,idx),"
        f"yhi,INDEX(ys,idx+1),"
        f"xlo,INDEX(xs,idx),"
        f"xhi,INDEX(xs,idx+1),"
        f"IF(xhi=xlo,ylo,ylo+(yhi-ylo)*(Dk-xlo)/(xhi-xlo)))"
    )

def _spot_from_grid(days_cell: str,
                    grid_rng="$G$6:$G$14",
                    start_rng="$H$6:$H$14",
                    pdays_rng="$I$6:$I$14") -> str:
    """
    スポット→BOJ シート用: 逆算グリッドレートからスポット OIS を再現。
    SUMPRODUCT(LN) で PRODUCT を計算 (Excel 配列の積)。
    9行目 (row 14) が flat 延長ダミー期間。
    """
    contrib = (f"1+{grid_rng}"
               f"*MIN(MAX({days_cell}-{start_rng},0),{pdays_rng})/365")
    return (f"=(EXP(SUMPRODUCT(LN({contrib})))-1)"
            f"/({days_cell}/365)")

def _spot_from_hike(days_cell: str) -> str:
    """
    利上げパス シート用: 理論グリッドレートからスポット OIS を計算。
    E19:E27 = grid rate (9行: M1-M8 data rows 19-26 + flat 延長 row 27)
    B19:B27 = 期間開始日数
    D19:D27 = 期間日数
    注意: row 18 はヘッダー行なので参照しないこと
    """
    contrib = ("1+$E$19:$E$27"
               f"*MIN(MAX({days_cell}-$B$19:$B$27,0),$D$19:$D$27)/365")
    return (f"=(EXP(SUMPRODUCT(LN({contrib})))-1)"
            f"/({days_cell}/365)")


# ─────────────────────────────────────────────────────────────────────────────
# Sheet ③: スポット→BOJ
# ─────────────────────────────────────────────────────────────────────────────

def add_spot_to_boj(wb: openpyxl.Workbook) -> None:
    if "スポット→BOJ" in wb.sheetnames:
        del wb["スポット→BOJ"]
    ws = wb.create_sheet("スポット→BOJ")
    ws.sheet_view.showGridLines = False

    for col, w in zip("ABCDEFGHIJ",
                      [8, 14, 9, 14, 12, 12, 14, 11, 10, 26]):
        ws.column_dimensions[col].width = w

    # ── タイトル ──────────────────────────────────────────────────────────────
    ws.merge_cells("A1:J1")
    _put(ws, 1, 1, "スポット OIS  →  BOJ グリッドレート 逆算（ブートストラップ）",
         font=_T, fill=_FH, align=_C)
    ws.row_dimensions[1].height = 30

    _note(ws, 2, 1,
          "【手法】市場スポット OIS（1M〜15M）を MPM 発効日で区分線形補間 → 8 点でブートストラップ"
          " → グリッドレート Mn 逆算。STEP 3 の残差で補間精度を確認。")

    # ── STEP 1/2 テーブル ─────────────────────────────────────────────────────
    _sec(ws, 4, 1, "STEP 1/2: MPM発効日でのスポット補間 → グリッドレート逆算（ACT/365 Fixed）", 10)
    _hdr(ws, 5, ["会合", "MPM発効日", "実日数\n(Dk)", "補間スポットOIS\n(Step1)",
                 "累積複利係数\n(Step2)", "期間複利係数\n(Step2)", "逆算Mn\n(Step2)",
                 "開始日数\n(H:helper)", "期間日数\n(I:helper)", "補間区間 (参考)"])

    for i in range(8):
        r = 6 + i          # rows 6-13
        dr = 9 + i         # data!C9:C16, data!D9:D16
        Dk = f"C{r}"       # days cell

        _lbl(ws, r, 1, f"M{i+1}", bold=True)

        # MPM 発効日（data シート参照）
        _out(ws, r, 2, f"=data!$C${dr}", FMT_D)

        # 実日数（data の累積日数列 D を使用）
        _out(ws, r, 3, f"=data!$D${dr}", FMT_I)

        # 補間スポット OIS
        _out(ws, r, 4, _interp(Dk), FMT_R4)

        # 累積複利係数: 1 + interp_spot * D_k / 365
        _out(ws, r, 5, f"=1+D{r}*C{r}/365", FMT_CF)

        # 期間複利係数
        if i == 0:
            _out(ws, r, 6, f"=E{r}", FMT_CF)   # M1: prev CF = 1
        else:
            _out(ws, r, 6, f"=E{r}/E{r-1}", FMT_CF)

        # 逆算グリッドレート Mn
        if i == 0:
            # M1: 期間 = [0, C6]、period_days = C6
            _out(ws, r, 7, f"=(F{r}-1)/(C{r}/365)", FMT_R4)
        else:
            # Mn: 期間 = [C{r-1}, C{r}]、period_days = C{r}-C{r-1}
            _out(ws, r, 7, f"=(F{r}-1)/((C{r}-C{r-1})/365)", FMT_R4)

        # Helper H: 期間開始日数
        if i == 0:
            _out(ws, r, 8, "=0", FMT_I)
        else:
            _out(ws, r, 8, f"=C{r-1}", FMT_I)

        # Helper I: 期間日数
        _out(ws, r, 9, f"=C{r}-H{r}", FMT_I)

        # 補間区間の説明（LET で 2 テナー名を取得）
        interp_label = (
            f'=LET(xs,data!$C$20:$C$32,ys,data!$D$20:$D$32,n,13,Dk,C{r},'
            f'idx,MAX(1,MIN(n-1,IF(Dk>=INDEX(xs,n),n-1,MATCH(Dk,xs,1)))),'
            f'INDEX(ys,idx)&"("&INDEX(xs,idx)&"d) ～ "&INDEX(ys,idx+1)&"("&INDEX(xs,idx+1)&"d)")'
        )
        _put(ws, r, 10, interp_label, font=_S, align=_L)

    # row 14: M8以降 flat 延長ダミー
    r14 = 14
    _put(ws, r14, 1, "M8以降\n(flat)", font=_S, align=_C)
    _out(ws, r14, 7, "=G13",   FMT_R4)  # M8 のグリッドレートを flat 延長
    _out(ws, r14, 8, "=C13",   FMT_I)   # 期間開始 = M8 の累積日数
    _out(ws, r14, 9, "=3650",  FMT_I)   # 十分大きい日数（~10年）
    _note(ws, r14, 10, "← M8 グリッドレートを flat 延長（12M・15M 残差検証用）")

    # ── STEP 3 残差検証テーブル ────────────────────────────────────────────────
    S3 = 16
    _sec(ws, S3, 1,
         "STEP 3: 残差検証 ─ 逆算グリッド → 再現スポット vs 市場スポット（補間精度・歪み確認）", 7)
    _hdr(ws, S3+1, ["テナー", "実日数", "市場スポット", "再現スポット\n(逆算グリッドから)",
                    "残差\n実際－再現 (bp)", "累積残差\n(bp)", "備考"])

    tenors = ["1M","2M","3M","4M","5M","6M","7M","8M","9M","10M","11M","12M","15M"]
    for j, tnr in enumerate(tenors):
        r = S3 + 2 + j        # rows 18-30
        dr = 20 + j           # data!C20:C32, data!B20:B32

        _lbl(ws, r, 1, tnr, bold=True)
        _out(ws, r, 2, f"=data!$C${dr}", FMT_I)

        # 市場スポット（参照元: data シート）
        _put(ws, r, 3, f"=data!$B${dr}", fill=_FI, fmt=FMT_R4, align=_C)

        # 再現スポット（G6:G14, H6:H14, I6:I14 を使用）
        _out(ws, r, 4, _spot_from_grid(f"B{r}"), FMT_R4)

        # 残差 (bp)
        _out(ws, r, 5, f"=(C{r}-D{r})*10000", FMT_BP)

        # 累積残差
        if j == 0:
            _out(ws, r, 6, f"=E{r}", FMT_BP)
        else:
            _out(ws, r, 6, f"=F{r-1}+E{r}", FMT_BP)

        # 備考
        note = "M8超→flat延長" if tnr in ("12M", "15M") else ""
        _put(ws, r, 7, note, font=_S, align=_L)

    # 注釈
    nr = S3 + 2 + len(tenors) + 1
    _note(ws, nr,   1, "【解釈】 残差 ≈ 0  →  市場スポットと BOJ グリッドの整合性が高い")
    _note(ws, nr+1, 1, "        残差が大きいテナー = 市場スポット曲線と BOJ 価格の乖離（歪み）")
    _note(ws, nr+2, 1, "        12M・15M は M8 発効日（約 358 日）を超えるため flat 延長で近似（誤差大きめ）")
    _note(ws, nr+3, 1, "【注意】 LET 関数は Excel 365 / Excel 2021 以降が必要です")


# ─────────────────────────────────────────────────────────────────────────────
# Sheet ④: 利上げパス
# ─────────────────────────────────────────────────────────────────────────────

def add_hike_path(wb: openpyxl.Workbook) -> None:
    if "利上げパス" in wb.sheetnames:
        del wb["利上げパス"]
    ws = wb.create_sheet("利上げパス")
    ws.sheet_view.showGridLines = False

    for col, w in zip("ABCDEFGHI",
                      [9, 14, 16, 16, 10, 14, 12, 14, 22]):
        ws.column_dimensions[col].width = w

    # ── タイトル ──────────────────────────────────────────────────────────────
    ws.merge_cells("A1:I1")
    _put(ws, 1, 1, "利上げパス  →  理論 BOJ グリッドレート・スポット OIS 曲線（2Y まで）",
         font=_T, fill=_FH, align=_C)
    ws.row_dimensions[1].height = 30

    _note(ws, 2, 1,
          "【使い方】 D5 に現在政策金利を入力。C7:C14（黄色セル）に各会合の利上げ幅（bp）を入力。"
          "0 = 据え置き、25 = 25bp 利上げ、-10 = 10bp 引下げ。")

    # ── 入力テーブル ─────────────────────────────────────────────────────────
    _sec(ws, 4, 1, "入力: 現在政策金利・各会合の利上げ幅", 4)
    _hdr(ws, 5, ["会合", "MPM発効日", "利上げ幅(bp)\n★ここを入力★", "期間中\n政策金利"])

    # 現在行（row 6）
    _lbl(ws, 6, 1, "現在", bold=True)
    _out(ws, 6, 2, "=data!$B$5", FMT_D)          # スポット日を表示
    _put(ws, 6, 3, "─", align=_C)
    _inp(ws, 6, 4, 0.00495, FMT_R4)              # ← 現在政策金利（黄色）

    # M1-M8（rows 7-14）
    for i in range(8):
        r  = 7 + i
        dr = 9 + i   # data!C9:C16

        _lbl(ws, r, 1, f"M{i+1}", bold=True)
        _out(ws, r, 2, f"=data!$C${dr}", FMT_D)  # MPM 発効日
        _inp(ws, r, 3, 0, "0")                    # 利上げ幅 (bp) ← 入力セル（黄色）

        # 期間中政策金利
        # M1: D7 = D6（現在政策金利、まだ会合前）
        # M2: D8 = D7 + C7/10000（M1 会合後の新レート）
        # M_n: D_{6+n} = D_{5+n} + C_{5+n}/10000
        if i == 0:
            _out(ws, r, 4, "=D6", FMT_R4)
        else:
            _out(ws, r, 4, f"=D{r-1}+C{r-1}/10000", FMT_R4)

    # M8 以降の政策金利（表示のみ）
    _lbl(ws, 15, 1, "M8後", bold=False)
    _out(ws, 15, 4, "=D14+C14/10000", FMT_R4)
    _note(ws, 15, 5, "← M8 会合後の最終政策金利")

    # ── 複利計算テーブル ──────────────────────────────────────────────────────
    _sec(ws, 17, 1, "理論グリッドレート・複利計算テーブル（ACT/365 Fixed）", 9)
    _hdr(ws, 18, ["会合", "期間開始\n日数", "期間終了\n日数", "期間\n日数",
                  "グリッドレート\n(期間中政策金利)", "期間複利係数",
                  "累積複利係数", "理論スポット OIS", "備考"])

    # M1-M8（rows 19-26）
    for i in range(8):
        r  = 19 + i
        dr = 9 + i    # data!D9:D16 = 累積日数
        ir = 7 + i    # 入力テーブルの政策金利セル D7:D14

        _lbl(ws, r, 1, f"M{i+1}", bold=True)

        # 期間開始日数
        if i == 0:
            _out(ws, r, 2, "=0", FMT_I)
        else:
            _out(ws, r, 2, f"=C{r-1}", FMT_I)

        # 期間終了日数（data の累積日数）
        _out(ws, r, 3, f"=data!$D${dr}", FMT_I)

        # 期間日数
        _out(ws, r, 4, f"=C{r}-B{r}", FMT_I)

        # グリッドレート = 入力テーブルの期間中政策金利
        _out(ws, r, 5, f"=$D${ir}", FMT_R4)

        # 期間複利係数
        _out(ws, r, 6, f"=1+E{r}*D{r}/365", FMT_CF)

        # 累積複利係数
        if i == 0:
            _out(ws, r, 7, f"=F{r}", FMT_CF)
        else:
            _out(ws, r, 7, f"=G{r-1}*F{r}", FMT_CF)

        # 理論スポット OIS（この期末まで）
        _out(ws, r, 8, f"=(G{r}-1)/(C{r}/365)", FMT_R4)

    # row 27: M8以降 flat 延長
    r27 = 27
    _put(ws, r27, 1, "M8以降\n(flat)", font=_S, align=_C)
    _out(ws, r27, 2, "=C26", FMT_I)          # 開始日数 = M8 終了日数
    _out(ws, r27, 3, "=C26+3650", FMT_I)     # 終了 = 大きな数（~10 年先）
    _out(ws, r27, 4, "=3650", FMT_I)         # 期間日数
    _out(ws, r27, 5, "=$D$15", FMT_R4)       # M8 後の政策金利 flat 延長
    _out(ws, r27, 6, "=1+E27*D27/365", FMT_CF)
    _note(ws, r27, 9, "← 2Y（730d）等の長テナー計算用 flat 延長ダミー")

    # ── 理論スポット OIS 曲線 ─────────────────────────────────────────────────
    _sec(ws, 29, 1, "理論スポット OIS 曲線（1M〜2Y）vs 市場スポット OIS", 9)
    _hdr(ws, 30, ["テナー", "実日数\n(概算)", "理論スポット OIS",
                  "市場スポット OIS\n(参考: data シート)", "差分\n理論－市場 (bp)",
                  "", "", "", ""])

    # テナー定義: (ラベル, EDATE月数, data 行 or None)
    tenor_defs = [
        ("1M",  1, 20), ("2M",  2, 21), ("3M",  3, 22),
        ("4M",  4, 23), ("5M",  5, 24), ("6M",  6, 25),
        ("7M",  7, 26), ("8M",  8, 27), ("9M",  9, 28),
        ("10M",10, 29), ("11M",11, 30), ("12M",12, 31),
        ("15M",15, 32), ("18M",18, None), ("21M",21, None), ("2Y", 24, None),
    ]

    for j, (tnr, months, drow) in enumerate(tenor_defs):
        r = 31 + j   # rows 31-46

        _lbl(ws, r, 1, tnr, bold=True)

        # 実日数: EDATE 使用（スポット日から何日後か）
        _out(ws, r, 2, f"=EDATE(data!$B$5,{months})-data!$B$5", FMT_I)

        # 理論スポット OIS（SUMPRODUCT(LN) 法）
        _out(ws, r, 3, _spot_from_hike(f"B{r}"), FMT_R4)

        # 市場スポット OIS（data シートにある場合のみ）
        if drow is not None:
            _out(ws, r, 4, f"=data!$B${drow}", FMT_R4)
            # 差分 (bp)
            _out(ws, r, 5, f"=(C{r}-D{r})*10000", FMT_BP)
        else:
            _put(ws, r, 4, "─", align=_C)
            _put(ws, r, 5, "─", align=_C)

    # 注釈
    nr = 31 + len(tenor_defs) + 1
    _note(ws, nr,   1, "【差分の解釈】")
    _note(ws, nr+1, 1, "  正値 (+bp): 利上げパス想定より市場は緩和的（ハト派的）")
    _note(ws, nr+2, 1, "  負値 (−bp): 利上げパス想定より市場は積極的（タカ派的）")
    _note(ws, nr+3, 1, "  2Y（18M〜）は市場クォートなし。理論値のみ表示。")
    _note(ws, nr+4, 1, "【前提】 確率論的不確実性なし（パスの確実実現を仮定）。")
    _note(ws, nr+5, 1, "        タームプレミアム・凸性調整は含まず。")
    _note(ws, nr+6, 1, "【注意】 LET 関数不使用のため Excel 2016 以降で動作します。")


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(FPATH):
        raise FileNotFoundError(f"ファイルが見つかりません: {FPATH}")

    print(f"読み込み中: {FPATH}")
    wb = openpyxl.load_workbook(FPATH)

    print("  シート③ 追加: スポット→BOJ")
    add_spot_to_boj(wb)

    print("  シート④ 追加: 利上げパス")
    add_hike_path(wb)

    wb.save(FPATH)
    print(f"保存完了: {FPATH}")
    print()
    print("  追加したシート:")
    print("    スポット→BOJ  ─ 市場スポット OIS (1M-15M) から M1-M8 グリッドレートを逆算")
    print("                    黄色セル: なし（全セル自動計算。元データは data シートを直接参照）")
    print()
    print("    利上げパス    ─ 各会合の利上げ幅を指定 → 理論スポット OIS 曲線（2Y まで）")
    print("                    黄色セル: D5=現在政策金利, C7:C14=各会合の利上げ幅(bp)")
    print()
    print("  動作確認:")
    print("    Excel で開いて「数式の更新」が走れば完了。")
    print("    C7:C14 に利上げ幅を入れると即座に全テナーの理論スポット OIS が更新されます。")

if __name__ == "__main__":
    main()

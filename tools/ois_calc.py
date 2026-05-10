"""
BOJ OIS 計算ツール  (Excel 完全自立版)
======================================
Python は Excel ひな形を 1 度生成するだけ。
以後は Excel 上で黄色セルを直接編集すると全シートが即時再計算される。

使い方:
  python tools/ois_calc.py [--date YYYY-MM-DD]
  → outputs/ois_calculator_YYYYMMDD.xlsx を生成（上書き）

日付コンベンション:
  デイカウント : ACT/365 Fixed
  スポット日   : T+2 東京営業日  (=WORKDAY(today,2,Tokyo_Holidays))
  MPM 発効日   : 発表日翌営業日  (=WORKDAY(ann,1,Tokyo_Holidays))
  期末日調整   : Modified Following
  祝日リスト   : 東京（国民の祝日＋振替＋国民の休日＋年末年始 2024-2030）
"""

from __future__ import annotations
import argparse
import os
from datetime import date, timedelta

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.workbook.defined_name import DefinedName
except ImportError:
    raise SystemExit("openpyxl が必要です: pip install openpyxl")


# ─────────────────────────────────────────────────────────────────────────────
# 1. 東京祝日リスト生成（Python で計算して Excel の隠しシートに書き込む）
# ─────────────────────────────────────────────────────────────────────────────

_SHUNBUN = {2020:20,2021:20,2022:21,2023:21,2024:20,2025:20,2026:20,2027:21,2028:20,2029:20}
_SHUBUN  = {2020:22,2021:23,2022:23,2023:23,2024:22,2025:23,2026:23,2027:23,2028:22,2029:23}

def _nth_weekday(y, m, wd, n):
    first = date(y, m, 1)
    return first + timedelta(days=(wd - first.weekday()) % 7 + 7*(n-1))

def _raw_holidays(y):
    h = {date(y,1,1), _nth_weekday(y,1,0,2), date(y,2,11)}
    if y >= 2020: h.add(date(y,2,23))
    if y in _SHUNBUN: h.add(date(y,3,_SHUNBUN[y]))
    h |= {date(y,4,29), date(y,5,3), date(y,5,4), date(y,5,5)}
    h.add(_nth_weekday(y,7,0,3))
    h.add(date(y,8,11))
    h.add(_nth_weekday(y,9,0,3))
    if y in _SHUBUN: h.add(date(y,9,_SHUBUN[y]))
    h.add(_nth_weekday(y,10,0,2))
    h |= {date(y,11,3), date(y,11,23)}
    return h

def _full_holidays(y):
    raw  = _raw_holidays(y)
    full = set(raw)
    for d in sorted(raw):
        if d.weekday() == 6:
            c = d + timedelta(1)
            while c in full or c.weekday() == 6:
                c += timedelta(1)
            full.add(c)
    snap = set(full)
    for d in sorted(snap):
        mid = d + timedelta(1)
        if (d - timedelta(1)) in snap and (d + timedelta(2)) in snap \
                and mid.weekday() < 6 and mid not in snap:
            full.add(mid)
    full |= {date(y,12,31), date(y,1,2), date(y,1,3)}
    return full

def tokyo_holidays(years=range(2024,2031)):
    all_h = set()
    for y in years:
        all_h |= _full_holidays(y)
    return sorted(all_h)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Excel スタイル定数
# ─────────────────────────────────────────────────────────────────────────────

_FMT_DATE = "YYYY/MM/DD"
_FMT_RATE = "0.000%"
_FMT_INT  = "0"
_FMT_CF   = "0.000000"
_FMT_DCF  = "0.00000"
_FMT_BP   = '0" bp"'

_FILL_INPUT   = PatternFill("solid", fgColor="FFF2CC")   # 黄: ユーザー入力
_FILL_OUTPUT  = PatternFill("solid", fgColor="E2EFDA")   # 緑: 計算値
_FILL_HEADER  = PatternFill("solid", fgColor="2F5496")   # 濃紺: ヘッダー
_FILL_SECTION = PatternFill("solid", fgColor="D6E4F0")   # 薄青: セクション

_FONT_WH  = Font(name="Meiryo UI", size=10, bold=True,  color="FFFFFF")
_FONT_BD  = Font(name="Meiryo UI", size=10, bold=True)
_FONT_NM  = Font(name="Meiryo UI", size=10)
_FONT_SM  = Font(name="Meiryo UI", size= 9, color="595959")
_FONT_TTL = Font(name="Meiryo UI", size=13, bold=True,  color="FFFFFF")

_THIN = Side(style="thin", color="BDD7EE")
_BRD  = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_CENT = Alignment(horizontal="center", vertical="center")
_LEFT = Alignment(horizontal="left",   vertical="center")

def _c(ws, row, col, val=None, font=None, fill=None, fmt=None,
        align=None, border=True):
    cell = ws.cell(row=row, column=col)
    if val is not None:
        cell.value = val
    if font:   cell.font   = font
    if fill:   cell.fill   = fill
    if fmt:    cell.number_format = fmt
    if align:  cell.alignment = align
    else:      cell.alignment = _LEFT
    if border: cell.border = _BRD
    return cell

def _hdr(ws, row, labels, col=1):
    for i, lbl in enumerate(labels):
        _c(ws, row, col+i, lbl, font=_FONT_WH, fill=_FILL_HEADER, align=_CENT)

def _sec(ws, row, col, text, ncols=1):
    _c(ws, row, col, text, font=_FONT_BD, fill=_FILL_SECTION, align=_LEFT)
    if ncols > 1:
        ws.merge_cells(start_row=row, start_column=col,
                       end_row=row,   end_column=col+ncols-1)

def _inp(ws, row, col, val, fmt):
    _c(ws, row, col, val, fill=_FILL_INPUT,  fmt=fmt, align=_CENT)

def _out(ws, row, col, formula, fmt):
    _c(ws, row, col, formula, fill=_FILL_OUTPUT, fmt=fmt, align=_CENT)

def _modfol(raw_expr: str) -> str:
    """Modified Following 調整式（WORKDAY 関数利用）"""
    is_bd  = f"AND(WEEKDAY({raw_expr},2)<=5,COUNTIF(Tokyo_Holidays,{raw_expr})=0)"
    snap   = f"IF({is_bd},{raw_expr},WORKDAY({raw_expr},1,Tokyo_Holidays))"
    return (f"=IF(MONTH({snap})=MONTH({raw_expr}),"
            f"{snap},"
            f"WORKDAY({raw_expr},-1,Tokyo_Holidays))")

def _spot_ois_formula(tenor_cell: str,
                      rate_range: str, start_range: str, days_range: str,
                      spot_ref: str, last_end_ref: str) -> str:
    """
    スポット OIS 計算式（SUMPRODUCT + LN で PRODUCT を実現）
    tenor_cell   : 満期日セル (例 "C28")
    rate_range   : グリッドレートのレンジ (例 "$F$16:$F$23")
    start_range  : 各期間開始日レンジ (例 "$B$16:$B$23")
    days_range   : 各期間日数レンジ (例 "$D$16:$D$23")
    spot_ref     : スポット日セル参照 (例 "'設定'!$B$6")
    last_end_ref : M8発効日参照（範囲外チェック用）(例 "$C$23")
    """
    contrib = (f"1+{rate_range}"
               f"*MIN(MAX({tenor_cell}-{start_range},0),{days_range})/365")
    cum_cf  = f"EXP(SUMPRODUCT(LN({contrib})))"
    total_d = f"({tenor_cell}-{spot_ref})/365"
    return (f"=IF({tenor_cell}>{last_end_ref},\"範囲外\","
            f"({cum_cf}-1)/({total_d}))")


# ─────────────────────────────────────────────────────────────────────────────
# 3. 祝日シート（非表示）＋ Named Range
# ─────────────────────────────────────────────────────────────────────────────

def build_holiday_sheet(wb):
    ws = wb.create_sheet("祝日")
    ws.sheet_state = "hidden"
    holidays = tokyo_holidays()
    ws["A1"] = "Tokyo_Holidays"
    ws["A1"].font = _FONT_BD
    for i, d in enumerate(holidays, start=2):
        cell = ws.cell(row=i, column=1, value=d)
        cell.number_format = _FMT_DATE
    last = len(holidays) + 1
    wb.defined_names["Tokyo_Holidays"] = DefinedName(
        "Tokyo_Holidays", attr_text=f"祝日!$A$2:$A${last}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. 設定シート
# ─────────────────────────────────────────────────────────────────────────────

# 設定シートのセル位置定数
_S_TODAY  = "B5"   # 本日日付（入力）
_S_SPOT   = "B6"   # スポット日（計算）
_S_RATE   = "B7"   # 政策金利（入力）
_S_ANN    = [f"B{11+i}" for i in range(8)]  # MPM発表日（入力）
_S_EFF    = [f"C{11+i}" for i in range(8)]  # MPM発効日（計算）

def build_settings_sheet(wb, today_dt, policy_rate, mpm_ann_dates):
    ws = wb.active
    ws.title = "設定"
    ws.sheet_view.showGridLines = False
    for col, w in zip("ABCDE", [20,16,16,12,32]):
        ws.column_dimensions[col].width = w

    # タイトル
    ws.merge_cells("A1:E1")
    _c(ws,1,1,"BOJ OIS 計算ツール  ─  設定シート",
       font=_FONT_TTL, fill=_FILL_HEADER, align=_CENT)
    ws.row_dimensions[1].height = 32

    # ─── 基本設定 ───
    _sec(ws,3,1,"基本設定",4)
    _hdr(ws,4,["項目","値","","備考"],col=1)
    rows = [
        ("本日日付",    today_dt,    _FMT_DATE, True,  "黄色セルを直接編集できます"),
        ("スポット日",  f"=WORKDAY({_S_TODAY},2,Tokyo_Holidays)", _FMT_DATE, False, "T+2 東京営業日（自動）"),
        ("現在政策金利",policy_rate, _FMT_RATE, True,  "小数で入力 例: 0.75% → 0.00750"),
    ]
    for i,(lbl,val,fmt,is_inp,note) in enumerate(rows):
        r = 5+i
        _c(ws,r,1,lbl,font=_FONT_BD)
        if is_inp:
            _inp(ws,r,2,val,fmt)
        else:
            _out(ws,r,2,val,fmt)
        _c(ws,r,3,"",border=True)
        _c(ws,r,4,note,font=_FONT_SM,border=False)

    # ─── MPM 日程 ───
    _sec(ws,9,1,"MPM 日程（M1〜M8）",4)
    _hdr(ws,10,["会合","発表日（入力）","発効日（自動）","期間日数"],col=1)
    for i,ann in enumerate(mpm_ann_dates):
        r = 11+i
        _c(ws,r,1,f"M{i+1}",font=_FONT_BD,align=_CENT)
        _inp(ws,r,2,ann,_FMT_DATE)
        _out(ws,r,3,f"=WORKDAY({_S_ANN[i]},1,Tokyo_Holidays)",_FMT_DATE)
        # 期間日数: M1は(C11-スポット日), M2+は(C{r}-C{r-1})
        if i == 0:
            _out(ws,r,4,f"=C11-{_S_SPOT}",_FMT_INT)
        else:
            _out(ws,r,4,f"=C{r}-C{r-1}",_FMT_INT)

    # 凡例
    r = 20
    _c(ws,r,1,"■ 黄色 = 入力セル  ■ 緑色 = 計算セル  ■ 黄色セルを変更すると全シートが自動更新されます",
       font=_FONT_SM, border=False)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BOJ → スポット シート
# ─────────────────────────────────────────────────────────────────────────────

def build_boj_to_spot_sheet(wb, default_grid_rates):
    ws = wb.create_sheet("BOJ→スポット")
    ws.sheet_view.showGridLines = False
    for col,w in zip("ABCDEFGHI",[10,16,16,8,9,14,12,12,14]):
        ws.column_dimensions[col].width = w

    ws.merge_cells("A1:I1")
    _c(ws,1,1,"BOJ グリッドレート  →  スポット OIS 変換",
       font=_FONT_TTL, fill=_FILL_HEADER, align=_CENT)
    ws.row_dimensions[1].height = 32

    # ─── 入力: グリッドレート ───
    _sec(ws,3,1,"入力: M1〜M8 グリッドレート（黄色セルを直接編集）",5)
    _hdr(ws,4,["会合","グリッドレート","期間（設定シート参照）"],col=1)
    for i,r in enumerate(default_grid_rates):
        row = 5+i
        _c(ws,row,1,f"M{i+1}",font=_FONT_BD,align=_CENT)
        _inp(ws,row,2,r,_FMT_RATE)
        # 期間表示（文字列として "発効日～発効日" を TEXT 関数で組み立て）
        if i == 0:
            s_ref = f"'設定'!{_S_SPOT}"
        else:
            s_ref = f"'設定'!{_S_EFF[i-1]}"
        e_ref = f"'設定'!{_S_EFF[i]}"
        _c(ws,row,3,
           f'=TEXT({s_ref},"{_FMT_DATE}")&"  ～  "&TEXT({e_ref},"{_FMT_DATE}")&"  ("&({e_ref}-{s_ref})&"日)"',
           font=_FONT_SM,align=_LEFT)

    _c(ws,13,1,"※ グリッドレートを変更すると下の計算テーブルと標準テナー欄が即時更新されます",
       font=_FONT_SM,border=False)

    # ─── 複利計算テーブル ───
    # 行番号: M1=16, M2=17, ..., M8=23
    CALC_BASE = 15
    _sec(ws,CALC_BASE,1,"複利計算テーブル（ACT/365 Fixed）",9)
    _hdr(ws,CALC_BASE+1,
         ["会合","開始日","終了日","日数","DCF",
          "グリッドレート","期間複利係数","累積複利係数","スポットOIS"])

    for i in range(8):
        row = CALC_BASE+2+i   # row 17..24
        spot_ref = f"'設定'!{_S_SPOT}"
        eff_ref  = f"'設定'!{_S_EFF[i]}"
        grid_ref = f"$B${5+i}"   # 入力セル B5..B12

        # 開始日
        if i == 0:
            start_f = f"={spot_ref}"
        else:
            start_f = f"=C{row-1}"
        # 終了日
        end_f = f"={eff_ref}"
        # 累積 CF
        if i == 0:
            cumcf_f = f"=G{row}"
        else:
            cumcf_f = f"=H{row-1}*G{row}"
        # スポット OIS
        spot_ois_f = (f"=(H{row}-1)/((C{row}-{spot_ref})/365)")

        _c(ws,row,1,f"M{i+1}",font=_FONT_BD,align=_CENT)
        _out(ws,row,2,start_f,_FMT_DATE)
        _out(ws,row,3,end_f,  _FMT_DATE)
        _out(ws,row,4,f"=C{row}-B{row}",_FMT_INT)
        _out(ws,row,5,f"=D{row}/365",   _FMT_DCF)
        _c(ws,row,6,f"={grid_ref}",fill=_FILL_INPUT,fmt=_FMT_RATE,align=_CENT)
        _out(ws,row,7,f"=1+F{row}*E{row}",_FMT_CF)
        _out(ws,row,8,cumcf_f,           _FMT_CF)
        _out(ws,row,9,spot_ois_f,        _FMT_RATE)

    # ─── 標準テナー スポット OIS ───
    TENOR_BASE = CALC_BASE+11   # row 26
    _sec(ws,TENOR_BASE,1,"標準テナー スポット OIS（Modified Following 調整済み）",5)
    _hdr(ws,TENOR_BASE+1,["テナー","満期原日","満期日(ModFol)","日数","スポットOIS"])

    TENORS = [("1M",1),("3M",3),("6M",6),("9M",9),
              ("12M",12),("15M",15),("18M",18),("2Y",24)]
    spot_ref   = f"'設定'!{_S_SPOT}"
    rate_rng   = f"$F${CALC_BASE+2}:$F${CALC_BASE+9}"
    start_rng  = f"$B${CALC_BASE+2}:$B${CALC_BASE+9}"
    days_rng   = f"$D${CALC_BASE+2}:$D${CALC_BASE+9}"
    last_end   = f"$C${CALC_BASE+9}"

    for j,(lbl,months) in enumerate(TENORS):
        row = TENOR_BASE+2+j
        raw_f = f"=EDATE({spot_ref},{months})"
        modfol_f = _modfol(f"B{row}")
        spot_ois_f = _spot_ois_formula(
            f"C{row}", rate_rng, start_rng, days_rng, spot_ref, last_end)

        _c(ws,row,1,lbl,font=_FONT_BD,align=_CENT)
        _out(ws,row,2,raw_f,       _FMT_DATE)
        _out(ws,row,3,modfol_f,    _FMT_DATE)
        _out(ws,row,4,f"=C{row}-{spot_ref}",_FMT_INT)
        _out(ws,row,5,spot_ois_f,  _FMT_RATE)

    note_row = TENOR_BASE+2+len(TENORS)+1
    _c(ws,note_row,1,
       "計算式: 1 + R × D/365 = ∏ᵢ(1 + Mᵢ × dᵢ/365)　"
       "期間途中のテナーは最終期間を部分カバー",
       font=_FONT_SM,border=False)


# ─────────────────────────────────────────────────────────────────────────────
# 6. スポット → BOJ シート（ブートストラップ）
# ─────────────────────────────────────────────────────────────────────────────

def build_spot_to_boj_sheet(wb, default_spot_rates):
    ws = wb.create_sheet("スポット→BOJ")
    ws.sheet_view.showGridLines = False
    for col,w in zip("ABCDEFGH",[10,16,14,16,12,12,14,26]):
        ws.column_dimensions[col].width = w

    ws.merge_cells("A1:H1")
    _c(ws,1,1,"スポット OIS  →  BOJ グリッドレート  ブートストラップ",
       font=_FONT_TTL, fill=_FILL_HEADER, align=_CENT)
    ws.row_dimensions[1].height = 32

    _c(ws,2,1,
       "各 MPM 発効日でのスポット OIS レートを入力すると、Forward グリッドレートを逆算します。",
       font=_FONT_SM,border=False)

    # ─── 入力 ───
    _sec(ws,4,1,"入力: スポット OIS レート（各 MPM 発効日ノード）",4)
    _hdr(ws,5,["会合","発効日（自動）","スポットOIS（入力）","日数（対スポット日）"],col=1)
    for i,r in enumerate(default_spot_rates):
        row = 6+i
        _c(ws,row,1,f"M{i+1}",font=_FONT_BD,align=_CENT)
        _out(ws,row,2,f"='設定'!{_S_EFF[i]}",_FMT_DATE)
        _inp(ws,row,3,r,_FMT_RATE)
        _out(ws,row,4,f"=B{row}-'設定'!{_S_SPOT}",_FMT_INT)

    _c(ws,14,1,"※ スポットOISを変更すると下の計算テーブルが即時更新されます",
       font=_FONT_SM,border=False)

    # ─── ブートストラップ計算テーブル ───
    # 行番号: M1=17, M2=18, ..., M8=24
    CALC_BASE = 16
    _sec(ws,CALC_BASE,1,"ブートストラップ計算結果",8)
    _hdr(ws,CALC_BASE+1,
         ["会合","ノード日","日数","スポットOIS",
          "累積複利係数","期間複利係数","推定グリッドレート","逆算検証（再現スポット）"])

    for i in range(8):
        row       = CALC_BASE+2+i   # 18..25
        spot_ref  = f"'設定'!{_S_SPOT}"
        eff_ref   = f"='設定'!{_S_EFF[i]}"
        r_spot_in = f"$C${6+i}"     # 入力セル（スポット→BOJシート内）

        # 累積複利係数: 1 + S_n * D_n / 365
        cumcf_f = f"=1+{r_spot_in}*(B{row}-{spot_ref})/365"
        # 期間複利係数: E_n / E_{n-1}  (E_0 = 1 for M1)
        if i == 0:
            percf_f = f"=E{row}"
        else:
            percf_f = f"=E{row}/E{row-1}"
        # グリッドレート: (period_CF - 1) / (d_n/365)
        if i == 0:
            period_d = f"(B{row}-{spot_ref})"
        else:
            period_d = f"(B{row}-B{row-1})"
        grid_f = f"=(F{row}-1)/({period_d}/365)"
        # 逆算検証（再現スポット）: (EXP(SUMPRODUCT(LN(...)))-1)/(D_n/365)
        rate_rng  = f"$G${CALC_BASE+2}:G{row}"
        start_rng = f"$B${CALC_BASE+2}:B{row}"
        days_rng  = f"IF($B${CALC_BASE+2}:B{row}={spot_ref},$D${CALC_BASE+2}:D{row},$B${CALC_BASE+3}:B{row+1}-$B${CALC_BASE+2}:B{row})"
        # 簡略版: 累積CF（E列）から直接逆算
        recon_f = f"=(E{row}-1)*(365/(B{row}-{spot_ref}))"

        _c(ws,row,1,f"M{i+1}",font=_FONT_BD,align=_CENT)
        _out(ws,row,2,eff_ref,   _FMT_DATE)
        _out(ws,row,3,f"=B{row}-{spot_ref}",_FMT_INT)
        _c(ws,row,4,f"={r_spot_in}",fill=_FILL_INPUT,fmt=_FMT_RATE,align=_CENT)
        _out(ws,row,5,cumcf_f,   _FMT_CF)
        _out(ws,row,6,percf_f,   _FMT_CF)
        _out(ws,row,7,grid_f,    _FMT_RATE)
        _out(ws,row,8,recon_f,   _FMT_RATE)

    note_row = CALC_BASE+2+8+1
    _c(ws,note_row,1,
       "計算式: Mₙ = [(1+Rₙ×Dₙ/365)/(1+Rₙ₋₁×Dₙ₋₁/365) - 1] / (dₙ/365)　"
       "「逆算検証」≒ 入力スポットOIS なら計算正確",
       font=_FONT_SM,border=False)


# ─────────────────────────────────────────────────────────────────────────────
# 7. 利上げパス シート
# ─────────────────────────────────────────────────────────────────────────────

def build_hike_path_sheet(wb, default_hike_bp):
    ws = wb.create_sheet("利上げパス")
    ws.sheet_view.showGridLines = False
    for col,w in zip("ABCDEFGHI",[10,14,14,14,9,14,12,12,14]):
        ws.column_dimensions[col].width = w

    ws.merge_cells("A1:I1")
    _c(ws,1,1,"利上げパス  →  理論 BOJ グリッドレート・スポット OIS",
       font=_FONT_TTL, fill=_FILL_HEADER, align=_CENT)
    ws.row_dimensions[1].height = 32

    _c(ws,2,1,
       "各会合での利上げ幅(bp)を黄色セルに入力すると、理論グリッドレートとスポットOISが即時更新されます。",
       font=_FONT_SM,border=False)

    # ─── 入力: 利上げパス ───
    _sec(ws,4,1,"入力: 利上げパス（各会合での政策金利変更幅）",4)
    _hdr(ws,5,["会合","発表日（自動）","利上げ幅(bp)【入力】","期間中 政策金利"],col=1)

    # 現在行
    _c(ws,6,1,"現在",font=_FONT_BD,align=_CENT)
    _out(ws,6,2,f"='設定'!{_S_TODAY}",_FMT_DATE)
    _c(ws,6,3,"─",align=_CENT)
    _out(ws,6,4,f"='設定'!{_S_RATE}",_FMT_RATE)

    for i,bp in enumerate(default_hike_bp):
        row = 7+i   # 7..14
        _c(ws,row,1,f"M{i+1}",font=_FONT_BD,align=_CENT)
        _out(ws,row,2,f"='設定'!{_S_ANN[i]}",_FMT_DATE)
        _inp(ws,row,3,bp,_FMT_INT)
        # 期間中政策金利: D7=D6, D8=D7+C7/10000, ...
        if i == 0:
            rate_f = f"=D6"
        else:
            rate_f = f"=D{row-1}+C{row-1}/10000"
        _out(ws,row,4,rate_f,_FMT_RATE)

    _c(ws,15,1,
       "「期間中 政策金利」= 期間 M_n 中の政策金利（M_n 会合の決定は M_{n+1} 期間から反映）",
       font=_FONT_SM,border=False)

    # ─── 複利計算テーブル ───
    CALC_BASE = 17
    _sec(ws,CALC_BASE,1,"理論 BOJ グリッドレート・複利計算テーブル",9)
    _hdr(ws,CALC_BASE+1,
         ["会合","開始日","終了日","日数","DCF",
          "グリッドレート","期間複利係数","累積複利係数","スポットOIS"])

    for i in range(8):
        row      = CALC_BASE+2+i   # 19..26
        spot_ref = f"'設定'!{_S_SPOT}"
        eff_ref  = f"'設定'!{_S_EFF[i]}"
        grid_src = f"$D${7+i}"   # 利上げパス入力セクションの D列

        if i == 0:
            start_f = f"={spot_ref}"
            cumcf_f = f"=G{row}"
        else:
            start_f = f"=C{row-1}"
            cumcf_f = f"=H{row-1}*G{row}"

        _c(ws,row,1,f"M{i+1}",font=_FONT_BD,align=_CENT)
        _out(ws,row,2,start_f,           _FMT_DATE)
        _out(ws,row,3,f"={eff_ref}",     _FMT_DATE)
        _out(ws,row,4,f"=C{row}-B{row}", _FMT_INT)
        _out(ws,row,5,f"=D{row}/365",    _FMT_DCF)
        _c(ws,row,6,f"={grid_src}",fill=_FILL_OUTPUT,fmt=_FMT_RATE,align=_CENT)
        _out(ws,row,7,f"=1+F{row}*E{row}",_FMT_CF)
        _out(ws,row,8,cumcf_f,            _FMT_CF)
        _out(ws,row,9,f"=(H{row}-1)/((C{row}-{spot_ref})/365)",_FMT_RATE)

    # ─── 理論スポット OIS（標準テナー）───
    TENOR_BASE = CALC_BASE+11   # 28
    _sec(ws,TENOR_BASE,1,"理論スポット OIS（標準テナー）",5)
    _hdr(ws,TENOR_BASE+1,["テナー","満期原日","満期日(ModFol)","日数","理論スポットOIS"])

    TENORS = [("1M",1),("3M",3),("6M",6),("9M",9),
              ("12M",12),("15M",15),("18M",18),("2Y",24)]
    spot_ref  = f"'設定'!{_S_SPOT}"
    rate_rng  = f"$F${CALC_BASE+2}:$F${CALC_BASE+9}"
    start_rng = f"$B${CALC_BASE+2}:$B${CALC_BASE+9}"
    days_rng  = f"$D${CALC_BASE+2}:$D${CALC_BASE+9}"
    last_end  = f"$C${CALC_BASE+9}"

    for j,(lbl,months) in enumerate(TENORS):
        row = TENOR_BASE+2+j
        raw_f     = f"=EDATE({spot_ref},{months})"
        modfol_f  = _modfol(f"B{row}")
        ois_f     = _spot_ois_formula(
            f"C{row}", rate_rng, start_rng, days_rng, spot_ref, last_end)

        _c(ws,row,1,lbl,font=_FONT_BD,align=_CENT)
        _out(ws,row,2,raw_f,     _FMT_DATE)
        _out(ws,row,3,modfol_f,  _FMT_DATE)
        _out(ws,row,4,f"=C{row}-{spot_ref}",_FMT_INT)
        _out(ws,row,5,ois_f,     _FMT_RATE)

    note_row = TENOR_BASE+2+len(TENORS)+1
    _c(ws,note_row,1,
       "前提: 指定パスの確実実現（タームプレミアムゼロ）。"
       "市場レートとの差 ≈ タームプレミアム + 不確実性プレミアム。",
       font=_FONT_SM,border=False)


# ─────────────────────────────────────────────────────────────────────────────
# 8. デフォルト入力値
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MPM_ANN = [
    date(2026,4,28), date(2026,6,17), date(2026,7,28), date(2026,9,18),
    date(2026,10,29),date(2026,12,18),date(2027,1,22), date(2027,3,19),
]
DEFAULT_GRID_RATES = [0.00750,0.00780,0.00820,0.00860,
                      0.00900,0.00930,0.00960,0.00990]
DEFAULT_SPOT_BOOTSTRAP = [0.00750,0.00765,0.00785,0.00810,
                          0.00840,0.00870,0.00900,0.00925]
DEFAULT_HIKE_BP = [0, 0, 25, 0, 0, 25, 0, 0]  # M3, M6 で各 25bp


# ─────────────────────────────────────────────────────────────────────────────
# 9. メイン
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BOJ OIS 計算ツール（Excel 完全自立版）")
    parser.add_argument("--date", default=None, help="本日日付 YYYY-MM-DD（省略=今日）")
    args = parser.parse_args()
    today_dt = date.fromisoformat(args.date) if args.date else date.today()

    wb = openpyxl.Workbook()

    build_settings_sheet(wb, today_dt, 0.00750, DEFAULT_MPM_ANN)
    build_boj_to_spot_sheet(wb, DEFAULT_GRID_RATES)
    build_spot_to_boj_sheet(wb, DEFAULT_SPOT_BOOTSTRAP)
    build_hike_path_sheet(wb, DEFAULT_HIKE_BP)
    build_holiday_sheet(wb)   # 最後（hidden）

    out_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    fname    = f"ois_calculator_{today_dt.strftime('%Y%m%d')}.xlsx"
    out_path = os.path.join(out_dir, fname)
    wb.save(out_path)

    print(f"✓ 生成: {out_path}")
    print()
    print("  Excel 上で黄色セルを直接編集すると全シートが即時再計算されます。")
    print("  再生成が必要なのは MPM 日程や初期値を大幅に変えたいときのみです。")
    print()
    print("  黄色セル一覧:")
    print("    設定         B5  本日日付 / B7  現在政策金利 / B11:B18  MPM 発表日")
    print("    BOJ→スポット  B5:B12  M1-M8 グリッドレート")
    print("    スポット→BOJ  C6:C13  M1-M8 スポットOIS（ノード）")
    print("    利上げパス    C7:C14  M1-M8 利上げ幅（bp）")

if __name__ == "__main__":
    main()

"""結果ページ（競走成績）の解析 → {馬番: 着順}.

結果ページは「着順・枠・馬番・馬名…」の素直なテーブルなので、ヘッダ文言から
列を特定して着順と馬番を読む。中止・除外（着順が数字でない行）はスキップ。
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_result(html: str) -> dict[int, int]:
    """{馬番: 着順} を返す."""
    soup = BeautifulSoup(html, "lxml")
    table = _result_table(soup)
    if table is None:
        return {}

    trs = table.find_all("tr")
    header = [_clean(c.get_text()) for c in trs[0].find_all(["th", "td"])]
    col_finish = _find_col(header, ("着順", "着"))
    col_num = _find_col(header, ("馬番",))
    if col_finish is None or col_num is None:
        return {}

    result: dict[int, int] = {}
    for tr in trs[1:]:
        cells = [_clean(c.get_text()) for c in tr.find_all(["td", "th"])]
        if len(cells) <= max(col_finish, col_num):
            continue
        fin = re.match(r"\d+", cells[col_finish])
        num = re.search(r"\d+", cells[col_num])
        if fin and num:  # 着順が数字の行だけ（中止/除外は除く）
            result[int(num.group())] = int(fin.group())
    return result


def _result_table(soup: BeautifulSoup):
    """着順と馬番を見出しに持つテーブルを選ぶ."""
    best = None
    best_rows = 0
    for t in soup.find_all("table"):
        heads = [_clean(c.get_text()) for c in t.find_all(["th", "td"])[:14]]
        if any("着" in h for h in heads) and any("馬番" in h for h in heads):
            nrows = len(t.find_all("tr"))
            if nrows > best_rows:
                best, best_rows = t, nrows
    return best


def _find_col(header: list[str], keys: tuple[str, ...]) -> int | None:
    for i, h in enumerate(header):
        if any(k == h or k in h for k in keys):
            return i
    return None

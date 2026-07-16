"""オッズページのデータ構造とパーサ（実HTMLで較正済み）.

楽天競馬のオッズページ構造（2026-06 検証）:
- tanfuku: 各馬行に td.number(馬番) / td.oddsWin(単勝) / td.oddsPlace(複勝 範囲)。
- umafuku/wide/sanrenfuku: 「順位 | 組番 | オッズ」のリスト表が存在し、
  組番は "4-7" / "4-7-10" 形式、オッズは単値（馬連・三連複）か範囲（ワイド）。
  このリスト表（人気順）を解析対象にする（三角マトリクスより堅牢）。

組み合わせは順不同を frozenset、馬番のキーで保持する。
範囲オッズ（複勝・ワイド）は (下限, 上限) のタプル。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

_FLOAT = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class OddsBook:
    win: dict[int, float] = field(default_factory=dict)                  # 単勝 {馬番: オッズ}
    place: dict[int, tuple[float, float]] = field(default_factory=dict)  # 複勝 {馬番: (下,上)}
    quinella: dict[frozenset, float] = field(default_factory=dict)       # 馬連 {{a,b}: オッズ}
    wide: dict[frozenset, tuple[float, float]] = field(default_factory=dict)  # ワイド {{a,b}: (下,上)}
    trio: dict[frozenset, float] = field(default_factory=dict)           # 三連複 {{a,b,c}: オッズ}
    exacta: dict[tuple, float] = field(default_factory=dict)             # 馬単 {(a,b): オッズ}（順序あり）
    trifecta: dict[tuple, float] = field(default_factory=dict)           # 三連単 {(a,b,c): オッズ}（順序あり）

    def has_combos(self) -> bool:
        return bool(self.quinella or self.wide or self.trio or self.exacta or self.trifecta)


# --- パーサ（実HTMLで較正） ------------------------------------------------
# 各 parse_* は対応するオッズページHTMLを受け取り OddsBook の該当辞書を埋める。
# fetch-odds で保存したHTMLを見て、CSSセレクタ/セル順を確定させる。

def parse_into(book: OddsBook, kind: str, html: str) -> None:
    """券種に応じて book を更新する（未較正の券種は何もしない）."""
    fn = {
        "tanfuku": parse_tanfuku,
        "umafuku": parse_umafuku,
        "wide": parse_wide,
        "sanrenfuku": parse_sanrenfuku,
        "umatan": parse_umatan,
        "sanrentan": parse_sanrentan,
    }.get(kind)
    if fn:
        fn(book, html)


def _floats(text: str) -> list[float]:
    return [float(x) for x in _FLOAT.findall(text or "")]


def _range(text: str) -> tuple[float, float] | None:
    vals = _floats(text)
    if not vals:
        return None
    return (vals[0], vals[-1])


def _combo_list_table(soup: BeautifulSoup):
    """「組番」「オッズ」を見出しに持つリスト表を返す（人気順の一覧）."""
    for t in soup.find_all("table"):
        heads = [th.get_text(strip=True) for th in t.find_all("th")[:4]]
        if any("組" in h for h in heads) and any("オッズ" in h for h in heads):
            return t
    return None


def _iter_combo_rows(html: str):
    """(馬番の順序ありリスト, オッズセル文字列) を列挙する.

    組番は "4-7"（順不同券種）/ "4→7"（順序券種）どちらも数字を出現順に取る。
    順不同の券種側で frozenset 化する。
    """
    soup = BeautifulSoup(html, "lxml")
    table = _combo_list_table(soup)
    if table is None:
        return
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        combo_txt = tds[1].get_text(strip=True)          # 例 "4-7" / "4→7→10"
        nums = [int(x) for x in re.findall(r"\d+", combo_txt)]
        if len(nums) < 2:
            continue
        odds_txt = tds[-1].get_text(strip=True)
        yield nums, odds_txt


def parse_tanfuku(book: OddsBook, html: str) -> None:
    """単勝/複勝。各馬行から td.number / td.oddsWin / td.oddsPlace を取る."""
    soup = BeautifulSoup(html, "lxml")
    for win_td in soup.select("td.oddsWin"):
        tr = win_td.find_parent("tr")
        num_td = tr.select_one("td.number") if tr else None
        if not num_td:
            continue
        num_m = re.search(r"\d+", num_td.get_text())
        if not num_m:  # 馬番が数字でない（取消・除外・見出し行など）はスキップ
            continue
        num = int(num_m.group())
        wv = _floats(win_td.get_text())
        if wv:
            book.win[num] = wv[0]
        place_td = tr.select_one("td.oddsPlace")
        if place_td and (rng := _range(place_td.get_text())):
            book.place[num] = rng


def parse_umafuku(book: OddsBook, html: str) -> None:
    """馬連（順不同・単値）."""
    for nums, odds_txt in _iter_combo_rows(html):
        if len(nums) == 2 and (v := _floats(odds_txt)):
            book.quinella[frozenset(nums)] = v[0]


def parse_wide(book: OddsBook, html: str) -> None:
    """ワイド（順不同・範囲）."""
    for nums, odds_txt in _iter_combo_rows(html):
        if len(nums) == 2 and (rng := _range(odds_txt)):
            book.wide[frozenset(nums)] = rng


def parse_sanrenfuku(book: OddsBook, html: str) -> None:
    """三連複（順不同・単値）."""
    for nums, odds_txt in _iter_combo_rows(html):
        if len(nums) == 3 and (v := _floats(odds_txt)):
            book.trio[frozenset(nums)] = v[0]


def parse_umatan(book: OddsBook, html: str) -> None:
    """馬単（順序あり・単値）."""
    for nums, odds_txt in _iter_combo_rows(html):
        if len(nums) == 2 and (v := _floats(odds_txt)):
            book.exacta[tuple(nums)] = v[0]


def parse_sanrentan(book: OddsBook, html: str) -> None:
    """三連単（順序あり・単値）."""
    for nums, odds_txt in _iter_combo_rows(html):
        if len(nums) == 3 and (v := _floats(odds_txt)):
            book.trifecta[tuple(nums)] = v[0]

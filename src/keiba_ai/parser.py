"""HTML → 構造化データ への変換（楽天競馬 出馬表 実構造に較正済み）.

楽天競馬の出馬表は「1頭=rowspan3の馬柱テーブル」。1頭分の先頭行に
以下のセルが並ぶ（2026-06 時点の実HTMLで確認）::

    td.number          馬番
    td.myForecast      自分の印（無視）
    td.name            父馬 / 馬名(span.mainHorse a) / 母馬 / 単勝オッズ(末尾)
    td.profile         性齢 毛色 斤量 騎手 （所属） 【勝率】【連対率】 調教師
    td.weight          馬体重(前走|? )
    td.weightDistance  馬体重 増減
    td.race.placeNN×5  近走5走（先頭の数字が着順）
    td.orderCourse / td.orderDistance  コース別/距離別成績

構造が変わった場合はこのファイルだけ直す。
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from .models import Horse, Race

_FLOAT = re.compile(r"\d+\.\d+")
_INT = re.compile(r"\d+")
# 性齢 毛色 斤量 騎手 （所属）
_PROFILE = re.compile(
    r"^(?P<sexage>\S+)\s+\S+\s+(?P<impost>[\d.]+)\s+(?P<jockey>\S+)\s*（(?P<belong>[^）]*)）"
)
_TRAINER = re.compile(r"】\s*(\S+)\s*$")
_DISTANCE = re.compile(r"(芝|ダ(?:ート)?)[\s]*([\d,]{3,5})\s*m")
_GOING = re.compile(r"(良|稍重|稍|重|不良)")


def parse_race_card(html: str, race_id: str) -> Race:
    soup = BeautifulSoup(html, "lxml")
    race = Race(race_id=race_id)

    _parse_meta(soup, race)

    for name_td in soup.select("td.name"):
        tr = name_td.find_parent("tr")
        if tr is None:
            continue
        horse = _parse_horse(tr, name_td)
        if horse and horse.name:
            race.horses.append(horse)

    race.horses.sort(key=lambda h: h.num)
    return race


def _parse_meta(soup: BeautifulSoup, race: Race) -> None:
    h1 = soup.select_one("h1.unique") or soup.find(["h1", "h2"])
    if h1:
        race.title = re.sub(r"\s+", " ", h1.get_text()).strip()

    # レース条件（距離・馬場）はヘッダ付近のテキストから拾う。
    # 馬柱の「1500左ダ」に誤反応しないよう "○○m" 形式に限定する。
    head_text = soup.get_text(" ")[:4000]
    if m := _DISTANCE.search(head_text):
        race.surface = "ダート" if m.group(1).startswith("ダ") else "芝"
        race.distance_m = int(m.group(2).replace(",", ""))
    # 馬場状態は "ダ：不良" / "芝：稍重" の形で出る（天候表記とは別）
    if m := re.search(r"(?:芝|ダ(?:ート)?)\s*[：:]\s*(不良|稍重|良|重)", head_text):
        race.going = m.group(1)


def _parse_horse(tr, name_td) -> Horse | None:
    num = _first_int(_cell_text(tr, "td.number"))

    # 馬名
    a = name_td.select_one("span.mainHorse a")
    name = a.get_text(strip=True) if a else ""

    # 単勝オッズ = 馬名セル末尾の小数
    floats = _FLOAT.findall(name_td.get_text(" ", strip=True))
    odds = float(floats[-1]) if floats else None

    # 性齢・斤量・騎手・調教師
    sex_age = weight = jockey = trainer = None
    prof = _cell_text(tr, "td.profile")
    if prof:
        if m := _PROFILE.search(prof):
            sex_age = m.group("sexage")
            weight = m.group("impost")
            jockey = m.group("jockey")
        if m := _TRAINER.search(prof):
            trainer = m.group(1)

    # 馬体重・増減、勝率（profile内の【x%】【y%】）
    extra: dict[str, str] = {}
    if wd := _cell_text(tr, "td.weightDistance"):
        extra["馬体重"] = wd.strip()
    rates = re.findall(r"【\s*([\d.]+)\s*%】", prof or "")
    if rates:
        extra["勝率1"] = rates[0]
        if len(rates) > 1:
            extra["勝率2"] = rates[1]

    # 近走着順（馬柱 先頭の数字）
    recent: list[str] = []
    for race_cell in tr.select("td.race"):
        if m := _INT.search(race_cell.get_text(" ", strip=True)):
            recent.append(m.group())

    return Horse(
        num=num or 0,
        name=name,
        sex_age=sex_age,
        weight=weight,
        jockey=jockey,
        trainer=trainer,
        odds_win=odds,
        recent_form=recent,
        extra=extra,
    )


# --- helpers --------------------------------------------------------------
def _cell_text(tr, selector: str) -> str:
    el = tr.select_one(selector)
    return el.get_text(" ", strip=True) if el else ""


def _first_int(text: str) -> int | None:
    m = _INT.search(text or "")
    return int(m.group()) if m else None

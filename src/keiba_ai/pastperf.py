"""出馬表から「オッズにない情報」を詳しく読む（近走5走の中身・血統・条件別成績）.

parser.py は近走の着順の数字しか読まない。こちらは各近走の
日付・場・距離・馬場・頭数・人気・騎手・斤量・タイム・着差・上がり3F・馬体重・馬番・通過順
まで読み、血統（父・母・母父）と馬ID・生年月日、コース別/距離別成績も返す。

1つの近走セル（td.race）の行構成（2026-11 の実HTMLで確認）::

    7 / 良 / 9頭 / 過去映像 / 高知 26.10.17 / Ｃ１－４ Ｃ / Ｃ１ / 1600右ダ / 7人 /
    新庄海 / 57.0 / 1:50.8 (1.7) / 42.6 528k 5番 / 4-4-7-7 / ダノンマカル

欠けている項目（ばんえい・取消など）は None。構造が変わったらこのファイルだけ直す。
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

_DATE = re.compile(r"^(\S+?)\s*(\d{2})\.(\d{2})\.(\d{2})$")
_DIST = re.compile(r"^(内|外)?(\d{3,4})(右|左|直)?(ダ|芝)$")  # 例: 1600右ダ / 外1700右ダ / 200ダ(ばんえい)
_POP = re.compile(r"^(\d+|-)人$")
_TIME = re.compile(r"^(?:(\d+):)?(\d+\.\d)\s*\((-?\d+\.\d)\)$")
_LAST = re.compile(r"^(?:(\d+\.\d)\s+)?(?:(\d+)k\s*)?(?:(\d+)番)?$")
_PASS = re.compile(r"^\d+(?:-\d+)+$")
_IMPOST = re.compile(r"^\d+(?:\.\d)?$")  # 斤量（ばんえいは 650 のような整数kg）
_HORSEID = re.compile(r"HORSEID/(\d+)")
_BIRTH = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})生")


def _lines(el) -> list[str]:
    return [t.strip() for t in el.get_text("\n").split("\n") if t.strip()]


def _int(s: str | None) -> int | None:
    return int(s) if s and s.isdigit() else None


def parse_past_cell(td) -> dict | None:
    """近走セル1つ → dict。空セル（出走歴なし）は None."""
    ls = _lines(td)
    if not ls:
        return None
    d: dict = {
        "finish": _int(ls[0]), "going": None, "n_runners": None, "track": None, "date": None,
        "class": None, "distance": None, "surface": None, "popularity": None, "jockey": None,
        "impost": None, "time": None, "margin": None, "last3f": None, "body_weight": None,
        "gate": None, "passing": None,
    }
    if (st := td.select_one(".stateInfo")):
        d["going"] = st.get_text(strip=True) or None
    if (nm := td.select_one(".numberInfo")):
        d["n_runners"] = _int(nm.get_text(strip=True).replace("頭", ""))
    if (rn := td.select_one(".raceName")):
        d["race_name"] = rn.get_text(" ", strip=True)
    for i, s in enumerate(ls):
        if m := _DATE.match(s):
            d["track"] = m.group(1)
            d["date"] = f"20{m.group(2)}{m.group(3)}{m.group(4)}"
            # 日付の2行後がクラス（1行後はレース名）
            if i + 2 < len(ls) and not _DIST.match(ls[i + 2]):
                d["class"] = ls[i + 2]
        elif m := _DIST.match(s):
            d["distance"] = int(m.group(2))
            d["surface"] = m.group(4)
            d["course_side"] = m.group(1)
        elif m := _POP.match(s):
            d["popularity"] = _int(m.group(1)) or 0  # 取消などの「-人」は0（以降の項目を読む目印）
        elif m := _TIME.match(s):
            d["time"] = round(int(m.group(1) or 0) * 60 + float(m.group(2)), 1)
            d["margin"] = float(m.group(3))
        elif _IMPOST.match(s) and d["impost"] is None and d["popularity"] is not None and d["time"] is None:
            d["impost"] = float(s)
            d["jockey"] = ls[i - 1] if i > 0 else None
        elif _PASS.match(s):
            d["passing"] = [int(x) for x in s.split("-")]
        elif (m := _LAST.match(s)) and any(m.groups()) and d["time"] is not None:
            d["last3f"] = float(m.group(1)) if m.group(1) else None
            d["body_weight"] = _int(m.group(2))
            d["gate"] = _int(m.group(3))
    return d


def _records(td) -> list[list[int]]:
    """コース別/距離別成績セル → [[1着,2着,3着,着外], ...]."""
    if td is None:
        return []
    out = []
    for box in td.select("div.textSet"):
        nums = [_int(x.get_text(strip=True)) for x in box.find_all("span")]
        if len(nums) == 4 and all(n is not None for n in nums):
            out.append(nums)
    return out


def parse_card_details(html: str) -> dict[int, dict]:
    """出馬表 → {馬番: {horse_id, sire, dam, damsire, birth, past:[...5走], course_rec, dist_rec}}."""
    soup = BeautifulSoup(html, "lxml")
    out: dict[int, dict] = {}
    for name_td in soup.select("td.name"):
        tr = name_td.find_parent("tr")
        num_td = tr.select_one("td.number") if tr else None
        num = _int(num_td.get_text(strip=True)) if num_td else None
        if num is None:
            continue
        a = name_td.select_one("span.mainHorse a")
        hid = _HORSEID.search(a["href"]).group(1) if a and _HORSEID.search(a.get("href", "")) else None
        # 馬名セル: 父 / 馬名 / 母 / (母父) …
        ls = _lines(name_td)
        name = a.get_text(strip=True) if a else None
        sire = ls[0] if ls and ls[0] != name else None
        dam = None
        if name in ls:
            j = ls.index(name)
            dam = ls[j + 1] if j + 1 < len(ls) else None
        damsire = None
        if (ap := name_td.select_one("div.append")):
            if m := re.search(r"[(（]\s*([^)）]+?)\s*[)）]", ap.get_text(" ", strip=True)):
                damsire = re.sub(r"\s+", " ", m.group(1))
        birth = None
        if m := _BIRTH.search(name_td.get_text(" ", strip=True)):
            birth = f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}"
        past = [p for td in tr.select("td.race") if (p := parse_past_cell(td))]
        out[num] = {
            "horse_id": hid, "sire": sire, "dam": dam, "damsire": damsire, "birth": birth,
            "past": past,
            "course_rec": _records(tr.select_one("td.orderCourse")),
            "dist_rec": _records(tr.select_one("td.orderDistance")),
        }
    return out

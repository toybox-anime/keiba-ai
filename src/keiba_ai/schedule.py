"""本日開催の「競馬場名 → 開催ID(RACEID)」を取得する.

楽天競馬の出馬表ランディング（`/race_card/list/`）には
「本日の発売情報」テーブルがあり、各競馬場のレース一覧へのリンク
（`/race_card/list/RACEID/<18桁>`）が並ぶ。ここから開催IDを引く。

中間8桁のエンコードが非自明なため、競馬場の指定はこの一覧経由で解決する
（手書きの場コードに依存しない）。
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

_RACEID_RE = re.compile(r"/race_card/list/RACEID/(\d{18})")


def parse_meetings(html: str) -> dict[str, str]:
    """「本日の発売情報」から {競馬場名: 開催ID} を返す."""
    soup = BeautifulSoup(html, "lxml")
    meetings: dict[str, str] = {}

    table = soup.find("table", class_="contentsTable")
    rows = table.find_all("tr") if table else soup.find_all("tr")
    for tr in rows:
        th = tr.find("th")
        if not th:
            continue
        # 「<span>浦和</span>競馬場」→ 競馬場名を取り出す
        span = th.find("span")
        track = (span.get_text(strip=True) if span else th.get_text(strip=True)).replace("競馬場", "")
        if not track:
            continue
        link = tr.find("a", href=_RACEID_RE)
        if not link:
            continue
        m = _RACEID_RE.search(link["href"])
        if m:
            # レース一覧リンク（末尾00基準）に正規化
            meetings[track] = m.group(1)[:-2] + "00"
    return meetings


def resolve_meeting_id(meetings: dict[str, str], track: str) -> str:
    """競馬場名（部分一致可）から開催IDを引く."""
    if track in meetings:
        return meetings[track]
    for name, mid in meetings.items():
        if track in name or name in track:
            return mid
    raise KeyError(
        f"本日の開催に『{track}』が見つかりません。開催中の競馬場: {list(meetings) or '（なし）'}"
    )

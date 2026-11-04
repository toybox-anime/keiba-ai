"""出馬表の詳しい情報（近走の中身・血統など）の保存と読み込み.

1レース=1行の JSONL を、レース日の年月ごとのファイル data/details/YYYYMM.jsonl に追記する
（1ファイルに全部入れると数十MBになり、毎日の追記でgit履歴も重くなるため月で分ける）。
近走は容量を抑えるため配列で持つ（並びは PAST_FIELDS）。
"""

from __future__ import annotations

import json
from pathlib import Path

DETAILS_DIR = Path("data/details")

# 近走1走の配列の並び（pastperf.parse_past_cell のキー）
PAST_FIELDS = (
    "finish", "going", "n_runners", "track", "date", "class", "distance", "surface", "course_side",
    "popularity", "jockey", "impost", "time", "margin", "last3f", "body_weight", "gate", "passing",
)
HORSE_FIELDS = ("horse_id", "sire", "dam", "damsire", "birth", "course_rec", "dist_rec")


def _pack(card: dict[int, dict]) -> dict:
    out = {}
    for num, h in card.items():
        out[str(num)] = {
            **{k: h.get(k) for k in HORSE_FIELDS},
            "past": [[p.get(k) for k in PAST_FIELDS] for p in h.get("past", [])],
        }
    return out


def _unpack(horses: dict) -> dict[int, dict]:
    out = {}
    for num, h in horses.items():
        out[int(num)] = {
            **{k: h.get(k) for k in HORSE_FIELDS},
            "past": [dict(zip(PAST_FIELDS, p)) for p in h.get("past", [])],
        }
    return out


def known_ids(root: Path) -> set[str]:
    """保存済みの race_id."""
    ids = set()
    for f in sorted((root / DETAILS_DIR).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith('{"race_id": "'):
                ids.add(line[13:31])
    return ids


def append(race_id: str, card: dict[int, dict], root: Path) -> bool:
    """1レース分を追記する。中身が空なら書かない."""
    if not card:
        return False
    path = root / DETAILS_DIR / f"{race_id[:6]}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"race_id": race_id, "horses": _pack(card)}, ensure_ascii=False,
                           separators=(",", ":")).replace('{"race_id":"', '{"race_id": "', 1) + "\n")
    return True


def load_all(root: Path) -> dict[str, dict[int, dict]]:
    """{race_id: {馬番: 詳細}}。壊れた行・重複は読み飛ばす（後勝ち）."""
    out: dict[str, dict[int, dict]] = {}
    for f in sorted((root / DETAILS_DIR).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            out[r["race_id"]] = _unpack(r["horses"])
    return out

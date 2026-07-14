"""予想台帳（ledger）と答え合わせ（採点）.

朝に Gemini の予想（◎○▲の馬番）を台帳へ記録し、夜に結果と突き合わせて
的中率を採点する。集計を翌朝のプロンプトに添えることで自己補正を促す。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

LEDGER = "predictions/ledger.jsonl"

_MARKS = {"honmei": "◎", "taikou": "○", "tanana": "▲"}


def parse_picks(text: str) -> dict[str, int | None]:
    """Geminiの予想文から ◎○▲ の馬番を取り出す（【PICKS】行を優先）."""
    target = text
    for line in text.splitlines():
        if "PICKS" in line or ("◎" in line and "○" in line):
            target = line
            break

    def grab(mark: str) -> int | None:
        # 印の直後〜数文字以内の最初の数字を拾う（◎6 / ◎ 6番 / ◎(6) 等に対応。他印は跨がない）
        m = re.search(mark + r"[^\d◎○▲△【】\n]{0,10}(\d{1,2})", target)
        return int(m.group(1)) if m else None

    return {k: grab(v) for k, v in _MARKS.items()}


def load_ledger(path: str | Path = LEDGER) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_ledger(rows: list[dict], path: str | Path = LEDGER) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def append_prediction(record: dict, path: str | Path = LEDGER) -> bool:
    """予想を台帳に追記（同一race_idは重複追記しない）。追記したらTrue."""
    rows = load_ledger(path)
    if any(r["race_id"] == record["race_id"] for r in rows):
        return False
    rows.append(record)
    save_ledger(rows, path)
    return True


def grade_record(rec: dict, result: dict[int, int]) -> dict:
    """1件の予想を結果（{馬番:着順}）で採点し、rec を更新して返す."""
    honmei = rec.get("honmei")
    fin = result.get(honmei) if honmei else None
    rec["honmei_finish"] = fin
    rec["hit_win"] = fin == 1
    rec["hit_place"] = fin is not None and fin <= 3  # 複勝圏
    # ○▲が3着内に来たか（連系のかすり具合）
    rec["others_place"] = sum(
        1 for k in ("taikou", "tanana")
        if rec.get(k) and result.get(rec[k]) is not None and result[rec[k]] <= 3
    )
    rec["graded"] = True
    return rec


def summarize(rows: list[dict], n: int = 30) -> dict:
    """採点済みの直近n件から的中率を集計."""
    graded = [r for r in rows if r.get("graded")]
    recent = graded[-n:]
    if not recent:
        return {"件数": 0}
    win = sum(1 for r in recent if r.get("hit_win"))
    place = sum(1 for r in recent if r.get("hit_place"))
    return {
        "件数": len(recent),
        "◎勝率%": round(win / len(recent) * 100, 1),
        "◎複勝率%": round(place / len(recent) * 100, 1),
    }


def recent_feedback_line(path: str | Path = LEDGER, n: int = 30) -> str:
    """翌朝プロンプトに添える成績サマリ文（自己補正用）."""
    s = summarize(load_ledger(path), n)
    if not s.get("件数"):
        return ""
    return (
        f"【あなたの直近成績 {s['件数']}レース】◎勝率{s['◎勝率%']}% / ◎複勝率{s['◎複勝率%']}%。"
        "これを踏まえ、自分の傾向（人気の過大評価など）を補正して予想すること。"
    )

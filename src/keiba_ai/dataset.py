"""学習データセットの構築・蓄積.

各レースについて「出馬表の特徴量 + 結果(着順)」を1行=1頭で JSONL に追記する。
60秒間隔の制約上データ収集は遅いので、追記式で少しずつ貯める方式にする。
"""

from __future__ import annotations

import json
from pathlib import Path

from .ml import FEATURE_NAMES, horse_features
from .models import Race

DEFAULT_PATH = "data/dataset.jsonl"


def race_result_to_rows(race: Race, result: dict[int, int]) -> list[dict]:
    """Race と結果{馬番:着順} から学習行を作る（着がある馬のみ）."""
    rows = []
    for h in race.horses:
        finish = result.get(h.num)
        if finish is None:
            continue
        feat = horse_features(race, h)
        rows.append(
            {
                "race_id": race.race_id,
                "num": h.num,
                "finish": finish,
                "is_win": 1 if finish == 1 else 0,
                "is_top3": 1 if finish <= 3 else 0,
                "features": {k: feat[k] for k in FEATURE_NAMES},
            }
        )
    return rows


def append_rows(rows: list[dict], path: str | Path = DEFAULT_PATH) -> int:
    """行を JSONL に追記し、追記件数を返す（race_id重複はスキップ）."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {r["race_id"] for r in load_dataset(path)}
    new = [r for r in rows if r["race_id"] not in existing]
    with path.open("a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new)


def load_dataset(path: str | Path = DEFAULT_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dataset_stats(path: str | Path = DEFAULT_PATH) -> dict:
    rows = load_dataset(path)
    races = {r["race_id"] for r in rows}
    return {"races": len(races), "horses": len(rows), "wins": sum(r["is_win"] for r in rows)}

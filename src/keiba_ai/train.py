"""LightGBM 勝率モデルの学習.

レース内で「勝ち馬を上位に並べる」ことを学ぶため、LambdaRank（ランキング学習）
を使う。グループ＝レース。出力スコアはレース内で正規化して勝率確率に変換する
（推論は model.py）。データが少ない間は二値分類にフォールバックする。
"""

from __future__ import annotations

import json
from pathlib import Path

from .dataset import load_dataset
from .ml import FEATURE_NAMES
from .model import META_PATH, MODEL_PATH  # 定数のみ（numpy/lightgbmを読み込まない）


def _to_arrays(rows: list[dict]):
    import numpy as np

    X = np.array([[r["features"][k] for k in FEATURE_NAMES] for r in rows], dtype=float)
    y_win = np.array([r["is_win"] for r in rows], dtype=int)
    race_ids = [r["race_id"] for r in rows]
    return X, y_win, race_ids


def _group_sizes(race_ids: list[str]) -> list[int]:
    """連続する同一race_idの件数（LightGBMのgroup用）。並び順を保持する前提."""
    sizes, cur, n = [], None, 0
    for rid in race_ids:
        if rid != cur and cur is not None:
            sizes.append(n)
            n = 0
        cur, n = rid, n + 1
    if n:
        sizes.append(n)
    return sizes


def train(
    dataset_path: str = "data/dataset.jsonl",
    model_out: str = MODEL_PATH,
    *,
    min_races: int = 30,
) -> dict:
    """データセットから LightGBM を学習し、モデルを保存する."""
    import lightgbm as lgb
    import numpy as np

    rows = load_dataset(dataset_path)
    # レース単位でまとめる（group化のため race_id でソート）
    rows.sort(key=lambda r: r["race_id"])
    races = {r["race_id"] for r in rows}
    if len(races) < min_races:
        return {
            "trained": False,
            "reason": f"レース数が不足: {len(races)} < {min_races}。データを増やしてください。",
            "races": len(races),
        }

    X, y_win, race_ids = _to_arrays(rows)
    groups = _group_sizes(race_ids)

    # ランキング学習（relevance: 1着=2, 3着内=1, それ以外=0）
    relevance = np.array(
        [2 if r["is_win"] else (1 if r.get("is_top3") else 0) for r in rows], dtype=int
    )
    dtrain = lgb.Dataset(X, label=relevance, group=groups, feature_name=FEATURE_NAMES)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbose": -1,
    }
    booster = lgb.train(params, dtrain, num_boost_round=200)

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_out)
    Path(META_PATH).write_text(
        json.dumps(
            {"features": FEATURE_NAMES, "objective": "lambdarank", "races": len(races)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "trained": True,
        "races": len(races),
        "horses": len(rows),
        "wins": int(y_win.sum()),
        "model": model_out,
    }

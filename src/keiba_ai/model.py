"""学習済み勝率モデルの推論.

LambdaRank のスコアはレース内の相対順位を表すので、softmax でレース内の
勝率確率（合計1）に変換する。温度 T で確信度を調整できる。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .ml import FEATURE_NAMES, race_to_matrix
from .models import Race

MODEL_PATH = "models/win_model.txt"
META_PATH = "models/win_model.meta.json"


def _softmax(scores: list[float], temp: float) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp((s - m) / temp) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


class WinModel:
    def __init__(self, booster, features: list[str], temp: float = 1.0) -> None:
        self.booster = booster
        self.features = features
        self.temp = temp

    @classmethod
    def load(cls, model_path: str = MODEL_PATH, meta_path: str = META_PATH, temp: float = 1.0):
        if not Path(model_path).exists():
            return None
        try:
            import lightgbm as lgb
        except ImportError:
            return None  # ML未インストールなら静かにモデル無しで動作
        booster = lgb.Booster(model_file=model_path)
        features = FEATURE_NAMES
        if Path(meta_path).exists():
            features = json.loads(Path(meta_path).read_text(encoding="utf-8")).get("features", FEATURE_NAMES)
        return cls(booster, features, temp)

    def predict_race(self, race: Race) -> dict[int, float]:
        """{馬番: 勝率} を返す（レース内で合計1に正規化）."""
        X, nums = race_to_matrix(race)
        if not X:
            return {}
        scores = list(self.booster.predict(X))
        probs = _softmax(scores, self.temp)
        return dict(zip(nums, probs))

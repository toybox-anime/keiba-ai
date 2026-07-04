"""機械学習用の特徴量エンジニアリング.

Race/Horse を数値ベクトルに変換する。学習・推論で同じ関数を使うことで
特徴量の不一致を防ぐ。欠損は NaN（LightGBM がそのまま扱える）。
"""

from __future__ import annotations

import math
import re

from .models import Horse, Race

# 学習・推論で共通の特徴量順（モデル保存時にも記録する）
FEATURE_NAMES: list[str] = [
    "num", "age", "sex_male", "sex_female", "sex_gelding",
    "impost", "odds_win", "implied_prob", "popularity",
    "form_avg", "form_best", "form_last", "form_win_rate", "form_count",
    "body_weight", "body_change", "rate1", "rate2",
    "distance_m", "surface_dirt", "going_code", "n_runners",
]

_GOING_CODE = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
_NAN = float("nan")


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return _NAN


def _age(sex_age: str | None) -> float:
    if not sex_age:
        return _NAN
    m = re.search(r"\d+", sex_age)
    return float(m.group()) if m else _NAN


def _sex_flags(sex_age: str | None) -> tuple[float, float, float]:
    s = sex_age or ""
    return (
        1.0 if s.startswith("牡") else 0.0,
        1.0 if s.startswith("牝") else 0.0,
        1.0 if s.startswith("セ") else 0.0,
    )


def _form_stats(recent: list[str]) -> tuple[float, float, float, float, float]:
    pos = []
    for r in recent:
        try:
            p = int(r)
        except (TypeError, ValueError):
            continue
        if p > 0:
            pos.append(p)
    if not pos:
        return (_NAN, _NAN, _NAN, _NAN, 0.0)
    avg = sum(pos) / len(pos)
    best = min(pos)
    last = pos[0]
    win_rate = sum(1 for p in pos if p == 1) / len(pos)
    return (avg, float(best), float(last), win_rate, float(len(pos)))


def _body(extra: dict) -> tuple[float, float]:
    txt = extra.get("馬体重", "")
    nums = re.findall(r"[+-]?\d+", txt)
    if not nums:
        return (_NAN, _NAN)
    weight = float(nums[0])
    change = float(nums[1]) if len(nums) > 1 else _NAN
    return (weight, change)


def horse_features(race: Race, horse: Horse) -> dict[str, float]:
    """1頭の特徴量ベクトル（dict）を返す."""
    male, female, gelding = _sex_flags(horse.sex_age)
    favg, fbest, flast, fwr, fcnt = _form_stats(horse.recent_form)
    bw, bc = _body(horse.extra)
    odds = _f(horse.odds_win)
    implied = 1.0 / odds if odds and not math.isnan(odds) and odds > 0 else _NAN
    return {
        "num": _f(horse.num),
        "age": _age(horse.sex_age),
        "sex_male": male, "sex_female": female, "sex_gelding": gelding,
        "impost": _f(horse.weight),
        "odds_win": odds,
        "implied_prob": implied,
        "popularity": _f(horse.popularity),
        "form_avg": favg, "form_best": fbest, "form_last": flast,
        "form_win_rate": fwr, "form_count": fcnt,
        "body_weight": bw, "body_change": bc,
        "rate1": _f(horse.extra.get("勝率1")),
        "rate2": _f(horse.extra.get("勝率2")),
        "distance_m": _f(race.distance_m),
        "surface_dirt": 1.0 if (race.surface or "").startswith("ダ") else 0.0,
        "going_code": float(_GOING_CODE.get(race.going or "", _NAN)),
        "n_runners": float(len(race.horses)),
    }


def race_to_matrix(race: Race) -> tuple[list[list[float]], list[int]]:
    """Race → (特徴量行列, 馬番リスト)。FEATURE_NAMES の順に並べる."""
    X, nums = [], []
    for h in race.horses:
        feat = horse_features(race, h)
        X.append([feat[name] for name in FEATURE_NAMES])
        nums.append(h.num)
    return X, nums

"""ドメインモデル（取得データの構造化表現）."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Horse:
    """出走馬1頭の情報."""

    num: int                      # 馬番
    name: str                     # 馬名
    frame: int | None = None      # 枠番
    sex_age: str | None = None    # 性齢 (例: 牡4)
    weight: str | None = None     # 斤量
    jockey: str | None = None     # 騎手
    trainer: str | None = None    # 調教師
    odds_win: float | None = None        # 単勝オッズ
    popularity: int | None = None        # 人気
    recent_form: list[str] = field(default_factory=list)  # 近走着順 ["1","3","2"]
    extra: dict[str, str] = field(default_factory=dict)   # 未分類の追加情報


@dataclass
class Race:
    """1レースの出馬表."""

    race_id: str
    title: str | None = None       # レース名
    track: str | None = None       # 競馬場
    race_no: int | None = None     # レース番号
    distance_m: int | None = None  # 距離
    surface: str | None = None     # ダート/芝
    going: str | None = None       # 馬場状態
    horses: list[Horse] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "race_id": self.race_id,
            "title": self.title,
            "track": self.track,
            "race_no": self.race_no,
            "distance_m": self.distance_m,
            "surface": self.surface,
            "going": self.going,
            "horses": [h.__dict__ for h in self.horses],
        }

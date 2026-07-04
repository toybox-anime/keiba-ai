"""楽天競馬の RACEID ユーティリティ.

実データ検証で判明した仕様:

- RACEID は 18桁。
- 先頭 8桁 = 開催日 (YYYYMMDD)。
- **末尾 2桁 = レース番号** (`00`=レース一覧/開催, `11`=11R)。
  （例: 浦和の払戻リンクが `...12`=12R、園田が `...07` で「8R以降中止」と一致）
- 中間 8桁 = 開催回・日次・競馬場を表すが、エンコードは非自明で
  **手計算では作れない**。→ サイトの一覧から開催IDを取得して使う方式にする。
  （`schedule.py` が本日開催の「競馬場名 → 開催ID」を取得する）

したがって、レースのRACEIDは「開催ID（末尾00）の下2桁をレース番号に差し替える」
ことで得る。これが唯一信頼できる組み立て方。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

RACEID_LEN = 18


def race_id_for(meeting_id: str, race_no: int) -> str:
    """開催ID（または同一開催の任意のRACEID）とレース番号から、対象レースのRACEIDを作る."""
    if len(meeting_id) != RACEID_LEN or not meeting_id.isdigit():
        raise ValueError(f"不正な開催ID: {meeting_id!r} (18桁数字が必要)")
    if not (1 <= race_no <= 12):
        raise ValueError(f"レース番号は1〜12: {race_no}")
    return meeting_id[:-2] + f"{race_no:02d}"


@dataclass(frozen=True)
class RaceIdInfo:
    race_id: str
    day: date
    race_no: int
    meeting_id: str   # 同一開催の基準ID（末尾00）


def parse_race_id(race_id: str) -> RaceIdInfo:
    """RACEID から、確実に分かる情報（日付・レース番号・開催ID）を取り出す.

    競馬場名は中間8桁のエンコードが非自明なため、ここでは解決しない
    （競馬場名は schedule.py 側の対応表で引く）。
    """
    if len(race_id) != RACEID_LEN or not race_id.isdigit():
        raise ValueError(f"不正な RACEID: {race_id!r} (18桁数字が必要)")
    y, m, d = int(race_id[0:4]), int(race_id[4:6]), int(race_id[6:8])
    race_no = int(race_id[-2:])
    meeting_id = race_id[:-2] + "00"
    return RaceIdInfo(race_id=race_id, day=date(y, m, d), race_no=race_no, meeting_id=meeting_id)

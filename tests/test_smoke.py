"""ネットワーク不要のスモークテスト."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keiba_ai import raceid as rid
from keiba_ai.features import build_feature_table, implied_prob_from_odds
from keiba_ai.parser import parse_race_card
from keiba_ai.report import _offline_report
from keiba_ai.features import summarize_market
from keiba_ai.schedule import parse_meetings, resolve_meeting_id


def test_race_id_for_and_parse():
    # 開催ID（末尾00）+ レース番号 → RACEID
    meeting = "202606261813030500"
    rid_str = rid.race_id_for(meeting, 11)
    assert rid_str == "202606261813030511"
    info = rid.parse_race_id(rid_str)
    assert info.day == date(2026, 6, 26)
    assert info.race_no == 11
    assert info.meeting_id == meeting


_SCHEDULE_HTML = """
<table class="contentsTable">
<tr><th scope="row"><span>浦和</span>競馬場</th>
    <td class="raceState"><a href="https://keiba.rakuten.co.jp/race_card/list/RACEID/202606261813030500">レース一覧</a></td></tr>
<tr><th scope="row"><span>園田</span>競馬場</th>
    <td class="raceState"><a href="https://keiba.rakuten.co.jp/race_card/list/RACEID/202606262726080300">レース一覧</a></td></tr>
</table>
"""


def test_parse_meetings():
    m = parse_meetings(_SCHEDULE_HTML)
    assert m["浦和"] == "202606261813030500"
    assert resolve_meeting_id(m, "園田") == "202606262726080300"


def test_implied_prob():
    assert implied_prob_from_odds(2.0) == 0.5
    assert implied_prob_from_odds(None) is None


# 楽天競馬の実構造（1頭=rowspan3の馬柱）を模した最小HTML
def _horse_row(num, sire, name, dam, odds, profile):
    return f"""
<tr>
  <td rowspan="3" class="number">{num}</td>
  <td rowspan="3" class="myForecast">-</td>
  <td rowspan="3" class="name">{sire}<span class="mainHorse"><a href="/horse_detail/detail/HORSEID/1">{name}</a></span>{dam}<div class="append">(トレーニング) {odds}</div></td>
  <td rowspan="3" class="profile change">{profile}</td>
  <td rowspan="3" class="weightDistance">480 +2</td>
  <td rowspan="3" class="race place09">3 良 10頭 浦和 26.05.29</td>
</tr>"""


SAMPLE_HTML = (
    '<html><body><h1 class="unique">テスト記念 11R 出馬表</h1>'
    "<p>ダ1,500m 天候：雨 ダ：不良</p><table>"
    + _horse_row(1, "父A", "アルファ", "母A", "2.5", "牡4 鹿毛 56.0 山田 （大 井） 【5.0%】 【9.9%】 田中")
    + _horse_row(2, "父B", "ブラボー", "母B", "5.0", "牝5 黒鹿毛 54.0 鈴木 （船 橋） 【1.0%】 【8.0%】 佐藤")
    + _horse_row(3, "父C", "チャーリー", "母C", "12.0", "セ6 栗毛 56.0 高橋 （浦 和） 【2.0%】 【7.0%】 渡辺")
    + "</table></body></html>"
)


def test_parse_and_report():
    race = parse_race_card(SAMPLE_HTML, "202606264400110011")
    assert len(race.horses) == 3
    h1 = race.horses[0]
    assert h1.name == "アルファ"
    assert h1.odds_win == 2.5
    assert h1.jockey == "山田"
    assert h1.sex_age == "牡4"
    assert race.distance_m == 1500
    assert race.going == "不良"
    rows = build_feature_table(race)
    assert rows[0]["馬名"] == "アルファ"  # 最低オッズが先頭
    md = _offline_report(race, rows, summarize_market(rows))
    assert "20歳以上" in md


def test_betting_plan_within_budget():
    from keiba_ai.betting import build_plan

    race = parse_race_card(SAMPLE_HTML, "202606264400110011")
    rows = build_feature_table(race)
    for style in ("conservative", "balanced", "aggressive"):
        plan = build_plan(rows, bankroll=3000, style=style)
        assert plan.total_stake <= 3000, style          # 軍資金を超えない
        assert all(b.stake % 100 == 0 for b in plan.bets)  # 100円単位
        assert all(b.stake >= 100 for b in plan.bets)


def test_ev_engine_math():
    from itertools import combinations
    from keiba_ai import ev

    win = {1: 2.0, 2: 4.0, 3: 6.0, 4: 8.0, 5: 20.0}
    p = ev.fair_win_probs(win)
    assert abs(sum(p.values()) - 1.0) < 1e-6
    assert abs(sum(ev.quinella_prob(p, i, j) for i, j in combinations(p, 2)) - 1.0) < 1e-6
    assert abs(sum(ev.trio_prob(p, i, j, k) for i, j, k in combinations(p, 3)) - 1.0) < 1e-6
    # ケリー: 妙味なし(EV<1)なら0、妙味あり(高オッズ)なら正
    assert ev.kelly_fraction(2.0, p[1]) == 0.0
    assert ev.kelly_fraction(10.0, p[1]) > 0  # フェアより高い払戻なら賭ける


def test_ev_plan_selects_value_and_caps_budget():
    from keiba_ai.betting import build_plan_ev
    from keiba_ai.ev import quinella_prob, fair_win_probs
    from keiba_ai.odds import OddsBook

    rows = [{"馬番": n, "馬名": f"馬{n}", "単勝": o} for n, o in
            [(1, 2.0), (2, 4.0), (3, 6.0), (4, 8.0), (5, 20.0)]]
    book = OddsBook(win={r["馬番"]: r["単勝"] for r in rows})
    # 馬連1-2 を意図的に割高（フェアより高い）に設定 → 妙味として選ばれるはず
    p = fair_win_probs(book.win)
    fair = 1 / quinella_prob(p, 1, 2)
    book.quinella[frozenset((1, 2))] = round(fair * 1.5, 1)   # 割高=買い
    book.quinella[frozenset((1, 3))] = round(1 / quinella_prob(p, 1, 3) * 0.5, 1)  # 割安=見送り

    plan = build_plan_ev(rows, book, bankroll=5000, style="balanced")
    combos = {b.horses for b in plan.bets}
    assert (1, 2) in combos        # 妙味ありは採用
    assert (1, 3) not in combos    # 割安は不採用
    assert plan.total_stake <= 5000


def test_ml_features_and_dataset():
    from keiba_ai.ml import FEATURE_NAMES, horse_features, race_to_matrix
    from keiba_ai.dataset import race_result_to_rows

    race = parse_race_card(SAMPLE_HTML, "202606264400110011")
    h = race.horses[0]
    feat = horse_features(race, h)
    assert set(feat) == set(FEATURE_NAMES)        # 特徴量の網羅
    assert feat["sex_male"] == 1.0 and feat["age"] == 4.0
    assert feat["surface_dirt"] == 1.0 and feat["going_code"] == 3.0  # 不良
    X, nums = race_to_matrix(race)
    assert len(X) == 3 and len(X[0]) == len(FEATURE_NAMES)

    rows = race_result_to_rows(race, {1: 1, 2: 3, 3: 2})  # 馬1が1着
    assert len(rows) == 3
    win = [r for r in rows if r["num"] == 1][0]
    assert win["is_win"] == 1 and win["is_top3"] == 1


def test_recommend_all_bet_types():
    from keiba_ai.betting import recommend_buy_methods
    from keiba_ai.ev import fair_win_probs, quinella_prob, exacta_prob
    from keiba_ai.odds import OddsBook

    rows = [{"馬番": n, "馬名": f"馬{n}", "単勝": o} for n, o in
            [(1, 2.0), (2, 4.0), (3, 6.0), (4, 8.0), (5, 20.0)]]
    book = OddsBook(win={r["馬番"]: r["単勝"] for r in rows})
    p = fair_win_probs(book.win)
    # 馬連1-2 と 馬単1→2 を割高（妙味あり）に設定
    book.quinella[frozenset((1, 2))] = round(1 / quinella_prob(p, 1, 2) * 1.5, 1)
    book.exacta[(1, 2)] = round(1 / exacta_prob(p, 1, 2) * 1.6, 1)   # 順序ありキー
    book.place[1] = (1.1, 1.4)

    rec = recommend_buy_methods(rows, book)
    kinds = {o["券種"] for o in rec["options"]}
    assert "馬単" in kinds and "馬連" in kinds      # 順序あり券種も比較に入る
    assert rec["best"]["EV"] >= 1.0               # 妙味のある買い目を推す
    assert rec["positive_count"] >= 1


if __name__ == "__main__":
    test_race_id_for_and_parse()
    test_implied_prob()
    test_parse_meetings()
    test_parse_and_report()
    test_betting_plan_within_budget()
    test_ev_engine_math()
    test_ev_plan_selects_value_and_caps_budget()
    test_ml_features_and_dataset()
    test_recommend_all_bet_types()
    print("OK: all smoke tests passed")

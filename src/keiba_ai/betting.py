"""軍資金から買い目（券種・組み合わせ）と賭け金配分を生成する.

設計:
- お金の計算は決定論的に行い、100円単位・合計が軍資金を超えないことを保証する。
- 「どの馬を軸/相手にするか」は、市場勝率（単勝オッズ）と近走の調子から
  算出する複合スコアで決める（後でMLスコアに差し替え可能）。
- リスクスタイルで券種配分を変える:
    conservative … 単勝・複勝中心（本命を厚く）
    balanced     … 単複 + 馬連/ワイドのフォーメーション
    aggressive   … 三連複/三連単のフォーメーション + 穴

注意: 馬連・三連複等のオッズは未取得のため、的中時の配当は変動する
（このモジュールは「配分」を決める。期待配当は単勝のみ算出）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations

from . import ev as evmod
from .odds import OddsBook

UNIT = 100  # 最低賭け金単位（円）

# スタイル別のケリー係数（全体に掛ける安全率）と EV しきい値
STYLE_KELLY: dict[str, tuple[float, float]] = {
    # style: (フラクショナルケリー係数, EV下限)
    "conservative": (0.10, 1.20),
    "balanced": (0.25, 1.10),
    "aggressive": (0.50, 1.05),
}

# 券種ごとの軍資金配分（合計1.0）。スタイル別。
STYLE_ALLOC: dict[str, dict[str, float]] = {
    "conservative": {"単勝": 0.35, "複勝": 0.45, "馬連": 0.20},
    "balanced": {"単勝": 0.20, "複勝": 0.20, "馬連": 0.30, "ワイド": 0.30},
    "aggressive": {"単勝": 0.15, "ワイド": 0.15, "三連複": 0.40, "三連単": 0.30},
}


@dataclass
class Bet:
    bet_type: str               # 単勝/複勝/馬連/ワイド/三連複/三連単
    horses: tuple[int, ...]     # 馬番の組（順序あり=三連単）
    stake: int                  # 賭け金（円, 100円単位）
    note: str = ""              # 期待配当など
    horse_names: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class BettingPlan:
    bankroll: int
    style: str
    bets: list[Bet]
    total_stake: int
    axis: dict | None = None        # 本命馬
    partners: list[dict] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "軍資金": self.bankroll,
            "スタイル": self.style,
            "投資合計": self.total_stake,
            "余り": self.bankroll - self.total_stake,
            "本命": self.axis,
            "相手": self.partners,
            "買い目": [
                {
                    "券種": b.bet_type,
                    "組": "-".join(map(str, b.horses)),
                    "馬名": " / ".join(b.horse_names),
                    "金額": b.stake,
                    "メモ": b.note,
                }
                for b in self.bets
            ],
            "注記": self.note,
        }


def rank_horses(rows: list[dict]) -> list[dict]:
    """市場勝率と近走の複合スコアで馬を順位付けして返す（降順）."""
    ranked = []
    for r in rows:
        market = (r.get("市場勝率%") or 0) / 100.0
        form = r.get("_form", 0.0)
        score = round(0.75 * market + 0.25 * form, 4)
        ranked.append({**r, "score": score})
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


def build_plan(
    rows: list[dict],
    bankroll: int,
    style: str = "balanced",
    *,
    max_partners: int = 4,
) -> BettingPlan:
    """軍資金と出走馬指標から買い目プランを生成する."""
    style = style if style in STYLE_ALLOC else "balanced"
    priced = [r for r in rows if r.get("単勝")]
    if len(priced) < 3 or bankroll < UNIT * 3:
        return BettingPlan(bankroll, style, [], 0, note="頭数または軍資金が不足しています。")

    ranked = rank_horses(priced)
    axis = ranked[0]
    partners = ranked[1 : 1 + max_partners]
    alloc = STYLE_ALLOC[style]

    odds_map = {r["馬番"]: r["単勝"] for r in priced}

    bets: list[Bet] = []
    for bet_type, ratio in alloc.items():
        budget = int(bankroll * ratio)
        sels = _selections_for(bet_type, axis, partners)
        bets.extend(_allocate(bet_type, sels, budget))

    # 単勝は的中時払い戻しの目安を注記（オッズ既知）
    for b in bets:
        if b.bet_type == "単勝" and (o := odds_map.get(b.horses[0])):
            b.note = f"的中時 約{int(b.stake * o):,}円"

    total = sum(b.stake for b in bets)
    # 端数調整: 合計が軍資金を超えないことは _allocate で保証済み
    return BettingPlan(
        bankroll=bankroll,
        style=style,
        bets=[b for b in bets if b.stake >= UNIT],
        total_stake=total,
        axis={"馬番": axis["馬番"], "馬名": axis["馬名"], "単勝": axis["単勝"]},
        partners=[{"馬番": p["馬番"], "馬名": p["馬名"], "単勝": p["単勝"]} for p in partners],
        note="馬連以降のオッズは未取得のため的中時配当は変動します。",
    )


def build_plan_ev(
    rows: list[dict],
    book: OddsBook,
    bankroll: int,
    style: str = "balanced",
    *,
    max_bets: int = 12,
    win_probs: dict[int, float] | None = None,
) -> BettingPlan:
    """EV＞しきい値の買い目をケリー基準で配分する（組み合わせオッズが必要）.

    win_probs を渡すと、市場由来のフェア確率の代わりにそれ（=MLモデルの勝率）を
    使う。モデルが市場と乖離するほど単勝・連系にもエッジが生まれる。
    """
    style = style if style in STYLE_KELLY else "balanced"
    kelly_coef, ev_min = STYLE_KELLY[style]
    win_odds = book.win or {r["馬番"]: r["単勝"] for r in rows if r.get("単勝")}
    p = win_probs or evmod.fair_win_probs(win_odds)
    names = {r["馬番"]: r["馬名"] for r in rows}
    if not p or bankroll < UNIT * 2:
        return BettingPlan(bankroll, style, [], 0, note="オッズまたは軍資金が不足しています。")

    # (券種, 馬番組, オッズ, 的中確率) の候補を集める
    cands: list[tuple[str, tuple[int, ...], float, float]] = []
    for n, o in win_odds.items():
        cands.append(("単勝", (n,), o, p.get(n, 0.0)))
    for combo, o in book.quinella.items():
        i, j = sorted(combo)
        cands.append(("馬連", (i, j), o, evmod.quinella_prob(p, i, j)))
    for combo, (lo, _hi) in book.wide.items():
        i, j = sorted(combo)
        cands.append(("ワイド", (i, j), lo, evmod.wide_prob(p, i, j)))  # 下限で保守的に
    for combo, o in book.trio.items():
        i, j, k = sorted(combo)
        cands.append(("三連複", (i, j, k), o, evmod.trio_prob(p, i, j, k)))
    for (i, j), o in book.exacta.items():
        cands.append(("馬単", (i, j), o, evmod.exacta_prob(p, i, j)))
    for (i, j, k), o in book.trifecta.items():
        cands.append(("三連単", (i, j, k), o, evmod.trifecta_prob(p, i, j, k)))

    # EV と ケリー比率を算出し、妙味のある買い目だけ残す
    scored = []
    for bet_type, horses, o, prob in cands:
        e = evmod.ev(o, prob)
        f = evmod.kelly_fraction(o, prob)
        if e >= ev_min and f > 0:
            scored.append((bet_type, horses, o, prob, e, f))
    scored.sort(key=lambda x: x[4], reverse=True)  # EV降順
    scored = scored[:max_bets]
    if not scored:
        return BettingPlan(
            bankroll, style, [], 0,
            note=f"EV≧{ev_min} の妙味ある買い目が見つかりませんでした（{style}）。",
        )

    # 配分: 軍資金の kelly_coef ぶんを「エッジ(ケリー比率)で重み付け」して配る。
    # ただし1点あたりフルケリー（bankroll×f）を上限に安全側へ寄せる。
    fsum = sum(f for *_, f in scored) or 1.0
    target = bankroll * kelly_coef
    bets: list[Bet] = []
    for bet_type, horses, o, prob, e, f in scored:
        amt = min(target * f / fsum, bankroll * f)
        stake = int(amt // UNIT) * UNIT
        if stake < UNIT and amt >= UNIT * 0.5:
            stake = UNIT  # 端数でも妙味があれば最低100円は張る
        if stake < UNIT:
            continue
        bets.append(
            Bet(
                bet_type,
                horses,
                stake,
                note=f"オッズ{o} EV{e} 的中{prob*100:.1f}%",
                horse_names=tuple(names.get(n, "") for n in horses),
            )
        )

    total = sum(b.stake for b in bets)
    axis_num = max(p, key=p.get)
    return BettingPlan(
        bankroll=bankroll,
        style=style,
        bets=bets,
        total_stake=total,
        axis={"馬番": axis_num, "馬名": names.get(axis_num, ""), "単勝": win_odds.get(axis_num)},
        partners=[],
        note=f"EV≧{ev_min}・{int(kelly_coef*100)}%ケリーで配分（組み合わせオッズ使用）。",
    )


def recommend_buy_methods(
    rows: list[dict], book: OddsBook, win_probs: dict[int, float] | None = None,
    bankroll: int | None = None,
) -> dict | None:
    """全券種の中から各々のベスト買い目を出し、総合おすすめと「自信のある買い目」を選ぶ.

    各券種で上位人気5頭の組み合わせのうち最も期待値(EV)が高い1点を代表とする。
    総合おすすめは「EV×√的中率」が最大（妙味と当てやすさのバランス）。
    軍資金が指定されれば、EVプラス（＝自信あり）の券種に自信度で金額配分する。
    """
    priced = [r for r in rows if r.get("単勝")]
    if len(priced) < 3 or not book or not book.has_combos():
        return None
    ranked = rank_horses(priced)
    names = {r["馬番"]: r["馬名"] for r in priced}
    win_odds = {r["馬番"]: r["単勝"] for r in priced}
    p = win_probs or evmod.fair_win_probs(win_odds)
    top = [r["馬番"] for r in ranked[:5]]
    fav = ranked[0]["馬番"]

    def lo(o):
        return o[0] if isinstance(o, tuple) else o

    def opt(kind, combo, odds, prob, hint):
        return {
            "券種": kind, "組": "→".join(map(str, combo)) if kind in ("馬単", "三連単")
            else "-".join(map(str, combo)),
            "馬名": " / ".join(names.get(n, "") for n in combo),
            "オッズ": round(odds, 1), "的中率%": round(prob * 100, 1),
            "EV": evmod.ev(odds, prob), "向き": hint, "_score": evmod.ev(odds, prob) * (prob ** 0.5),
        }

    def best(kind, keys_probs, hint):
        bo = None
        for key, odds, prob in keys_probs:
            if not odds or prob <= 0:
                continue
            o = opt(kind, key, lo(odds), prob, hint)
            if bo is None or o["EV"] > bo["EV"]:
                bo = o
        return bo

    options = []
    # 単勝・複勝（本命）
    options.append(opt("単勝", (fav,), win_odds[fav], p.get(fav, 0), "配当中・本命の素直な勝負"))
    if book.place.get(fav):
        options.append(opt("複勝", (fav,), book.place[fav][0], evmod.place_prob(p, fav), "当てやすい・手堅い"))
    # 2頭系
    pairs = list(combinations(top, 2))
    if book.wide:
        options.append(best("ワイド", [(c, book.wide.get(frozenset(c)), evmod.wide_prob(p, *c)) for c in pairs], "当てやすさと配当のバランス"))
    if book.quinella:
        options.append(best("馬連", [(c, book.quinella.get(frozenset(c)), evmod.quinella_prob(p, *c)) for c in pairs], "中配当・順不同で2頭"))
    if book.exacta:
        oprs = [(c, book.exacta.get(tuple(c)), evmod.exacta_prob(p, *c)) for c in permutations(top, 2)]
        options.append(best("馬単", oprs, "高配当・着順を当てる"))
    # 3頭系
    trios = list(combinations(top, 3))
    if book.trio:
        options.append(best("三連複", [(c, book.trio.get(frozenset(c)), evmod.trio_prob(p, *c)) for c in trios], "高配当・3頭順不同"))
    if book.trifecta:
        tpr = [(c, book.trifecta.get(tuple(c)), evmod.trifecta_prob(p, *c)) for c in permutations(top, 3)]
        options.append(best("三連単", tpr, "最高配当・3頭の着順"))

    options = [o for o in options if o]
    if not options:
        return None
    # プラス期待値(EV≧1.0)を優先し、その中で EV×√的中率（妙味×当てやすさ）が最大を推す。
    positive = [o for o in options if o["EV"] >= 1.0]
    if positive:
        best_overall = max(positive, key=lambda o: o["_score"])
        reason = (
            f"プラス期待値（EV={best_overall['EV']}）の中で、的中率"
            f"（{best_overall['的中率%']}%）とのバランスが最良。"
        )
    else:
        best_overall = max(options, key=lambda o: o["的中率%"])  # 最も手堅い
        reason = "全券種でEV1.0超なし＝理論上は見送り推奨。買うなら最も手堅いこの券種。"
    # 「自信のある買い目」＝EVプラスの券種。軍資金があれば自信度(EV×√的中率)で金額配分。
    confident = sorted(positive, key=lambda o: o["_score"], reverse=True)
    if bankroll and confident:
        weights = [o["_score"] for o in confident]
        for o, s in zip(confident, weighted_split(weights, bankroll)):
            o["stake"] = s

    return {
        "options": sorted(options, key=lambda o: (o["EV"] >= 1.0, o["_score"]), reverse=True),
        "best": best_overall,
        "confident": confident,
        "positive_count": len(positive),
        "reason": reason,
    }


def weighted_split(weights: list[float], bankroll: int) -> list[int]:
    """軍資金を重み（＝自信度）に比例して100円単位で配分する（合計≤軍資金）.

    重みの大きい点ほど多く張る。重みが正の点には最低100円を確保する。
    """
    n = len(weights)
    if n == 0 or bankroll < UNIT:
        return [0] * n
    wsum = sum(w for w in weights if w > 0) or 1.0
    stakes = [int((bankroll * max(w, 0) / wsum) // UNIT) * UNIT for w in weights]
    # 正の重みの点に最低100円
    for i, w in enumerate(weights):
        if w > 0 and stakes[i] < UNIT:
            stakes[i] = UNIT
    # 予算超過なら大きい順に100円ずつ削る
    while sum(stakes) > bankroll:
        j = max(range(n), key=lambda k: stakes[k])
        stakes[j] -= UNIT
    # 余りは重みの大きい順に100円ずつ足す
    order = sorted(range(n), key=lambda k: -weights[k])
    rem, idx = bankroll - sum(stakes), 0
    while rem >= UNIT and order:
        stakes[order[idx % n]] += UNIT
        rem -= UNIT
        idx += 1
    return stakes


def wide_suggestions(
    rows: list[dict], book: OddsBook | None = None, *, n_partners: int = 3, bankroll: int | None = None
) -> dict | None:
    """ワイドの推奨組み合わせ（軸流し・ボックス）を作る.

    オッズ(book)があれば的中確率・EVも添える。買い目配分とは別の「買い方ガイド」。
    """
    priced = [r for r in rows if r.get("単勝")]
    if len(priced) < 3:
        return None
    ranked = rank_horses(priced)
    axis = ranked[0]
    partners = ranked[1 : 1 + n_partners]
    names = {r["馬番"]: r["馬名"] for r in priced}
    p = evmod.fair_win_probs({r["馬番"]: r["単勝"] for r in priced})

    def annotate(i: int, j: int) -> dict:
        d = {"combo": (i, j), "names": (names.get(i, ""), names.get(j, ""))}
        prob = evmod.wide_prob(p, i, j)
        d["prob"] = round(prob * 100, 1)
        if book and book.wide:
            if rng := book.wide.get(frozenset((i, j))):
                d["odds"] = rng                      # (下限, 上限)
                d["ev"] = evmod.ev(rng[0], prob)     # 下限で保守的に
        return d

    nagashi = [annotate(axis["馬番"], pt["馬番"]) for pt in partners]
    box_nums = [axis["馬番"]] + [pt["馬番"] for pt in partners[:2]]  # 上位3頭ボックス
    box = [annotate(i, j) for i, j in combinations(box_nums, 2)]

    # 軍資金が指定されていれば「何円ずつ」を自信度（的中率）に比例配分
    if bankroll:
        for items in (nagashi, box):
            weights = [d["prob"] for d in items]  # 的中率が高い＝自信あり→多く張る
            for d, s in zip(items, weighted_split(weights, bankroll)):
                d["stake"] = s

    return {
        "axis": {"馬番": axis["馬番"], "馬名": axis["馬名"]},
        "box_nums": box_nums,
        "nagashi": nagashi,
        "box": box,
        "bankroll": bankroll,
    }


def _selections_for(bet_type: str, axis: dict, partners: list[dict]) -> list[tuple]:
    """券種ごとに馬番の組リストを作る（軸＝先頭スコア馬の流し）."""
    a = axis["馬番"]
    ps = [p["馬番"] for p in partners]
    names = {h["馬番"]: h["馬名"] for h in [axis, *partners]}

    if bet_type == "単勝":
        combos = [(a,)]
    elif bet_type == "複勝":
        combos = [(a,)] + ([(ps[0],)] if ps else [])
    elif bet_type in ("馬連", "ワイド"):
        combos = [tuple(sorted((a, p))) for p in ps[:3]]
    elif bet_type == "三連複":
        # 軸1頭流し: 軸 + 相手2頭の組み合わせ
        combos = [tuple(sorted((a, x, y))) for x, y in combinations(ps[:3], 2)]
    elif bet_type == "三連単":
        # 軸1着固定 → 相手で2,3着（順序あり）
        combos = [(a, x, y) for x, y in permutations(ps[:3], 2)]
    else:
        combos = []
    return [(c, tuple(names.get(n, "") for n in c)) for c in combos]


def _allocate(bet_type: str, sels: list[tuple], budget: int) -> list[Bet]:
    """budget を組数で均等配分（100円単位、超過しない）."""
    if not sels or budget < UNIT:
        return []
    n = len(sels)
    per = (budget // n // UNIT) * UNIT
    if per < UNIT:
        # 全点には足りない → 買える点数だけ各100円
        buyable = min(n, budget // UNIT)
        return [
            Bet(bet_type, c, UNIT, horse_names=names)
            for (c, names) in sels[:buyable]
        ]
    return [Bet(bet_type, c, per, horse_names=names) for (c, names) in sels]

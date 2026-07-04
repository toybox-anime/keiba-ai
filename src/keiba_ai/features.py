"""特徴量・指標の計算.

オッズや近走から、LLM に渡す前段の「客観指標」を作る。
ここを充実させるほどレポートの根拠が定量的になる。
将来は LightGBM 等の勝率モデルの出力（score列）もここで合流させる。
"""

from __future__ import annotations

from .models import Race


def implied_prob_from_odds(odds_win: float | None) -> float | None:
    """単勝オッズから市場の暗黙勝率（控除率込み）を出す."""
    if not odds_win or odds_win <= 0:
        return None
    return round(1.0 / odds_win, 4)


def form_score(recent_form: list[str]) -> float:
    """近走着順から調子スコア（0〜1, 高いほど好調）。直近を重視。"""
    if not recent_form:
        return 0.0
    weights = [0.5, 0.3, 0.2]
    s = w_sum = 0.0
    for w, pos in zip(weights, recent_form[:3]):
        try:
            p = int(pos)
        except (TypeError, ValueError):
            continue
        if p <= 0:
            continue
        s += w * (1.0 / p)  # 1着=1.0, 2着=0.5, ...
        w_sum += w
    return round(s / w_sum, 4) if w_sum else 0.0


def build_feature_table(race: Race) -> list[dict]:
    """各馬の指標一覧を返す（レポート生成への入力）."""
    rows: list[dict] = []
    for h in race.horses:
        imp = implied_prob_from_odds(h.odds_win)
        rows.append(
            {
                "馬番": h.num,
                "馬名": h.name,
                "性齢": h.sex_age,
                "騎手": h.jockey,
                "斤量": h.weight,
                "馬体重": h.extra.get("馬体重"),
                "単勝": h.odds_win,
                "人気": h.popularity,
                "市場勝率%": round(imp * 100, 1) if imp is not None else None,
                "近走": h.recent_form,
                "_form": form_score(h.recent_form),
                "model_score": None,  # 将来のMLスコア用プレースホルダ
            }
        )
    # 人気（オッズ）順に並べる
    rows.sort(key=lambda r: (r["単勝"] is None, r["単勝"] or 9999))
    # 人気が未取得なら、単勝オッズの安い順から補完する
    rank = 0
    for r in rows:
        if r["単勝"] is not None:
            rank += 1
            if r["人気"] is None:
                r["人気"] = rank
    return rows


def summarize_market(rows: list[dict]) -> dict:
    """レースの市場全体像（人気の偏り等）を要約."""
    priced = [r for r in rows if r["単勝"]]
    if not priced:
        return {"頭数": len(rows), "オッズ取得": 0}
    fav = priced[0]
    return {
        "頭数": len(rows),
        "オッズ取得": len(priced),
        "1番人気": f'{fav["馬名"]}（{fav["単勝"]}倍）',
        "1番人気の市場勝率%": fav["市場勝率%"],
    }

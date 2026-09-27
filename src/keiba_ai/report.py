"""予想レポート生成（Claude API）.

特徴量テーブルを Claude に渡し、根拠付きの予想レポートを Markdown で生成する。
API キーは環境変数 ANTHROPIC_API_KEY を使用。
キーが無い場合は、LLM を使わずに指標ベースの簡易レポートを返す（オフライン動作）。
"""

from __future__ import annotations

import json
import os

from .betting import build_plan, build_plan_ev, recommend_buy_methods, wide_box
from .features import build_feature_table, summarize_market
from .models import Race
from .odds import OddsBook

_STYLE_GUIDE = {
    "balanced": "堅実さと妙味のバランスを取り、本命・対抗・単穴・連下を提示する。",
    "conservative": "人気・実績を重視し、堅い決着を前提に少点数で狙う。",
    "aggressive": "市場の歪み（オッズの過小評価）を重視し、中穴〜大穴も積極的に拾う。",
}

_SYSTEM = """あなたは地方競馬（楽天競馬）に精通したプロの競馬予想ハンデキャッパーです。
与えられたデータ（各馬の近走着順・騎手・斤量・馬体重・性齢・距離・馬場・オッズ）に、
あなた自身の競馬知識（脚質・展開・枠順・距離/馬場適性・騎手の特徴・人気と実力の乖離など）を
組み合わせて、自分の頭で予想します。GEMINI等に負けない、踏み込んだ本気の予想を書いてください。

進め方:
- まず【あなたの本命予想】として ◎本命 ○対抗 ▲単穴 △連下 を、馬番・馬名つきで選び、
  「なぜその馬か」を近走・展開・適性から具体的に根拠づける（市場オッズの追認ではなく、自分の見解）。
- 次に、別途算出済みの『買い目プラン/自信のある買い目（EV）』を金額つきで提示する。
  金額・組み合わせは確定済みなので数値は変えない。あなたの本命予想と照らして、買うべきか一言添える。

ルール:
- データに無い"具体的な数字"（タイムや過去着順など）を捏造しない。ただし脚質や展開の推論は自由。
- 軍資金を超える投資は提案しない。
- 最後に必ず『※馬券は自己責任・20歳以上。余裕資金の範囲で。』を入れる。"""


def generate_report(
    race: Race,
    *,
    model: str,
    max_tokens: int,
    style: str,
    bankroll: int | None = None,
    odds_book: OddsBook | None = None,
    win_model=None,
) -> str:
    rows = build_feature_table(race)

    # MLモデルがあれば勝率を予測して特徴量表に反映
    model_probs = None
    if win_model is not None:
        model_probs = win_model.predict_race(race)
        by_num = {r["馬番"]: r for r in rows}
        for num, prob in model_probs.items():
            if num in by_num:
                by_num[num]["model_score"] = round(prob, 4)
                by_num[num]["モデル勝率%"] = round(prob * 100, 1)

    market = summarize_market(rows)
    plan = None
    if bankroll:
        if odds_book is not None and odds_book.has_combos():
            plan = build_plan_ev(rows, odds_book, bankroll, style, win_probs=model_probs)  # EV/ケリー配分
        else:
            plan = build_plan(rows, bankroll, style)                 # 均等フォーメーション

    wide = wide_box(rows, odds_book, bankroll=bankroll)  # 一番のおすすめ＝ワイド3頭ボックス（金額つき）
    recommend = (  # 全券種比較＋おすすめ＋自信のある買い目（金額つき）
        recommend_buy_methods(rows, odds_book, model_probs, bankroll=bankroll) if odds_book else None
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _offline_report(race, rows, market, plan, wide, recommend)

    try:
        import anthropic
    except ImportError:
        return _offline_report(race, rows, market, plan, wide, recommend)

    client = anthropic.Anthropic(api_key=api_key)
    style_text = _STYLE_GUIDE.get(style, _STYLE_GUIDE["balanced"])
    payload = {
        "レース": {
            "race_id": race.race_id,
            "レース名": race.title,
            "距離m": race.distance_m,
            "馬場": race.going,
        },
        "市場サマリ": market,
        "出走馬指標": rows,
        "買い目プラン": plan.to_dict() if plan else "（軍資金未指定のため未生成）",
        "券種比較とおすすめ": recommend or "（オッズ未取得）",
        "一番のおすすめ_ワイド3頭ボックス": _compact_box_line(wide),
    }
    user_msg = (
        f"方針: {style_text}\n\n"
        "以下のデータから、あなた自身の本気の予想レポートを作成してください。\n"
        "構成:\n"
        "①レース概観（距離・馬場・想定される展開・隊列）\n"
        "②【あなたの本命予想】◎○▲△を馬番・馬名つきで選び、近走・脚質・展開・適性から根拠を述べる\n"
        "   （市場の人気順をなぞらず、人気と実力の乖離＝妙味も指摘する）\n"
        "③【一番のおすすめ買い目】ワイド3頭ボックスを金額つきで（判定が見送りなら見送りと明記）\n"
        "④【参考】券種比較の妙味（上級者向け・少額で）\n"
        "⑤買い目プランの金額配分（提示済みの金額をそのまま示す）\n"
        "⑥リスクと注意点\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


# Gem（カスタムGemini）に1回だけ設定する指示文（圧縮版・精度維持）。
GEM_INSTRUCTIONS = """地方競馬(楽天競馬)のプロ予想家。貼られた「出走馬データ＋EV分析」で毎回:
①予想:◎○▲△を馬番・馬名で。根拠は近走・脚質・展開・距離/馬場適性・騎手から簡潔に(人気のなぞり禁止、人気と実力の乖離=妙味も指摘)。
②買い目:安定重視でワイド3頭ボックス(3点)が本線。貼付の推奨ボックスを予想で入替えるか判断し何円ずつ(合計超過不可)。他券種は参考。
③リスク:危険な人気馬、妙味なしなら「見送り」。
根拠つき・簡潔に。末尾「※馬券は自己責任・20歳以上」。"""


def gem_instructions() -> str:
    return GEM_INSTRUCTIONS


def _horse_rows_md(rows: list[dict]) -> list[str]:
    out = [
        "| 馬番 | 馬名 | 性齢 | 騎手 | 斤量 | 単勝 | 人気 | 近走5走 | 馬体重 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        kin = "-".join(r.get("近走") or []) or "-"
        out.append(
            f'| {r["馬番"]} | {r["馬名"]} | {r.get("性齢") or "-"} | {r.get("騎手") or "-"} | '
            f'{r.get("斤量") or "-"} | {r["単勝"] or "-"} | {r["人気"] or "-"} | {kin} | {r.get("馬体重") or "-"} |'
        )
    return out


def _compact_value_line(recommend: dict | None) -> str:
    conf = (recommend or {}).get("confident") or []
    if not conf:
        return "EV1.0超の妙味なし（見送り寄り）"
    return " / ".join(f'{o["券種"]}{o["組"]}(EV{o["EV"]},的中{o["的中率%"]}%)' for o in conf[:5])


def build_gemini_prompt(
    race: Race, *, bankroll: int | None = None, style: str = "balanced",
    odds_book: OddsBook | None = None, gem_mode: bool = False, compact: bool = False,
    feedback: str = "",
) -> str:
    """Geminiに渡す『予想依頼文』を生成する.

    gem_mode=True: 指示文を省く（Gem側に指示済み）。
    compact=True: EV表を1行に圧縮しトークンを大幅削減（精度の源=出走馬データは維持）。
    """
    rows = build_feature_table(race)
    market = summarize_market(rows)
    plan = None
    if bankroll:
        if odds_book is not None and odds_book.has_combos():
            plan = build_plan_ev(rows, odds_book, bankroll, style)
        else:
            plan = build_plan(rows, bankroll, style)
    wide = wide_box(rows, odds_book, bankroll=bankroll)
    recommend = recommend_buy_methods(rows, odds_book, bankroll=bankroll) if odds_book else None

    # --- コンパクト版（トークン節約）: 出走馬 + 妙味1行 + 短い依頼 ---
    if compact:
        budget = f"軍資金{bankroll:,}円で配分。" if bankroll else ""
        out = [
            f"# {race.title or race.race_id} ダ{race.distance_m or '?'}m {race.going or ''} {len(race.horses)}頭",
            "# 出走馬（近走=左が直近）",
            *_horse_rows_md(rows),
            "",
            f"# 本線=ワイド3頭ボックス: {_compact_box_line(wide)}",
            f"# 他券種の妙味(EV>1.0・参考): {_compact_value_line(recommend)}",
            "",
        ]
        if feedback:
            out.append(feedback)
        out += [
            f"# 依頼: ◎○▲△を馬番名つきで（近走・展開・適性・人気妙味から根拠簡潔に）。"
            f"買い目はワイド3頭ボックス中心で何円ずつ（あなたの予想で3頭を入替えてもよい）。{budget}"
            "危険な人気馬・見送り判断も一言。",
            "最後に必ず1行:【PICKS】◎<馬番> ○<馬番> ▲<馬番>（採点用。数字のみ）",
        ]
        return "\n".join(out)

    if gem_mode:
        out: list[str] = [f"# レース: {race.title or race.race_id}"]
    else:
        out = [
            "あなたは地方競馬（楽天競馬）に精通したプロの競馬予想家です。",
            "以下のレースデータと、ツールが計算した期待値(EV)分析を踏まえ、本気で予想してください。",
            "",
            f"# レース: {race.title or race.race_id}",
        ]
    out += [
        f"- 距離 {race.distance_m or '不明'}m / 馬場 {race.going or '不明'} / {len(race.horses)}頭立て",
        "",
        "# 出走馬（近走は左が直近の着順）",
        *_horse_rows_md(rows),
    ]

    if wide:
        out += _render_wide_box(wide)
    if recommend:
        out.append("")
        out.append("# ツールのEV分析（参考。割高=妙味のある買い目）")
        out += _render_recommendation(recommend)
    if plan and plan.bets:
        out += _render_plan(plan)

    budget_line = f"軍資金は{bankroll:,}円です。" if bankroll else "軍資金は任意で構いません。"
    if gem_mode:
        # Gem側に指示済みなので短く依頼するだけ
        out += ["", f"↑このレースを予想して、買い目を提案してください。{budget_line}"]
    else:
        out += [
            "",
            "# お願い（必ず守ってください）",
            "1. **あなた自身の本命予想**：◎本命 ○対抗 ▲単穴 △連下 を馬番・馬名つきで選び、",
            "   近走・脚質・想定展開・距離/馬場適性・騎手から根拠を述べる（人気順のなぞりはNG。妙味も指摘）。",
            f"2. **買い目の提案**：{budget_line}安定重視のため**ワイド3頭ボックス（3点）を本線**に、"
            "あなたの予想で3頭を入れ替えるべきか判断し「何円ずつ」買うか具体的に。合計が軍資金を超えないこと。",
            "3. **リスク**：危険な人気馬や、見送り推奨ならその旨も。",
            "",
            "※馬券は自己責任・20歳以上。",
        ]
    return "\n".join(out)


def _offline_report(race: Race, rows: list[dict], market: dict, plan=None, wide=None, recommend=None) -> str:
    """API キー無しでも動く簡易レポート（指標＋おすすめ＋買い目＋ワイド）."""
    lines = [
        f"# 予想メモ（オフライン版） {race.title or race.race_id}",
        "",
        f"- 距離: {race.distance_m or '不明'}m / 馬場: {race.going or '不明'}",
        f"- 市場サマリ: {market}",
        "",
        "## 出走馬指標",
        "| 馬番 | 馬名 | 性齢 | 騎手 | 単勝 | 人気 | 市場勝率% | 近走 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        kin = "-".join(r.get("近走") or []) or "-"
        lines.append(
            f'| {r["馬番"]} | {r["馬名"]} | {r.get("性齢") or "-"} | {r.get("騎手") or "-"} | '
            f'{r["単勝"] or "-"} | {r["人気"] or "-"} | {r["市場勝率%"] or "-"} | {kin} |'
        )

    if wide:
        lines += _render_wide_box(wide)
    if recommend:
        lines += _render_recommendation(recommend)
    if plan and plan.bets:
        lines += _render_plan(plan)

    lines += [
        "",
        "> ANTHROPIC_API_KEY を設定すると Claude による根拠付きレポートを生成します。",
        "",
        "※馬券は自己責任・20歳以上。余裕資金の範囲で。",
    ]
    return "\n".join(lines)


def _render_plan(plan) -> list[str]:
    out = [
        "",
        f"## 買い目プラン（軍資金 {plan.bankroll:,}円 / {plan.style}）",
        "",
        f"- 本命: {plan.axis['馬番']} {plan.axis['馬名']}（単勝{plan.axis['単勝']}）",
    ]
    if plan.partners:
        out.append("- 相手: " + "、".join(f"{p['馬番']}{p['馬名']}" for p in plan.partners))
    out += [
        "",
        "| 券種 | 組み合わせ | 金額 | メモ |",
        "|---|---|---|---|",
    ]
    for b in plan.bets:
        combo = "-".join(map(str, b.horses))
        out.append(f"| {b.bet_type} | {combo} | {b.stake:,}円 | {b.note} |")
    out += [
        "",
        f"**投資合計 {plan.total_stake:,}円 / 軍資金 {plan.bankroll:,}円（余り {plan.bankroll - plan.total_stake:,}円）**",
        "",
        f"> {plan.note}",
    ]
    return out


def _render_recommendation(rec: dict) -> list[str]:
    b = rec["best"]
    out = [
        "",
        "## 📊 参考：券種ごとの妙味（上級者向け・買うなら少額で）",
        "",
        f"### ★ {b['券種']}　{b['組']}（{b['馬名']}）",
        "",
        f"- オッズ {b['オッズ']} / 的中率(目安) {b['的中率%']}% / 期待値(EV) {b['EV']}",
        f"- {rec['reason']}",
    ]

    # 自信のある買い目（EVプラス＝妙味あり）を、券種を問わず金額つきで提示
    confident = rec.get("confident") or []
    has_money = any("stake" in o for o in confident)
    if confident:
        out += [
            "",
            "### ✅ 自信のある買い目（妙味のある券種を厳選）",
            "",
            ("| 券種 | 組み合わせ | 馬名 | オッズ | 的中率 | EV |" + (" 金額 |" if has_money else "")),
            ("|---|---|---|---|---|---|" + ("---|" if has_money else "")),
        ]
        for o in confident:
            row = f'| {o["券種"]} | {o["組"]} | {o["馬名"]} | {o["オッズ"]} | {o["的中率%"]}% | {o["EV"]} |'
            if has_money:
                row += f' {o.get("stake", 0):,}円 |'
            out.append(row)
        if has_money:
            total = sum(o.get("stake", 0) for o in confident)
            out += [
                ("| **合計** | | | | | |" + f" **{total:,}円** |"),
                "",
                "> EVプラス（期待値1.0超）の券種だけを選び、自信度（EV×当てやすさ）で金額配分。",
            ]
    else:
        out += ["", "### ✅ 自信のある買い目", "", "> EVプラスの買い目はありません（理論上は見送り推奨）。"]

    out += [
        "",
        "### 券種ごとの比較（各券種のベスト1点）",
        "",
        "| 券種 | 組み合わせ | 馬名 | オッズ | 的中率 | EV | 向いている人 |",
        "|---|---|---|---|---|---|---|",
    ]
    for o in rec["options"]:
        mark = "★" if o is b else ""
        out.append(
            f'| {mark}{o["券種"]} | {o["組"]} | {o["馬名"]} | {o["オッズ"]} | '
            f'{o["的中率%"]}% | {o["EV"]} | {o["向き"]} |'
        )
    out += [
        "",
        "> EVは期待値（1.0超で理論上プラス）。的中率と配当のバランスで「★」を総合おすすめに選定。",
        "> 当てやすさ優先なら上の方の券種、一発の大きさ優先なら下の方の券種。",
    ]
    return out


_BOX_BADGE = {"buy": "✅ 買い", "recheck": "⚠️ 直前に再確認", "skip": "❌ 見送り推奨", "noodds": "⏳ オッズ未取得"}


def _box_label(box: dict) -> str:
    return "・".join(map(str, box["box_nums"]))


def _render_wide_box(wb: dict) -> list[str]:
    """一番のおすすめ＝ワイド3頭ボックス（3点）."""
    m, names = wb["main"], wb["names"]
    money = bool(wb.get("bankroll"))
    horses = " / ".join(f"{n} {names.get(n, '')}" for n in m["box_nums"])
    kind = "人気上位3頭" if wb["is_top3"] else "上位5頭から妙味のある3頭"
    stats = f"的中率(どれか1点当たる) {m['hit']}%"
    if "ev" in m:
        stats += f" / 合成オッズ {m['synth']}倍 / 期待値(EV) {m['ev']}"
    out = [
        "",
        f"## 🎯 一番のおすすめ：ワイド3頭ボックス（{_box_label(m)}）　{_BOX_BADGE[wb['status']]}",
        "",
        f"- 3頭（{kind}）: {horses}",
        f"- {stats}",
        f"- {wb['reason']}",
        "",
        "| 組み合わせ | 馬名 | オッズ | 的中率 |" + (" 金額 | 当たったら |" if money else ""),
        "|---|---|---|---|" + ("---|---|" if money else ""),
    ]
    for d in m["items"]:
        i, j = d["combo"]
        odds = f'{d["odds"][0]}-{d["odds"][1]}' if "odds" in d else "-"
        row = f'| {i}-{j} | {d["names"][0]} − {d["names"][1]} | {odds} | {d["prob"]}% |'
        if money:
            pay = f'{d["payout"]:,}円〜' if "payout" in d else "-"
            row += f' {d.get("stake", 0):,}円 | {pay} |'
        out.append(row)
    if money:
        total = sum(d.get("stake", 0) for d in m["items"])
        out.append(f"| **合計** | | | | **{total:,}円** | |")
    out += [
        "",
        "> 3頭のうち2頭が3着以内なら的中（1頭崩れてもOK）。3頭とも3着以内なら3点すべて的中。",
        "> 金額は「どの1点が当たっても払戻がほぼ同じ」になる配分。楽天競馬では「通常」投票で3点を"
        "個別に購入（同額でよければ「ボックス」投票でも可）。",
    ]
    if wb["alts"]:
        out += [
            "",
            "### 他の候補ボックス",
            "",
            "| 3頭 | 的中率 | 合成オッズ | EV |",
            "|---|---|---|---|",
        ]
        out += [f'| {_box_label(b)} | {b["hit"]}% | {b["synth"]}倍 | {b["ev"]} |' for b in wb["alts"]]
    return out


def _compact_box_line(wb: dict | None) -> str:
    if not wb:
        return "なし"
    m = wb["main"]
    stake = "・".join(
        f'{d["combo"][0]}-{d["combo"][1]}' + (f'{d["stake"]}円' if "stake" in d else "") for d in m["items"]
    )
    extra = f",合成{m['synth']}倍,EV{m['ev']}" if "ev" in m else ""
    return f"{_box_label(m)}（{_BOX_BADGE[wb['status']]},的中{m['hit']}%{extra}）{stake}"

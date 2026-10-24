"""紙上の買い目（ペーパートレード）の記録と答え合わせ.

predict / scan で出た「買い」をその時のオッズつきで記録し、後で払戻ページと照合して
券種（戦略）ごとの的中率・回収率を出す。実際に買ったかどうかとは無関係。

戦略:
  ワイド3頭BOX   … wide_box が「買い」判定のとき、3点を表示どおりの金額で
  複勝・馬連・三連複(EV妙味) … 券種比較で EV1.0〜2.0 のベスト1点を100円（EV2超は未成熟オッズとして除外）
  複勝(1番人気)  … 比較の基準。毎回 1番人気の複勝を100円

1レース×1戦略=1件。同じレースを何度も分析したら、未採点の分は最新（発走に近い）記録で上書きする。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .betting import BOX_EV_SUSPECT
from .ev import LAMBDA2, LAMBDA3, WIN_ALPHA
from .result import combo_key

KIND_OF = {"複勝": "place", "ワイド": "wide", "馬連": "quinella", "三連複": "trio"}
EV_STRATEGIES = ("複勝", "馬連", "三連複")
BASELINE = "複勝(1番人気)"
BOX = "ワイド3頭BOX"
MODEL_TAG = f"着順補正 α{WIN_ALPHA}/λ{LAMBDA2}/{LAMBDA3}"  # 確率モデルが変わったら成績を分けて集計する
STRATEGY_ORDER = (BOX, "複勝(EV妙味)", "馬連(EV妙味)", "三連複(EV妙味)", BASELINE)


def build_entries(race_id: str, label: str, rows: list[dict], book, wb: dict | None, rec: dict | None,
                  *, source: str) -> list[dict]:
    """その時点の分析結果から記録すべき紙上の買い目を作る."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    base = {"date": f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}", "race_id": race_id,
            "label": label, "recorded_at": now, "source": source, "graded": False, "model": MODEL_TAG}
    out: list[dict] = []

    if wb and wb["status"] == "buy":
        m = wb["main"]
        out.append({**base, "strategy": BOX, "kind": "wide", "ev": m.get("ev"), "points": [
            {"combo": combo_key("wide", d["combo"]), "stake": d.get("stake") or 100, "odds": d["odds"][0]}
            for d in m["items"]
        ]})

    for o in (rec or {}).get("options", []):
        # EV2超は売上の薄いオッズの疑い（ワイドBOXの「直前に再確認」と同じ扱い）なので記録しない
        if o["券種"] in EV_STRATEGIES and 1.0 <= o["EV"] <= BOX_EV_SUSPECT:
            kind = KIND_OF[o["券種"]]
            out.append({**base, "strategy": f"{o['券種']}(EV妙味)", "kind": kind, "ev": o["EV"], "points": [
                {"combo": combo_key(kind, re.findall(r"\d+", o["組"])), "stake": 100, "odds": o["オッズ"]}
            ]})

    priced = [r for r in rows if r.get("単勝")]
    if priced and book and book.place:
        fav = min(priced, key=lambda r: r["単勝"])["馬番"]
        if fav in book.place:
            out.append({**base, "strategy": BASELINE, "kind": "place", "ev": None, "points": [
                {"combo": str(fav), "stake": 100, "odds": book.place[fav][0]}
            ]})
    return out


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return rows


def save(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def record(new: list[dict], path: Path, *, race_id: str | None = None) -> int:
    """紙上の買い目を記録する。同じレースの未採点分は今回の分析で置き換える（買いが消えた戦略も消す）."""
    rows = load(path)
    rid = race_id or (new[0]["race_id"] if new else None)
    if rid is None:
        return 0
    rows = [r for r in rows if r["race_id"] != rid or r.get("graded")]
    done = {(r["race_id"], r["strategy"]) for r in rows}
    rows += [e for e in new if (e["race_id"], e["strategy"]) not in done]
    save(rows, path)
    return len(new)


def grade(path: Path, fetch_payouts, *, today: str | None = None) -> int:
    """未採点の記録を払戻と照合する。fetch_payouts(race_id) → parse_payouts の結果."""
    rows = load(path)
    today = today or datetime.now().strftime("%Y-%m-%d")
    cache: dict[str, dict] = {}
    done = 0
    for r in rows:
        if r.get("graded") or r["date"] > today:
            continue
        rid = r["race_id"]
        if rid not in cache:
            try:
                cache[rid] = fetch_payouts(rid)
            except Exception:  # noqa: BLE001
                cache[rid] = {}
        pay = cache[rid]
        if not pay.get("win"):
            continue  # まだ確定していない（または取得失敗）
        table = pay.get(r["kind"], {})
        ret = 0
        for pt in r["points"]:
            pt["payout"] = table.get(pt["combo"], 0) * pt["stake"] // 100
            ret += pt["payout"]
        r["invest"] = sum(pt["stake"] for pt in r["points"])
        r["return"] = ret
        r["hit"] = ret > 0
        r["graded"] = True
        done += 1
    save(rows, path)
    return done


def summarize(rows: list[dict]) -> list[dict]:
    """戦略ごとの成績（採点済みのみ）."""
    out = []
    names = [s for s in STRATEGY_ORDER if any(r["strategy"] == s for r in rows)]
    for s in names:
        g = [r for r in rows if r["strategy"] == s and r.get("graded")]
        pending = sum(1 for r in rows if r["strategy"] == s and not r.get("graded"))
        inv = sum(r["invest"] for r in g)
        ret = sum(r["return"] for r in g)
        out.append({
            "戦略": s, "レース数": len(g), "的中": sum(1 for r in g if r["hit"]),
            "投資": inv, "払戻": ret, "回収率%": round(ret / inv * 100, 1) if inv else None, "未確定": pending,
        })
    return out


def _table(rows: list[dict]) -> list[str]:
    lines = [
        "| 戦略 | レース数 | 的中率 | 投資 | 払戻 | 回収率 | 未確定 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summarize(rows):
        n = s["レース数"]
        hit = f"{s['的中'] / n * 100:.0f}%" if n else "-"
        roi = f"**{s['回収率%']}%**" if s["回収率%"] is not None else "-"
        lines.append(f"| {s['戦略']} | {n} | {hit} | {s['投資']:,}円 | {s['払戻']:,}円 | {roi} | {s['未確定']} |")
    return lines


def render(rows: list[dict]) -> str:
    current = [r for r in rows if r.get("model") == MODEL_TAG]
    old = [r for r in rows if r.get("model") != MODEL_TAG]
    lines = ["# 紙上の買い目 成績（ペーパートレード）", "", f"## 今の確率モデル（{MODEL_TAG}）", ""]
    lines += _table(current) if current else ["（まだ記録なし）"]
    if old:
        lines += ["", "## 参考：補正前のモデル（人気馬の2・3着を過大評価していた時期）", ""] + _table(old)
    lines += [
        "",
        "> 回収率100%超で理論上プラス。30〜50レースまでは運の影響が大きいので、数字は参考程度に。",
        "> 三連複は的中率が低く、判断には数百レース必要。複勝(1番人気)は比較用の基準。",
        "",
        "## 直近の記録",
        "",
        "| 日付 | レース | 戦略 | 買い目 | 投資 | 払戻 |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["date"], x["race_id"]), reverse=True)[:40]:
        combos = " ".join(f'{p["combo"]}({p["stake"]})' for p in r["points"])
        ret = f'{r["return"]:,}円' if r.get("graded") else "未確定"
        lines.append(f'| {r["date"]} | {r["label"]} | {r["strategy"]} | {combos} | '
                     f'{sum(p["stake"] for p in r["points"]):,}円 | {ret} |')
    return "\n".join(lines)

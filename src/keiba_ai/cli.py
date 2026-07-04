"""コマンドラインインターフェース.

使い方::

    # 設定の読み込みと依存は requirements.txt を pip install 済み前提
    python -m keiba_ai.cli predict --date 2026-06-26 --track 大井 --race 11
    python -m keiba_ai.cli predict --race-id 202606264400110000
    python -m keiba_ai.cli calibrate --race-id 202606264400110000   # 実HTML保存（パーサ較正用）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

from . import parser as race_parser
from . import result as race_result
from .betting import recommend_buy_methods
from .dataset import append_rows, dataset_stats, race_result_to_rows
from .features import build_feature_table
from .model import WinModel
from .odds import OddsBook, parse_into
from .raceid import race_id_for
from .report import build_gemini_prompt, gem_instructions, generate_report
from .schedule import parse_meetings, resolve_meeting_id
from .scraper import PoliteScraper
from .train import train as train_model

# EVモードで取得する券種（全券種）
EV_ODDS_KINDS = ("tanfuku", "umafuku", "umatan", "wide", "sanrenfuku", "sanrentan")
# スキャンは主要な連系のみ取得して高速化（単勝は出馬表から取得）
SCAN_ODDS_KINDS = ("umafuku", "wide", "sanrenfuku")

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_race_id(args, scraper: PoliteScraper, cfg: dict) -> str:
    """RACEID を解決する。--race-id 優先。無ければ本日開催一覧から競馬場名で引く."""
    if args.race_id:
        return args.race_id
    if not (args.track and args.race):
        sys.exit("--race-id か、--track と --race を指定してください")
    print("[0] 本日の開催一覧を取得中...", file=sys.stderr)
    landing = scraper.get(
        scraper.race_card_landing_url(),
        max_age_sec=cfg["scraper"]["cache_ttl_hours"] * 3600,
    )
    meetings = parse_meetings(landing)
    try:
        mid = resolve_meeting_id(meetings, args.track)
    except KeyError as e:
        sys.exit(f"❌ {e.args[0]}\n   過去のレースは --race-id <18桁> で指定できます。")
    return race_id_for(mid, args.race)


def _make_scraper(cfg: dict, *, delay_override: float | None = None) -> PoliteScraper:
    s = cfg["scraper"]
    return PoliteScraper(
        base_url=s["base_url"],
        cache_dir=ROOT / cfg["paths"]["cache_dir"],
        crawl_delay_sec=s["crawl_delay_sec"] if delay_override is None else delay_override,
        timeout_sec=s["timeout_sec"],
        max_retries=s["max_retries"],
        user_agent=s["user_agent"],
    )


def cmd_predict(args, cfg: dict) -> None:
    # predictは1レースだけ見る直前用途なので短い間隔を使う（締め切りに間に合わせる）
    scraper = _make_scraper(cfg, delay_override=cfg["scraper"].get("interactive_delay_sec", 5))
    race_id = _resolve_race_id(args, scraper, cfg)

    print(f"[1/3] 出馬表を取得中... ({race_id})", file=sys.stderr)
    html = scraper.get(scraper.race_card_url(race_id), max_age_sec=cfg["scraper"]["cache_ttl_hours"] * 3600)
    race = race_parser.parse_race_card(html, race_id)
    print(f"      → {len(race.horses)} 頭を検出", file=sys.stderr)

    if not race.horses:
        print("⚠️ 出走馬を検出できませんでした。`calibrate` でHTML構造を確認し parser.py を較正してください。", file=sys.stderr)

    # 生データを保存
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{race_id}.json").write_text(
        json.dumps(race.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[2/3] 指標を計算中...", file=sys.stderr)
    _ = build_feature_table(race)

    # EVモード: 組み合わせオッズを取得して OddsBook を作る
    odds_book = None
    if args.ev and args.budget:
        odds_book = _load_odds_book(scraper, race_id, cfg, fresh=args.fresh)
        print(f"[EV] オッズ取得時刻: {datetime.now():%H:%M:%S}（発走直前ほど精度向上）", file=sys.stderr)

    reports_dir = ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    style = args.style or cfg["report"]["style"]

    # --- Geminiモード: 貼り付け用の予想依頼文を出力 ---
    if args.gemini or args.gem:
        gem = bool(args.gem)
        prompt = build_gemini_prompt(
            race, bankroll=args.budget, style=style, odds_book=odds_book, gem_mode=gem
        )
        out = reports_dir / f"{race_id}_gemini.txt"
        out.write_text(prompt, encoding="utf-8")
        if gem:
            gi = reports_dir / "gem_instructions.txt"
            gi.write_text(gem_instructions(), encoding="utf-8")
            print(f"\n✅ Gem用データ: {out}", file=sys.stderr)
            print(f"（初回のみ）Gemの指示文 → {gi} を Gemini の Gem 設定に貼って作成してください。", file=sys.stderr)
            print("　以降はこのデータを、作成したGemに貼るだけでOK ↓\n", file=sys.stderr)
        else:
            print(f"\n✅ Gemini用の依頼文: {out}", file=sys.stderr)
            print("　↓これを丸ごとコピーして Gemini チャットに貼ってください↓\n", file=sys.stderr)
        print(prompt)
        return

    # --- 通常モード: Claudeでレポート生成 ---
    _maybe_retrain()  # データが十分なら（インタラクティブ環境で）モデル自動更新
    win_model = WinModel.load()
    if win_model is not None:
        print("[EV] 学習済み勝率モデルを使用", file=sys.stderr)

    print("[3/3] レポートを生成中...", file=sys.stderr)
    rep = cfg["report"]
    report_md = generate_report(
        race,
        model=rep["model"],
        max_tokens=rep["max_tokens"],
        style=style,
        bankroll=args.budget,
        odds_book=odds_book,
        win_model=win_model,
    )

    out = reports_dir / f"{race_id}.md"
    out.write_text(report_md, encoding="utf-8")
    print(f"\n✅ レポート: {out}\n", file=sys.stderr)
    print(report_md)


def _load_odds_book(
    scraper: PoliteScraper, race_id: str, cfg: dict, *, fresh: bool = False, kinds=None
) -> OddsBook:
    """各券種オッズを取得・解析して OddsBook を返す.

    オッズは発走直前まで変動するため、短命キャッシュ（既定10分）を使う。
    fresh=True ならキャッシュを無視して必ず最新を取り直す。
    kinds で取得する券種を絞れる（スキャン高速化用）。
    """
    book = OddsBook()
    max_age = 0 if fresh else cfg["scraper"].get("odds_cache_minutes", 10) * 60
    for kind in (kinds or EV_ODDS_KINDS):
        print(f"[EV] {kind} オッズ取得中...{'(最新取得)' if fresh else ''}", file=sys.stderr)
        html = scraper.get(scraper.odds_url(race_id, kind), max_age_sec=max_age)
        try:
            parse_into(book, kind, html)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ {kind} の解析に失敗: {e}", file=sys.stderr)
    print(
        f"[EV] 取得: 単勝{len(book.win)} 馬連{len(book.quinella)} 馬単{len(book.exacta)} "
        f"ワイド{len(book.wide)} 三連複{len(book.trio)} 三連単{len(book.trifecta)}",
        file=sys.stderr,
    )
    return book


def cmd_calibrate(args, cfg: dict) -> None:
    """パーサ較正用に実HTMLを保存する."""
    scraper = _make_scraper(cfg)
    race_id = _resolve_race_id(args, scraper, cfg)
    url = scraper.race_card_url(race_id)
    print(f"取得: {url}", file=sys.stderr)
    html = scraper.get(url, max_age_sec=cfg["scraper"]["cache_ttl_hours"] * 3600)
    out = ROOT / cfg["paths"]["raw_dir"] / f"{race_id}_card.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"保存: {out} ({len(html):,} bytes)", file=sys.stderr)
    print("→ このHTMLを開き、出走表テーブルのセレクタ／列順を parser.py に反映してください。", file=sys.stderr)


def cmd_scan(args, cfg: dict) -> None:
    """本日の全レースを分析し、妙味（EVプラス）のあるレース・買い目をランキング表示する."""
    scraper = _make_scraper(cfg, delay_override=cfg["scraper"].get("interactive_delay_sec", 5))

    if args.track or args.race_id:
        meetings = {(args.track or "指定開催"): _resolve_meeting_id(args, scraper, cfg)}
    else:
        landing = scraper.get(scraper.race_card_landing_url(), max_age_sec=0)
        meetings = parse_meetings(landing)
    if not meetings:
        sys.exit("本日の開催が見つかりません。")

    maxr = args.races or 12
    print(f"[scan] 対象: {list(meetings)} / 各最大{maxr}R を分析（数分かかります）", file=sys.stderr)
    card_age = cfg["scraper"]["cache_ttl_hours"] * 3600
    found, skipped, day = [], [], ""

    for track, mid in meetings.items():
        day = f"{mid[0:4]}-{mid[4:6]}-{mid[6:8]}"
        for n in range(1, maxr + 1):
            rid = race_id_for(mid, n)
            try:
                race = race_parser.parse_race_card(
                    scraper.get(scraper.race_card_url(rid), max_age_sec=card_age), rid
                )
            except Exception:  # noqa: BLE001
                race = None
            if not race or not race.horses:
                break  # この開催はここで終わり
            book = _load_odds_book(scraper, rid, cfg, fresh=args.fresh, kinds=SCAN_ODDS_KINDS)
            rec = recommend_buy_methods(build_feature_table(race), book, bankroll=args.budget)
            label = f"{track}{n}R"
            if rec and rec.get("confident"):
                b = rec["best"]
                found.append({"label": label, "rid": rid, "best": b, "n": len(rec["confident"])})
                print(f"  {label}: ★妙味あり {b['券種']} {b['組']} EV{b['EV']}", file=sys.stderr)
            else:
                skipped.append(label)
                print(f"  {label}: 妙味なし", file=sys.stderr)

    found.sort(key=lambda x: x["best"]["_score"], reverse=True)
    report = _render_scan(day, list(meetings), found, skipped, args.budget)
    out = ROOT / cfg["paths"]["reports_dir"] / f"scan_{day}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\n✅ 狙い目レポート: {out}\n", file=sys.stderr)
    print(report)


def _render_scan(day, tracks, found, skipped, budget) -> str:
    lines = [
        f"# 本日の狙い目 {day}",
        f"- 分析対象: {' / '.join(tracks)} ／ 妙味あり {len(found)}レース・見送り {len(skipped)}レース",
        "",
        "## 🎯 狙い目ランキング（EVプラス＝妙味のあるレース）",
    ]
    if found:
        lines += [
            "| 順位 | レース | 一番のおすすめ | 馬名 | オッズ | 的中率 | EV | 妙味数 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for i, f in enumerate(found, 1):
            b = f["best"]
            lines.append(
                f'| {i} | {f["label"]} | {b["券種"]} {b["組"]} | {b["馬名"]} | '
                f'{b["オッズ"]} | {b["的中率%"]}% | {b["EV"]} | {f["n"]} |'
            )
        lines += [
            "",
            "### 詳しい買い目を見る（上位レースを深掘り）",
            "```",
            f"python keiba.py predict --race-id {found[0]['rid']} "
            f"{'--budget '+str(budget)+' ' if budget else ''}--ev --fresh --gemini",
            "```",
        ]
    else:
        lines.append("（妙味のあるレースはありませんでした＝本日は無理に買わないのが正解）")

    if skipped:
        lines += ["", "## 見送り（EVプラスなし）", "・".join(skipped)]
    lines += ["", "※馬券は自己責任・20歳以上。EVは目安で、的中を保証しません。"]
    return "\n".join(lines)


def _collect_one(scraper: PoliteScraper, race_id: str, cfg: dict, *, quiet: bool = False) -> dict:
    """1レースの出馬表＋結果を取得しデータセットに追記。結果dictを返す."""
    max_age = cfg["scraper"]["cache_ttl_hours"] * 3600
    card_html = scraper.get(scraper.race_card_url(race_id), max_age_sec=max_age)
    race = race_parser.parse_race_card(card_html, race_id)
    if not race.horses:
        return {"race_id": race_id, "horses": 0, "result": 0, "added": 0, "exists": False}

    res_html = scraper.get(scraper.result_url(race_id), max_age_sec=max_age)
    result = race_result.parse_result(res_html)
    if not result:
        if not quiet:
            (ROOT / cfg["paths"]["raw_dir"] / f"{race_id}_result.html").write_text(res_html, encoding="utf-8")
        return {"race_id": race_id, "horses": len(race.horses), "result": 0, "added": 0, "exists": True}

    added = append_rows(race_result_to_rows(race, result), ROOT / "data/dataset.jsonl")
    return {"race_id": race_id, "horses": len(race.horses), "result": len(result), "added": added, "exists": True}


def cmd_collect(args, cfg: dict) -> None:
    """出馬表＋結果を取得し、学習データセットに追記する（1レース）."""
    scraper = _make_scraper(cfg)
    race_id = _resolve_race_id(args, scraper, cfg)
    print(f"[collect] 取得 {race_id}", file=sys.stderr)
    r = _collect_one(scraper, race_id, cfg)
    print(f"  → 出走 {r['horses']}頭 / 着順取得 {r['result']}頭 / 追記 {r['added']}行", file=sys.stderr)
    if r["exists"] and not r["result"]:
        print(f"⚠️ 着順を取得できませんでした（HTML: data/raw/{race_id}_result.html）。", file=sys.stderr)
    print(f"[collect] 現状: {dataset_stats(ROOT / 'data/dataset.jsonl')}", file=sys.stderr)


def _resolve_meeting_id(args, scraper: PoliteScraper, cfg: dict) -> str:
    """開催基準ID（末尾00）を解決する。--race-id か --track で指定."""
    if args.race_id:
        return args.race_id[:-2] + "00"
    if not args.track:
        sys.exit("--race-id か --track を指定してください")
    landing = scraper.get(
        scraper.race_card_landing_url(), max_age_sec=cfg["scraper"]["cache_ttl_hours"] * 3600
    )
    try:
        return resolve_meeting_id(parse_meetings(landing), args.track)
    except KeyError as e:
        sys.exit(f"❌ {e.args[0]}")


def cmd_collect_day(args, cfg: dict) -> None:
    """開催1日分（全レース）の出馬表＋結果をまとめて収集する."""
    scraper = _make_scraper(cfg)
    meeting_id = _resolve_meeting_id(args, scraper, cfg)
    max_races = args.races or 12
    print(f"[collect-day] 開催 {meeting_id} の最大{max_races}レースを収集"
          f"（60秒間隔のため約{max_races*2}分）", file=sys.stderr)

    total_added = 0
    for n in range(1, max_races + 1):
        rid = race_id_for(meeting_id, n)
        r = _collect_one(scraper, rid, cfg)
        if not r["exists"]:
            print(f"  {n:2}R: レース無し（打ち切り）", file=sys.stderr)
            break
        status = f"出走{r['horses']} 着順{r['result']} 追記{r['added']}"
        if not r["result"]:
            status += " ⚠️未確定/結果なし"
        print(f"  {n:2}R: {status}", file=sys.stderr)
        total_added += r["added"]

    print(f"[collect-day] 完了。追記{total_added}行。現状: {dataset_stats(ROOT / 'data/dataset.jsonl')}",
          file=sys.stderr)


def _trained_race_count() -> int:
    """現モデルが何レースで学習されたか（メタから）。未学習は0."""
    meta = ROOT / "models" / "win_model.meta.json"
    if not meta.exists():
        return 0
    try:
        return int(json.loads(meta.read_text(encoding="utf-8")).get("races", 0))
    except (ValueError, OSError):
        return 0


def _maybe_retrain(*, grow: int = 10, quiet: bool = False) -> None:
    """データが (前回学習+grow) レース以上に増えていれば再学習する.

    予想時など、ML依存が確実に読めるインタラクティブ環境で呼ぶ前提。
    失敗してもpredictは止めない（モデル無しで続行）。
    """
    races = dataset_stats(ROOT / "data/dataset.jsonl").get("races", 0)
    if races < 30 or races < _trained_race_count() + grow:
        return
    print(f"[model] データ{races}レースでモデルを自動更新中...", file=sys.stderr)
    try:
        res = train_model(str(ROOT / "data/dataset.jsonl"), min_races=30)
        if res.get("trained"):
            print(f"[model] 更新完了（{res['races']}レースで学習）", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 ML未導入などでも予想は続行
        if not quiet:
            print(f"[model] 学習スキップ（{type(e).__name__}）。モデル無しで続行します。", file=sys.stderr)


def _log_auto(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, file=sys.stderr)
    log = ROOT / "data" / "auto.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def cmd_auto(args, cfg: dict) -> None:
    """本日の全開催を自動収集し、データが増えたら再学習する（無人実行向け）."""
    scraper = _make_scraper(cfg)
    # 夜1回の実行。必ず「本日」の開催一覧を最新取得する（キャッシュ不使用）。
    landing = scraper.get(scraper.race_card_landing_url(), max_age_sec=0)
    meetings = parse_meetings(landing)
    if not meetings:
        _log_auto("本日の開催が見つかりませんでした。")
        return
    _log_auto(f"自動収集開始。本日の開催: {list(meetings)}")

    grew = 0
    for track, mid in meetings.items():
        for n in range(1, (args.races or 12) + 1):
            rid = race_id_for(mid, n)
            r = _collect_one(scraper, rid, cfg)
            if not r["exists"]:
                break
            grew += r["added"]
        _log_auto(f"  {track}: 収集完了")

    stats = dataset_stats(ROOT / "data/dataset.jsonl")
    _log_auto(f"収集 {grew}行追記。データセット: {stats}")

    # 学習はタスク環境だとML依存が解決できないことがあるため、ここでは行わない。
    # 次回 predict 実行時（インタラクティブ環境）に自動で再学習される。
    if grew > 0 and not args.no_train:
        try:
            res = train_model(str(ROOT / "data/dataset.jsonl"), min_races=args.min_races or 30)
            _log_auto(f"再学習: {'完了 ' + str(res) if res.get('trained') else res.get('reason')}")
        except Exception as e:  # noqa: BLE001
            _log_auto(f"再学習は次回predict時に自動実行されます（タスク環境では{type(e).__name__}）。")
    _log_auto("自動実行を終了しました。")


def cmd_train(args, cfg: dict) -> None:  # noqa: ARG001
    """データセットから勝率モデルを学習する."""
    print("[train] 学習中...", file=sys.stderr)
    res = train_model(str(ROOT / "data/dataset.jsonl"), min_races=args.min_races or 30)
    if res.get("trained"):
        print(f"✅ 学習完了: {res}", file=sys.stderr)
    else:
        print(f"⏳ 未学習: {res.get('reason')}", file=sys.stderr)


def cmd_fetch_odds(args, cfg: dict) -> None:
    """EV計算の較正用に、各券種のオッズHTMLを取得・保存する."""
    scraper = _make_scraper(cfg)
    race_id = _resolve_race_id(args, scraper, cfg)
    kinds = (args.types or "tanfuku,umafuku,wide,sanrenfuku").split(",")
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        kind = kind.strip()
        url = scraper.odds_url(race_id, kind)
        print(f"取得[{kind}]: {url}", file=sys.stderr)
        html = scraper.get(url, max_age_sec=cfg["scraper"]["cache_ttl_hours"] * 3600)
        out = raw_dir / f"{race_id}_{kind}.html"
        out.write_text(html, encoding="utf-8")
        print(f"  保存: {out} ({len(html):,} bytes)", file=sys.stderr)
    print("→ これらのHTMLから odds.py のパーサを較正します。", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    # Windowsコンソール/リダイレクト時の絵文字・日本語の文字化け/例外を防ぐ
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(prog="keiba-ai", description="楽天競馬 予想AIエージェント")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("predict", "scan", "calibrate", "fetch-odds", "collect", "collect-day", "train", "auto"):
        sp = sub.add_parser(name)
        if name not in ("train", "auto"):
            sp.add_argument("--race-id", help="18桁のRACEID")
            sp.add_argument("--date", help="開催日 YYYY-MM-DD")
            sp.add_argument("--track", help="競馬場名 (例: 大井)")
            sp.add_argument("--race", type=int, help="レース番号 1-12")
        if name == "scan":
            sp.add_argument("--budget", type=int, help="軍資金（円）。指定すると各レースの金額も算出")
            sp.add_argument("--races", type=int, help="各開催で分析する最大レース数（既定12）")
            sp.add_argument("--fresh", action="store_true", help="オッズを最新取得（締め切り前）")
        if name == "collect-day":
            sp.add_argument("--races", type=int, help="収集する最大レース数（既定12）")
        if name == "auto":
            sp.add_argument("--races", type=int, help="各開催で収集する最大レース数（既定12）")
            sp.add_argument("--no-train", action="store_true", help="収集のみ（再学習しない）")
        if name in ("train", "auto"):
            sp.add_argument("--min-races", type=int, help="学習に必要な最小レース数（既定30）")
        if name == "predict":
            sp.add_argument("--budget", type=int, help="軍資金（円）。指定すると買い目プランを生成")
            sp.add_argument(
                "--style",
                choices=["conservative", "balanced", "aggressive"],
                help="賭け方のスタイル（既定は config.yaml の値）",
            )
            sp.add_argument(
                "--ev",
                action="store_true",
                help="EV/ケリーモード。組み合わせオッズを取得し期待値で配分（取得に時間がかかる）",
            )
            sp.add_argument(
                "--fresh",
                action="store_true",
                help="オッズのキャッシュを無視して必ず最新を取得（発走直前の判断用）",
            )
            sp.add_argument(
                "--gemini",
                action="store_true",
                help="Geminiチャット貼り付け用の予想依頼文を出力（Claude APIを使わない・無料）",
            )
            sp.add_argument(
                "--gem",
                action="store_true",
                help="Gem（カスタムGemini）用。指示文を省いたデータ中心の貼り付け文を出力",
            )
        if name == "fetch-odds":
            sp.add_argument("--types", help="取得する券種 カンマ区切り（既定: tanfuku,umafuku,wide,sanrenfuku）")

    args = p.parse_args(argv)
    cfg = load_config()
    dispatch = {
        "predict": cmd_predict,
        "scan": cmd_scan,
        "calibrate": cmd_calibrate,
        "fetch-odds": cmd_fetch_odds,
        "collect": cmd_collect,
        "collect-day": cmd_collect_day,
        "train": cmd_train,
        "auto": cmd_auto,
    }
    dispatch[args.cmd](args, cfg)


if __name__ == "__main__":
    main()

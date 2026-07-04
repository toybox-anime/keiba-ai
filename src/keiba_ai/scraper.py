"""礼儀正しいスクレイパー.

設計方針:
- robots.txt の Crawl-Delay: 60 を尊重し、リクエスト間隔を強制する。
- 取得した HTML はディスクにキャッシュし、同じページを何度も叩かない。
  （日次予想では同じ出馬表を複数回参照するため、キャッシュ効果が大きい）
- 失敗時はバックオフ付きでリトライ。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests


class PoliteScraper:
    def __init__(
        self,
        base_url: str,
        cache_dir: str | Path,
        crawl_delay_sec: float = 60.0,
        timeout_sec: float = 20.0,
        max_retries: int = 3,
        user_agent: str = "keiba-ai-research/0.1",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.crawl_delay_sec = crawl_delay_sec
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self._last_request_ts: float = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    # --- パス系 ---------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{key}.html"

    def race_card_landing_url(self) -> str:
        """本日開催一覧（本日の発売情報テーブル）を含むランディング."""
        return f"{self.base_url}/race_card/list/"

    def race_card_url(self, race_id: str) -> str:
        return f"{self.base_url}/race_card/list/RACEID/{race_id}"

    # 券種 → オッズページのパス
    ODDS_KINDS = {
        "tanfuku": "単複",
        "umafuku": "馬連",
        "umatan": "馬単",
        "wide": "ワイド",
        "sanrenfuku": "三連複",
        "sanrentan": "三連単",
    }

    def odds_url(self, race_id: str, kind: str = "tanfuku") -> str:
        if kind not in self.ODDS_KINDS:
            raise ValueError(f"未知のオッズ種別: {kind} (有効: {list(self.ODDS_KINDS)})")
        return f"{self.base_url}/odds/{kind}/RACEID/{race_id}"

    def result_url(self, race_id: str) -> str:
        return f"{self.base_url}/race_performance/list/RACEID/{race_id}"

    # --- 取得本体 -------------------------------------------------------
    def get(self, url: str, *, use_cache: bool = True, max_age_sec: float | None = None) -> str:
        """URL を取得して HTML 文字列を返す。キャッシュ優先。"""
        cache_file = self._cache_path(url)
        if use_cache and cache_file.exists():
            if max_age_sec is None or (time.time() - cache_file.stat().st_mtime) < max_age_sec:
                return cache_file.read_text(encoding="utf-8")

        self._respect_delay()
        html = self._fetch_with_retry(url)
        cache_file.write_text(html, encoding="utf-8")
        return html

    def _respect_delay(self) -> None:
        elapsed = time.time() - self._last_request_ts
        wait = self.crawl_delay_sec - elapsed
        if wait > 0:
            time.sleep(wait)

    def _fetch_with_retry(self, url: str) -> str:
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout_sec)
                self._last_request_ts = time.time()
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding or "utf-8"
                return resp.text
            except requests.RequestException as e:  # noqa: PERF203
                last_err = e
                self._last_request_ts = time.time()
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"取得失敗: {url} ({last_err})")

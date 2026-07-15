"""Gemini API（無料枠）を呼ぶ最小クライアント.

REST で呼ぶ（requests のみ・追加SDK不要）。APIキーは環境変数 GEMINI_API_KEY。
指定モデルが使えない場合は、利用可能なモデルを自動検出してフォールバックする。
"""

from __future__ import annotations

import os
import time

import requests

DEFAULT_MODEL = "gemini-flash-latest"
_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _key(api_key: str | None) -> str:
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が未設定です（GitHub Secret / 環境変数）。")
    return api_key


def available_models(api_key: str | None = None, *, timeout: float = 30.0) -> list[str]:
    """このキーで generateContent が使えるモデル名の一覧."""
    resp = requests.get(f"{_BASE}/models?key={_key(api_key)}", timeout=timeout)
    resp.raise_for_status()
    out = []
    for m in resp.json().get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            out.append(m["name"].split("/")[-1])
    return out


# 新規キーで使えなくなった非推奨モデル（ListModelsには出るが呼ぶと404）
_DEPRECATED = {"gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"}


def _pick_fallback(models: list[str]) -> str | None:
    """使えるモデルを選ぶ。latestエイリアス→flash-lite→その他flashの順。非推奨は避ける."""
    flash = [m for m in models if "flash" in m and "thinking" not in m and m not in _DEPRECATED]

    def rank(m: str) -> int:
        if "latest" in m:
            return 0
        if "lite" in m:
            return 1
        return 2

    flash.sort(key=rank)
    return (flash or [m for m in models if m not in _DEPRECATED] or models or [None])[0]


def generate(
    prompt: str, *, api_key: str | None = None, model: str = DEFAULT_MODEL,
    timeout: float = 120.0, max_tokens: int | None = 2500,
) -> str:
    """プロンプトを Gemini に投げて本文テキストを返す（モデル自動フォールバック付き）.

    max_tokens で出力トークンを制限（冗長さを抑えてコスト削減）。
    """
    api_key = _key(api_key)
    body: dict = {"contents": [{"parts": [{"text": prompt}]}]}
    if max_tokens:
        body["generationConfig"] = {"maxOutputTokens": max_tokens}

    def _call(m: str):
        return requests.post(f"{_BASE}/models/{m}:generateContent?key={api_key}", json=body, timeout=timeout)

    resp = _call(model)
    if resp.status_code == 404:  # モデルが使えない → 使えるモデルを探して再試行
        alt = _pick_fallback(available_models(api_key))
        if alt and alt != model:
            model, resp = alt, _call(alt)

    # 429（分あたり制限）は少し待ってリトライ。ただし日次上限(PerDay)は回復しないので即中断。
    for attempt in range(1, 3):
        if resp.status_code not in (429, 503) or "PerDay" in resp.text:
            break
        time.sleep(min(15 * attempt, 30))
        resp = _call(model)

    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API エラー {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Gemini応答の解析に失敗: {e} / {str(data)[:300]}") from e

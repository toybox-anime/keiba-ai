"""Gemini API（無料枠）を呼ぶ最小クライアント.

REST で呼ぶ（requests のみ・追加SDK不要）。APIキーは環境変数 GEMINI_API_KEY。
"""

from __future__ import annotations

import os

import requests

DEFAULT_MODEL = "gemini-2.5-flash"
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def generate(prompt: str, *, api_key: str | None = None, model: str = DEFAULT_MODEL, timeout: float = 120.0) -> str:
    """プロンプトを Gemini に投げて本文テキストを返す."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が未設定です（GitHub Secret / 環境変数）。")
    url = _ENDPOINT.format(model=model) + f"?key={api_key}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API エラー {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Gemini応答の解析に失敗: {e} / {str(data)[:300]}") from e

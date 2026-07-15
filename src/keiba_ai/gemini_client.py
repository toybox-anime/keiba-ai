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
    timeout: float = 180.0, max_tokens: int | None = 2000,
) -> str:
    """プロンプトを Gemini に投げて本文テキストを返す.

    - thinking(思考)を無効化して高速化（対応しないモデルは自動で外して再試行）。
    - モデル404は使えるモデルへ自動フォールバック。
    - タイムアウト/429/503 はリトライ（日次上限PerDayは即中断）。
    """
    api_key = _key(api_key)

    def _body(with_think: bool) -> dict:
        gc: dict = {}
        if max_tokens:
            gc["maxOutputTokens"] = max_tokens
        if with_think:
            gc["thinkingConfig"] = {"thinkingBudget": 0}  # 思考オフ＝速い・安い
        b: dict = {"contents": [{"parts": [{"text": prompt}]}]}
        if gc:
            b["generationConfig"] = gc
        return b

    def _call(m: str, with_think: bool = True):
        try:
            return requests.post(
                f"{_BASE}/models/{m}:generateContent?key={api_key}", json=_body(with_think), timeout=timeout
            )
        except requests.exceptions.RequestException:
            return None  # タイムアウト等 → リトライ対象

    def _try(m: str):
        r = _call(m, True)
        if r is not None and r.status_code == 400 and "think" in r.text.lower():
            r = _call(m, False)  # thinkingConfig非対応モデル
        return r

    resp = _try(model)
    if resp is not None and resp.status_code == 404:  # モデル不在 → 使えるモデルへ
        alt = _pick_fallback(available_models(api_key))
        if alt and alt != model:
            model, resp = alt, _try(alt)

    for attempt in range(1, 4):  # タイムアウト/429/503 リトライ
        retryable = resp is None or (resp.status_code in (429, 503) and "PerDay" not in resp.text)
        if not retryable:
            break
        time.sleep(min(10 * attempt, 30))
        resp = _try(model)

    if resp is None:
        raise RuntimeError("Gemini API 応答なし（タイムアウト）")
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API エラー {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Gemini応答の解析に失敗: {e} / {str(data)[:300]}") from e

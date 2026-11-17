"""Thin client for a hosted LLM (Gemini or Groq) that returns parsed JSON."""
import json
import re
import time

import httpx

from . import config


class LLMError(RuntimeError):
    """Raised when the LLM is unavailable or returns something unusable.

    Messages never include response bodies or URLs, because those can contain
    resume text or API keys.
    """


def llm_available() -> bool:
    key = config.GEMINI_API_KEY if config.LLM_PROVIDER == "gemini" else config.GROQ_API_KEY
    return bool(key)


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _gemini(prompt: str, system: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.LLM_MODEL}:generateContent"
    response = httpx.post(
        url,
        headers={"x-goog-api-key": config.GEMINI_API_KEY},  # header, not query string, so it never lands in logs
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        },
        timeout=config.LLM_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def _groq(prompt: str, system: str) -> str:
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
        json={
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=config.LLM_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_llm_json(prompt: str, system: str = "You are a precise assistant. Reply with valid JSON only.") -> dict:
    if not llm_available():
        raise LLMError("No LLM API key configured")
    if config.LLM_PROVIDER not in ("gemini", "groq"):
        raise LLMError(f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'")
    call = _gemini if config.LLM_PROVIDER == "gemini" else _groq

    last_error = "unknown error"
    for attempt in range(2):  # one retry for rate limits / transient server errors
        try:
            return _parse_json(call(prompt, system))
        except httpx.HTTPStatusError as exc:
            last_error = f"HTTP {exc.response.status_code}"
            if exc.response.status_code not in (429, 500, 502, 503, 504):
                break
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = type(exc).__name__
        time.sleep(1.5 * (attempt + 1))
    raise LLMError(f"LLM call failed ({last_error})")

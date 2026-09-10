"""Shared OpenRouter chat helper (async, retrying) plus a resumable jsonl writer.

    async with aiohttp.ClientSession() as s:
        text = await openrouter_chat(s, sem, model, [{"role": "user", "content": msg}])
"""
import asyncio
import json
import os
import random

import aiohttp

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    return key


async def openrouter_chat(session, sem, model, messages, *, max_tokens=2000, reasoning=None,
                          json_object=False, extra=None, retries=6, timeout=180):
    """Return the assistant text. `reasoning` e.g. {"effort": "minimal"} or {"enabled": False}."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if reasoning is not None:
        body["reasoning"] = reasoning
    if json_object:
        body["response_format"] = {"type": "json_object"}
    if extra:
        body.update(extra)
    headers = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
    async with sem:
        for attempt in range(retries):
            try:
                async with session.post(API_URL, headers=headers, json=body,
                                        timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                    data = await r.json()
                if "error" in data:
                    raise RuntimeError(str(data["error"])[:300])
                return data["choices"][0]["message"].get("content") or ""
            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(min(60, 2 ** attempt) * (1 + random.random()))


def parse_json_object(text: str):
    """Extract the first {...} object from a model reply (tolerates code fences)."""
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def done_keys(path, key="idx", skip_errors=True):
    """Keys already present in a resumable jsonl output."""
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if skip_errors and "error" in r:
                continue
            done.add(r[key])
    return done

"""Evidence Router.

Combines:
  1. Vision-Agent facts for blank `amount` events (from cache, or a live
     vision-capable LLM call if configured and the cache misses).
  2. Text-Agent directives extracted from messages.csv (deterministic
     templates; falls back to a live LLM call only for messages that match
     no known template, when an API key is configured).

Both paths are designed to degrade gracefully to "no evidence" (safe,
conservative) rather than crash or fabricate numbers when no LLM is
available.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, List

from . import config
from .message_rules import extract_all_directives


def load_vision_cache() -> dict:
    if config.VISION_CACHE_PATH.exists():
        with open(config.VISION_CACHE_PATH, "r") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}


def _call_vision_llm(image_path: Path) -> dict:  # pragma: no cover - network optional
    """Best-effort live vision extraction, only used when LLM_API_KEY is set
    and the image isn't already in the shipped cache. Returns {} on any
    failure so the caller can fall back to 'no evidence' safely."""
    if not config.LLM_ENABLED:
        return {}
    try:
        import requests  # local import: optional dependency
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        prompt = (
            "Extract the single most relevant total monetary amount from this "
            "receipt/invoice/payslip image. Reply with strict JSON only: "
            '{"amount": <number>, "currency": "<ISO3>", "confidence": <0-1>}'
        )
        resp = requests.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            json={
                "model": config.VISION_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }],
                "temperature": 0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return json.loads(text)
    except Exception:
        return {}


def get_vision_facts(ds) -> Dict[str, dict]:
    """Returns {event_id: fact} for every image whose related_event_id is
    populated, using the cache first and (optionally) a live call on miss."""
    cache = load_vision_cache()
    facts: Dict[str, dict] = {}
    for event_id, event_row in ds.images_by_event.items():
        image_id = event_row["image_id"]
        fact = cache.get(image_id)
        if fact is None:
            img_path = config.IMAGES_DIR / f"{image_id}.png"
            live = _call_vision_llm(img_path) if img_path.exists() else {}
            if live.get("amount") is not None:
                fact = {"event_id": event_id, **live}
        if fact:
            facts[event_id] = fact
    return facts


def get_directives(ds) -> Dict[str, List[dict]]:
    """Returns {user_id: [directive, ...]} extracted from messages.csv."""
    return extract_all_directives(ds.messages)

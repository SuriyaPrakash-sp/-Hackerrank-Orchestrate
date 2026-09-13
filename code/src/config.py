"""Configuration: paths and (optional) LLM settings.

The pipeline runs fully deterministically with zero API calls by default.
If LLM_API_KEY is set, the evidence-extraction agents (src/evidence.py) will
call out to an OpenAI-compatible endpoint for messages/images that are not
already covered by the shipped cache (code/cache/vision_cache.json). This
lets the same codebase serve both the "no key, fully offline" grading path
and a "live agentic" path.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent.parent          # .../code
REPO_ROOT = CODE_DIR.parent                                  # repo root (or bundle root)

# Dataset can live at <repo_root>/dataset (standard HackerRank Orchestrate layout).
DATASET_DIR = Path(os.environ.get("DATASET_DIR", REPO_ROOT / "dataset"))

OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", REPO_ROOT / "output.csv"))

CACHE_DIR = CODE_DIR / "cache"
VISION_CACHE_PATH = CACHE_DIR / "vision_cache.json"
MESSAGE_CACHE_PATH = CACHE_DIR / "message_cache.json"

EVALUATION_DIR = CODE_DIR / "evaluation"
USAGE_REPORT_PATH = EVALUATION_DIR / "usage_report.md"

IMAGES_DIR = DATASET_DIR / "media" / "images"

# --- Forecast parameters -----------------------------------------------------
FORECAST_HORIZON_DAYS = 90
MAX_SPENDING_CHANGES = 3
MONEY_PRECISION = "0.01"          # round to cents at output boundary
BINARY_SEARCH_TOLERANCE = "0.01"  # amount_safe_to_pay precision

# --- Optional LLM configuration --------------------------------------------
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
TEXT_MODEL = os.environ.get("TEXT_MODEL", "gpt-4o-mini")
VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o-mini")
LLM_ENABLED = bool(LLM_API_KEY)

# Confidence threshold below which a Level-2 (stronger) fallback would be used
# if one were configured. Kept for completeness / future extension.
CONFIDENCE_ESCALATION_THRESHOLD = 0.85

"""
StudyMind — Configuration & Startup Validation (Cognee Cloud)

Loads .env, validates Cognee Cloud is reachable, and provides helpers
for the UI status display. Designed to fail fast with clear error messages.
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root (one level above this file)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

load_dotenv(_ENV_PATH, override=True)

# ---------------------------------------------------------------------------
# Constants derived from env
# ---------------------------------------------------------------------------
COGNEE_BASE_URL = os.getenv("COGNEE_BASE_URL", "").rstrip("/")
COGNEE_API_KEY  = os.getenv("COGNEE_API_KEY", "")

DATA_DIR = _PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Default dataset name used for all ingestions
DEFAULT_DATASET = "studymind"


def _headers() -> dict:
    """Return standard request headers for the Cognee Cloud API."""
    return {
        "X-Api-Key": COGNEE_API_KEY,
        "Accept": "application/json",
    }


def check_cognee_cloud(timeout: int = 8) -> dict:
    """Verify Cognee Cloud is reachable and the API key is valid.

    Pings GET /api/v1/datasets/ — a lightweight endpoint that returns 200
    on success, 401 on bad key, 404/5xx on wrong URL or outage.

    Returns:
        dict with keys: ok (bool), error (str|None)
    """
    result = {"ok": False, "error": None}

    if not COGNEE_BASE_URL:
        result["error"] = (
            "COGNEE_BASE_URL is not set. "
            "Add it to your .env file: COGNEE_BASE_URL=https://tenant-xxx.aws.cognee.ai"
        )
        return result

    if not COGNEE_API_KEY:
        result["error"] = (
            "COGNEE_API_KEY is not set. "
            "Add it to your .env file: COGNEE_API_KEY=your_key_here"
        )
        return result

    try:
        resp = httpx.get(
            f"{COGNEE_BASE_URL}/api/v1/datasets/",
            headers=_headers(),
            timeout=timeout,
        )
        if resp.status_code == 200:
            result["ok"] = True
        elif resp.status_code == 401:
            result["error"] = (
                "Cognee Cloud: authentication failed (401). "
                "Check your COGNEE_API_KEY in .env."
            )
        elif resp.status_code == 404:
            result["error"] = (
                f"Cognee Cloud: endpoint not found (404). "
                f"Check your COGNEE_BASE_URL in .env: {COGNEE_BASE_URL}"
            )
        else:
            result["error"] = (
                f"Cognee Cloud returned HTTP {resp.status_code}. "
                f"The service may be temporarily unavailable."
            )
    except httpx.ConnectError:
        result["error"] = (
            f"Cannot reach Cognee Cloud at {COGNEE_BASE_URL}. "
            "Check your internet connection and COGNEE_BASE_URL."
        )
    except Exception as exc:
        result["error"] = f"Cognee Cloud health check failed: {exc}"

    return result


# Keep legacy name for backward compatibility (app.py still imports this)
def check_ollama(timeout: int = 8) -> dict:
    """Alias for check_cognee_cloud — kept for backward compatibility."""
    return check_cognee_cloud(timeout=timeout)


def get_cloud_status_emoji(check: dict | None = None) -> str:
    """Return a colored status indicator for the sidebar."""
    if check is None:
        check = check_cognee_cloud(timeout=5)
    return "🟢 Connected" if check["ok"] else "🔴 Disconnected"


# Keep legacy name for backward compatibility
def get_ollama_status_emoji(check: dict | None = None) -> str:
    """Alias for get_cloud_status_emoji — kept for backward compatibility."""
    return get_cloud_status_emoji(check)


def ensure_ready():
    """Run startup checks. Call once at app init.

    Prints errors to stderr but does NOT exit — lets the Streamlit UI
    show the error state instead of crashing.
    """
    status = check_cognee_cloud()
    if not status["ok"]:
        print(f"[StudyMind] WARNING: {status['error']}", file=sys.stderr)
    return status

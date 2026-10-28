"""
StudyMind — Configuration & Startup Validation

Settings come from .env via pydantic-settings. Also holds the shared async
HTTP client and the health checks for Cognee Cloud and the LLM.
"""

from pathlib import Path

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = _PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Default dataset name used for all ingestions
DEFAULT_DATASET = "studymind"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_PROJECT_ROOT / ".env", extra="ignore")

    cognee_base_url: str = ""
    cognee_api_key: str = ""

    # Any LiteLLM model string: ollama/gemma3:4b, groq/..., anthropic/...
    llm_model: str = "ollama/qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"

    database_url: str = f"sqlite:///{(DATA_DIR / 'studymind.db').as_posix()}"

    # Tracing is enabled only when both keys are set
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Retrieval
    chunker: str = "structure"  # structure | fixed
    embed_model: str = "bge-small"  # bge-small | nomic
    retriever: str = "hybrid"  # hybrid | dense
    qdrant_path: str = (DATA_DIR / "qdrant").as_posix()  # ":memory:" in tests

    # Where the Streamlit UI finds the API
    api_url: str = "http://localhost:8000"


settings = Settings()
settings.cognee_base_url = settings.cognee_base_url.rstrip("/")


def _headers() -> dict:
    """Return standard request headers for the Cognee Cloud API."""
    return {
        "X-Api-Key": settings.cognee_api_key,
        "Accept": "application/json",
    }


_client: httpx.AsyncClient | None = None


def http_client() -> httpx.AsyncClient:
    """Shared async HTTP client (tests replace it with a MockTransport one)."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient()
    return _client


def set_http_client(client: httpx.AsyncClient | None) -> None:
    global _client
    _client = client


async def check_cognee_cloud(timeout: int = 8) -> dict:
    """Verify Cognee Cloud is reachable and the API key is valid.

    Pings GET /api/v1/datasets/ — a lightweight endpoint that returns 200
    on success, 401 on bad key, 404/5xx on wrong URL or outage.

    Returns:
        dict with keys: ok (bool), error (str|None)
    """
    result = {"ok": False, "error": None}

    if not settings.cognee_base_url:
        result["error"] = (
            "COGNEE_BASE_URL is not set. "
            "Add it to your .env file: COGNEE_BASE_URL=https://tenant-xxx.aws.cognee.ai"
        )
        return result

    if not settings.cognee_api_key:
        result["error"] = (
            "COGNEE_API_KEY is not set. "
            "Add it to your .env file: COGNEE_API_KEY=your_key_here"
        )
        return result

    try:
        resp = await http_client().get(
            f"{settings.cognee_base_url}/api/v1/datasets/",
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
                f"Check your COGNEE_BASE_URL in .env: {settings.cognee_base_url}"
            )
        else:
            result["error"] = (
                f"Cognee Cloud returned HTTP {resp.status_code}. "
                f"The service may be temporarily unavailable."
            )
    except httpx.ConnectError:
        result["error"] = (
            f"Cannot reach Cognee Cloud at {settings.cognee_base_url}. "
            "Check your internet connection and COGNEE_BASE_URL."
        )
    except Exception as exc:
        result["error"] = f"Cognee Cloud health check failed: {exc}"

    return result


async def check_llm(timeout: int = 5) -> dict:
    """Verify the configured LLM is usable.

    For ollama/* models: Ollama must be running and the model pulled.
    Hosted models are not pinged (that would cost a request).
    """
    model = settings.llm_model
    result = {"ok": True, "error": None, "model": model}
    if not model.startswith("ollama/"):
        return result

    name = model.split("/", 1)[1]
    try:
        resp = await http_client().get(f"{settings.ollama_base_url}/api/tags", timeout=timeout)
        pulled = {m["name"] for m in resp.json().get("models", [])}
        # Ollama lists untagged models as "<name>:latest"
        if name not in pulled and f"{name}:latest" not in pulled:
            result.update(ok=False, error=f"Model not pulled. Run: ollama pull {name}")
    except Exception as exc:
        result.update(ok=False, error=f"Cannot reach Ollama at {settings.ollama_base_url}: {exc}")
    return result


def get_cloud_status_emoji(check: dict) -> str:
    """Return a colored status indicator for the sidebar."""
    return "🟢 Connected" if check["ok"] else "🔴 Disconnected"

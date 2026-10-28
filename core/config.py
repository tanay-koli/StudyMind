"""
StudyMind — Configuration & Startup Validation

Settings come from .env via pydantic-settings. Also holds the shared async
HTTP client and the LLM health check.
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
    query_rewrite: str = "none"  # none | multi | hyde
    qdrant_path: str = (DATA_DIR / "qdrant").as_posix()  # ":memory:" in tests

    # Where the Streamlit UI finds the API
    api_url: str = "http://localhost:8000"


settings = Settings()

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

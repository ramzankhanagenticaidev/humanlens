"""
Central configuration for HumanLens.

All tunables live here so nothing is hard-coded deep inside the app.
Values can be overridden with environment variables (see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:  # pragma: no cover - dotenv is a listed dependency but optional at runtime
    pass


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


@dataclass
class Settings:
    # --- Storage ---
    db_path: str = os.getenv("HUMANLENS_DB_PATH", str(BASE_DIR / "app" / "humanlens.db"))
    model_dir: str = os.getenv("HUMANLENS_MODEL_DIR", str(BASE_DIR / "app" / "models"))

    # --- Analysis limits ---
    min_words_for_analysis: int = _env_int("HUMANLENS_MIN_WORDS", 40)
    max_chars_for_analysis: int = _env_int("HUMANLENS_MAX_CHARS", 50_000)

    # --- Detector behaviour ---
    # Sensitivity shifts the decision threshold. 0.5 = neutral.
    # Lower threshold => detector flags more text as AI-like (more sensitive).
    detector_sensitivity: float = _env_float("HUMANLENS_SENSITIVITY", 0.5)

    # --- Privacy ---
    store_full_text: bool = _env_bool("HUMANLENS_STORE_FULL_TEXT", True)

    # --- Optional local LLM (Ollama) ---
    ollama_enabled: bool = _env_bool("HUMANLENS_OLLAMA_ENABLED", False)
    ollama_host: str = os.getenv("HUMANLENS_OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("HUMANLENS_OLLAMA_MODEL", "llama3.1")
    ollama_timeout_s: int = _env_int("HUMANLENS_OLLAMA_TIMEOUT", 30)

    # --- API ---
    api_host: str = os.getenv("HUMANLENS_API_HOST", "127.0.0.1")
    api_port: int = _env_int("HUMANLENS_API_PORT", 8000)

    # --- Production / security ---
    # If set, every request must include header "X-API-Key: <value>" (except
    # /health). Leave unset for local single-user development.
    api_key: str = os.getenv("HUMANLENS_API_KEY", "")

    # Comma-separated list of origins allowed to call the API from a browser,
    # e.g. "https://app.example.com,https://www.example.com".
    # Defaults to localhost-only, which is safe for the bundled Streamlit
    # frontend running on the same machine.
    allowed_origins: list = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv(
                "HUMANLENS_ALLOWED_ORIGINS",
                "http://localhost:8501,http://127.0.0.1:8501",
            ).split(",")
            if o.strip()
        ]
    )

    # Requests allowed per client (per API key, or per IP if no key is set)
    # in a rolling 60-second window. Set to 0 to disable rate limiting
    # (not recommended once the API is reachable beyond localhost).
    rate_limit_per_minute: int = _env_int("HUMANLENS_RATE_LIMIT_PER_MINUTE", 60)

    # Set to "production" to disable the interactive /docs and /redoc pages.
    environment: str = os.getenv("HUMANLENS_ENV", "development")

    def ensure_dirs(self) -> None:
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

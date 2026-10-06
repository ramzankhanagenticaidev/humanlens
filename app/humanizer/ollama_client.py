"""
Optional local LLM integration via Ollama.

Ollama runs entirely locally (https://ollama.com) — no API key, no
cloud calls. This client is intentionally defensive: any failure
(Ollama not installed, not running, model not pulled, timeout) results
in a returned error message and the humanizer engine falls back to the
deterministic rule-based rewriter. The app must remain fully usable
without Ollama installed.

Note on `settings.ollama_enabled`: this only sets the *default* state
of the "use local LLM" checkbox in the UI. Whether a given request
actually attempts Ollama is decided by the caller (the checkbox /
`use_ollama` request field), NOT re-checked here — `is_available()`
only asks "is an Ollama server actually reachable right now", so that
explicitly requesting Ollama for one request always attempts it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import requests

from app.config import settings

REWRITE_SYSTEM_PROMPT = (
    "You are a careful writing editor. Rewrite the user's text to sound more "
    "natural, clear, and human, while strictly preserving its meaning, facts, "
    "numbers, dates, names, and intent. Do not add new information. Do not "
    "remove information. Keep roughly the same length. Return only the "
    "rewritten text, with no preamble or explanation."
)


@dataclass
class OllamaResult:
    output: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.output is not None


class OllamaClient:
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None, timeout: Optional[int] = None):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout_s

    def is_available(self) -> bool:
        """True iff an Ollama server responds at self.host. Does NOT check
        settings.ollama_enabled — that's a UI default, not a hard gate."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Returns the names of models currently pulled locally, or an
        empty list if Ollama isn't reachable."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    def rewrite(self, text: str, model: Optional[str] = None) -> OllamaResult:
        """Attempts a rewrite via Ollama. Always returns an OllamaResult —
        never raises — with a specific, human-readable `error` on any
        failure so the UI can tell the user exactly what went wrong
        (host unreachable vs. model not found vs. timeout vs. empty output)
        instead of a generic "it didn't work"."""
        use_model = model or self.model

        if not self.is_available():
            return OllamaResult(
                error=f"Could not reach an Ollama server at {self.host}. "
                      f"Is Ollama running? Try `ollama serve` in a terminal."
            )

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": use_model,
                    "prompt": f"{REWRITE_SYSTEM_PROMPT}\n\nText:\n{text}",
                    "stream": False,
                },
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            return OllamaResult(error=f"Ollama request timed out after {self.timeout}s (model: {use_model}).")
        except Exception as exc:
            return OllamaResult(error=f"Ollama request failed: {exc}")

        if resp.status_code != 200:
            detail = resp.text
            try:
                detail = resp.json().get("error", detail)
            except Exception:
                pass
            available = self.list_models()
            hint = (
                f" Models currently pulled: {', '.join(available)}."
                if available
                else " No models appear to be pulled — try `ollama pull <model>`."
            )
            return OllamaResult(
                error=f"Ollama rejected the request for model '{use_model}': {detail}.{hint}"
            )

        try:
            data = resp.json()
        except Exception:
            return OllamaResult(error="Ollama returned a response that wasn't valid JSON.")

        output = (data.get("response") or "").strip()
        if not output:
            return OllamaResult(error=f"Ollama (model: {use_model}) returned an empty response.")

        return OllamaResult(output=output)


_client_singleton: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = OllamaClient()
    return _client_singleton

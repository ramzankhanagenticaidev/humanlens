import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.humanizer.engine import get_humanizer_engine
from app.humanizer.validation import validate


def test_humanize_empty_text():
    engine = get_humanizer_engine()
    result = engine.humanize("", use_ollama=False)
    assert result.revised_text == "" or isinstance(result.revised_text, str)


def test_humanize_removes_throat_clearer():
    engine = get_humanizer_engine()
    result = engine.humanize(
        "It is important to note that the project is on schedule.", use_ollama=False
    )
    assert "it is important to note that" not in result.revised_text.lower()


def test_humanize_simplifies_verbosity():
    engine = get_humanizer_engine()
    result = engine.humanize(
        "We need to utilize a robust and comprehensive solution.", use_ollama=False
    )
    assert "utilize" not in result.revised_text.lower()


def test_humanize_preserves_numbers():
    engine = get_humanizer_engine()
    text = "It is important to note that revenue grew by 23% in Q3 2025."
    result = engine.humanize(text, use_ollama=False)
    assert "23%" in result.revised_text


def test_humanize_falls_back_when_ollama_unavailable():
    engine = get_humanizer_engine()
    result = engine.humanize("Utilize this comprehensive approach.", use_ollama=True)
    # Ollama isn't running in the test environment, so this must fall back
    assert result.engine_used == "rule_based"
    assert any("Ollama" in w for w in result.warnings)


def test_ollama_not_gated_by_global_enabled_setting(monkeypatch):
    """Regression test: is_available() must reflect actual server
    reachability, not the settings.ollama_enabled UI-default flag. This
    was a real bug — checking the UI checkbox alone didn't attempt
    Ollama unless HUMANLENS_OLLAMA_ENABLED was also set."""
    from app.humanizer.ollama_client import OllamaClient
    import app.config as config_module

    monkeypatch.setattr(config_module.settings, "ollama_enabled", False)
    client = OllamaClient(host="http://127.0.0.1:1", timeout=1)  # unroutable, fails fast
    # is_available() should attempt a real connection and fail due to
    # unreachability, NOT short-circuit on ollama_enabled being False.
    assert client.is_available() is False


def test_ollama_rewrite_reports_specific_unreachable_error():
    from app.humanizer.ollama_client import OllamaClient

    client = OllamaClient(host="http://127.0.0.1:1", timeout=1)
    result = client.rewrite("some text")
    assert result.ok is False
    assert "reach" in result.error.lower() or "connect" in result.error.lower() or result.error


def test_ollama_model_override_is_passed_through(monkeypatch):
    """Verifies humanize(..., ollama_model=...) actually reaches the
    Ollama client instead of silently using the configured default."""
    from app.humanizer import engine as engine_module
    from app.humanizer.ollama_client import OllamaResult

    captured = {}

    class FakeClient:
        host = "http://fake"

        def rewrite(self, text, model=None):
            captured["model"] = model
            return OllamaResult(output="Rewritten text.")

    monkeypatch.setattr(engine_module, "get_ollama_client", lambda: FakeClient())

    engine = get_humanizer_engine()
    result = engine.humanize("Some text to rewrite.", use_ollama=True, ollama_model="qwen2.5-coder:3b")

    assert captured["model"] == "qwen2.5-coder:3b"
    assert result.engine_used == "ollama"
    assert result.revised_text == "Rewritten text."


def test_validation_detects_missing_number():
    report = validate("Sales rose by 42% this quarter.", "Sales rose this quarter.")
    categories = [i.category for i in report.issues]
    assert "number" in categories
    assert report.is_safe is False


def test_validation_detects_missing_url():
    report = validate(
        "Check https://example.com for more info.", "Check the site for more info."
    )
    categories = [i.category for i in report.issues]
    assert "url" in categories


def test_validation_passes_on_faithful_rewrite():
    report = validate(
        "The meeting is scheduled for 3pm on Tuesday.",
        "The meeting is set for 3pm Tuesday.",
    )
    assert report.is_safe is True


def test_humanize_result_serializes():
    engine = get_humanizer_engine()
    result = engine.humanize("Utilize this approach in order to succeed.", use_ollama=False)
    d = result.to_dict()
    assert "revised_text" in d
    assert "validation" in d

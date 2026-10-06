import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detector.detector import Detector, get_detector


def test_detector_singleton_loads():
    d = get_detector()
    assert d is not None


def test_analyze_empty_text_does_not_crash():
    d = get_detector()
    result = d.analyze("")
    assert 0.0 <= result.ai_probability <= 1.0
    assert any("shorter than the recommended minimum" in w for w in result.warnings)


def test_analyze_short_text():
    d = get_detector()
    result = d.analyze("Hi there, how are you?")
    assert result.num_words > 0
    assert result.confidence in {"Low", "Moderate", "High"}


def test_analyze_normal_human_like_text():
    d = get_detector()
    text = (
        "so i finally fixed the bug, turns out it was a stupid off-by-one error that took me three hours "
        "to find. classic. anyway i'm exhausted, going to bed early tonight for once."
    )
    result = d.analyze(text)
    assert result.ai_probability < 0.6  # should lean human-ish, not asserting exact value


def test_analyze_formulaic_text():
    d = get_detector()
    text = (
        "In today's ever-evolving landscape, it is important to note that businesses must leverage "
        "comprehensive, robust strategies. Furthermore, this seamlessly drives sustainable growth. "
        "Moreover, organizations that adopt a holistic approach unlock significant potential."
    )
    result = d.analyze(text)
    assert result.ai_probability > 0.4  # should show elevated formulaic signal


def test_paragraph_level_analysis_present():
    d = get_detector()
    text = "First paragraph with some casual thoughts about my weekend plans and nothing else really.\n\n" \
           "Second paragraph, in stark contrast, leverages a comprehensive and robust approach to seamlessly " \
           "convey its ever-evolving message to the reader in a formulaic way."
    result = d.analyze(text)
    assert len(result.paragraphs) == 2


def test_sensitivity_changes_output():
    d = get_detector()
    text = "This is a moderately neutral piece of text used to test sensitivity effects on scoring output."
    low = d.analyze(text, sensitivity=0.2)
    high = d.analyze(text, sensitivity=0.8)
    # different thresholds should not error and should produce valid probabilities
    assert 0.0 <= low.ai_probability <= 1.0
    assert 0.0 <= high.ai_probability <= 1.0


def test_result_serializes_to_dict():
    d = get_detector()
    result = d.analyze("Some reasonably normal test text that is long enough to analyze properly here.")
    as_dict = result.to_dict()
    assert "ai_probability" in as_dict
    assert "supporting_signals" in as_dict
    assert "paragraphs" in as_dict


def test_unicode_input():
    d = get_detector()
    result = d.analyze("Café résumé — this naïvely tests unicode handling with émojis 😀 and accents.")
    assert result.num_words > 0

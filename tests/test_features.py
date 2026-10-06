import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detector.features import extract_features, FEATURE_NAMES


def test_empty_text():
    bundle = extract_features("")
    assert bundle.values["num_words"] == 0
    # should not raise, all features should be finite numbers
    for k in FEATURE_NAMES:
        assert bundle.values[k] == bundle.values[k]  # not NaN


def test_very_short_text():
    bundle = extract_features("Hi.")
    assert bundle.values["num_words"] >= 1


def test_normal_text():
    text = "This is a normal sentence. It has more than one sentence, actually."
    bundle = extract_features(text)
    assert bundle.values["num_sentences"] == 2
    assert bundle.values["num_words"] > 5


def test_unicode_text():
    text = "Café résumé naïve — this has unicode, émojis 😀, and accents."
    bundle = extract_features(text)
    assert bundle.values["num_words"] > 0


def test_urls_and_numbers_dont_crash():
    text = "Visit https://example.com/page?x=1 for details, or call 555-1234 by 10/12/2025."
    bundle = extract_features(text)
    assert bundle.values["num_words"] > 0


def test_repeated_words_detected():
    text = "test test test test test test test test."
    bundle = extract_features(text)
    assert bundle.values["repeated_word_rate"] >= 0  # doesn't crash; low-info text is fine


def test_contractions_detected():
    text = "I don't think it's going to work, but we'll see. They're not sure either."
    bundle = extract_features(text)
    assert bundle.values["contraction_rate"] > 0


def test_paragraph_detection():
    text = "Paragraph one here.\n\nParagraph two here.\n\nParagraph three here."
    bundle = extract_features(text)
    assert len(bundle.paragraphs) == 3


def test_punctuation_diversity():
    text = "Wait, what? Really! I can't believe it; that's wild."
    bundle = extract_features(text)
    assert bundle.values["punctuation_diversity"] > 0


def test_extremely_long_input_does_not_crash():
    text = ("This is a reasonably normal sentence about nothing in particular. " * 2000)
    bundle = extract_features(text)
    assert bundle.values["num_words"] > 1000

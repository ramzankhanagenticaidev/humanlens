import csv
import random
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

import merge_datasets


# ---------------------------------------------------------------------------
# merge_datasets.py
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)


def test_merge_dedupes_exact_matches(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "out.csv"
    _write_csv(a, [{"text": "Hello world.", "label": "human"}, {"text": "Formulaic AI text.", "label": "ai"}])
    _write_csv(b, [{"text": "Hello world.", "label": "human"}, {"text": "Different text here.", "label": "ai"}])

    with patch("sys.argv", ["merge_datasets.py", str(a), str(b), "--out", str(out)]):
        merge_datasets.main()

    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # 1 duplicate removed from 4 total


def test_merge_reports_correct_balance(tmp_path):
    a = tmp_path / "a.csv"
    out = tmp_path / "out.csv"
    _write_csv(a, [
        {"text": "human one", "label": "human"},
        {"text": "human two", "label": "human"},
        {"text": "ai one", "label": "ai"},
    ])
    with patch("sys.argv", ["merge_datasets.py", str(a), "--out", str(out)]):
        merge_datasets.main()
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert sum(1 for r in rows if r["label"] == "human") == 2
    assert sum(1 for r in rows if r["label"] == "ai") == 1


def test_merge_skips_missing_file(tmp_path, capsys):
    out = tmp_path / "out.csv"
    real = tmp_path / "real.csv"
    _write_csv(real, [{"text": "some human text", "label": "human"}, {"text": "some ai text", "label": "ai"}])
    with patch("sys.argv", ["merge_datasets.py", str(tmp_path / "missing.csv"), str(real), "--out", str(out)]):
        merge_datasets.main()
    assert out.exists()
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_merge_exits_on_empty_class(tmp_path):
    a = tmp_path / "a.csv"
    out = tmp_path / "out.csv"
    _write_csv(a, [{"text": "only human text", "label": "human"}])
    with patch("sys.argv", ["merge_datasets.py", str(a), "--out", str(out)]):
        with pytest.raises(SystemExit):
            merge_datasets.main()


def test_merge_skips_rows_with_invalid_label(tmp_path):
    a = tmp_path / "a.csv"
    out = tmp_path / "out.csv"
    _write_csv(a, [
        {"text": "valid human", "label": "human"},
        {"text": "invalid label row", "label": "robot"},
        {"text": "valid ai", "label": "ai"},
    ])
    with patch("sys.argv", ["merge_datasets.py", str(a), "--out", str(out)]):
        merge_datasets.main()
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# download_raid.py — logic tested against a mock matching RAID's real schema
# (id, source_id, model, decoding, repetition_penalty, attack, domain, generation)
# ---------------------------------------------------------------------------
import download_raid  # noqa: E402


def _mock_raid_row(is_human, model, domain, attack="none", decoding="greedy", rep="no", n_words=60, rng=None):
    rng = rng or random.Random()
    text = " ".join(
        rng.choice(["the", "system", "reported", "clearly", "however", "results", "showed", "improvement"])
        for _ in range(n_words)
    )
    return {
        "id": "id1", "source_id": "id1" if is_human else "src1",
        "model": "human" if is_human else model,
        "decoding": decoding, "repetition_penalty": rep, "attack": attack,
        "domain": domain, "title": "t", "prompt": "p", "generation": text,
    }


class _FakeStreamingDataset:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def test_raid_filters_out_adversarial_by_default():
    rng = random.Random(1)
    rows = [_mock_raid_row(True, None, "news", n_words=60, rng=rng) for _ in range(30)]
    rows += [_mock_raid_row(False, "chatgpt", "news", attack="paraphrase", decoding="sampling", rep="yes", rng=rng) for _ in range(30)]
    with patch("datasets.load_dataset", return_value=_FakeStreamingDataset(rows)):
        human_rows, ai_rows = download_raid.build_sample(domains=["news"], per_class=10, include_adversarial=False)
    assert len(ai_rows) == 0  # all AI rows were adversarial and should be excluded


def test_raid_filters_out_too_short_text():
    rng = random.Random(2)
    rows = [_mock_raid_row(True, None, "news", n_words=5, rng=rng) for _ in range(10)]  # under MIN_WORDS
    rows += [_mock_raid_row(True, None, "news", n_words=60, rng=rng) for _ in range(10)]
    with patch("datasets.load_dataset", return_value=_FakeStreamingDataset(rows)):
        human_rows, ai_rows = download_raid.build_sample(domains=["news"], per_class=10, include_adversarial=False)
    assert all(len(r["generation"].split()) >= download_raid.MIN_WORDS for r in human_rows)


def test_raid_respects_domain_filter():
    rng = random.Random(3)
    rows = [_mock_raid_row(True, None, "poetry", n_words=60, rng=rng) for _ in range(10)]  # excluded domain
    rows += [_mock_raid_row(True, None, "news", n_words=60, rng=rng) for _ in range(10)]
    with patch("datasets.load_dataset", return_value=_FakeStreamingDataset(rows)):
        human_rows, _ = download_raid.build_sample(domains=["news"], per_class=10, include_adversarial=False)
    assert all(r["domain"] == "news" for r in human_rows)


def test_raid_regression_fills_quota_with_few_distinct_models():
    """Regression test for a real bug caught during development: when
    fewer than 11 distinct AI models appear in the stream, the per-model
    soft ceiling must not prevent reaching the overall per_class total."""
    rng = random.Random(7)
    rows = [_mock_raid_row(True, None, "news", n_words=60, rng=rng) for _ in range(60)]
    rows += [_mock_raid_row(False, rng.choice(["chatgpt", "gpt4"]), "news", n_words=60, rng=rng) for _ in range(60)]
    rng.shuffle(rows)
    with patch("datasets.load_dataset", return_value=_FakeStreamingDataset(rows)):
        human_rows, ai_rows = download_raid.build_sample(domains=["news"], per_class=20, include_adversarial=False)
    assert len(human_rows) == 20
    assert len(ai_rows) >= 18  # should nearly/fully fill despite only 2 distinct models


def test_raid_balances_across_available_models():
    rng = random.Random(4)
    models = ["chatgpt", "gpt4", "gpt3", "gpt2"]
    rows = [_mock_raid_row(True, None, "news", n_words=60, rng=rng) for _ in range(100)]
    rows += [_mock_raid_row(False, rng.choice(models), "news", n_words=60, rng=rng) for _ in range(200)]
    rng.shuffle(rows)
    with patch("datasets.load_dataset", return_value=_FakeStreamingDataset(rows)):
        _, ai_rows = download_raid.build_sample(domains=["news"], per_class=40, include_adversarial=False)
    seen_models = {r["model"] for r in ai_rows}
    assert len(seen_models) > 1  # not all pulled from a single model


def test_raid_include_adversarial_flag_keeps_attacked_rows():
    rng = random.Random(5)
    rows = [_mock_raid_row(True, None, "news", n_words=60, rng=rng) for _ in range(10)]
    rows += [_mock_raid_row(False, "chatgpt", "news", attack="paraphrase", decoding="sampling", rep="yes", rng=rng) for _ in range(10)]
    with patch("datasets.load_dataset", return_value=_FakeStreamingDataset(rows)):
        _, ai_rows = download_raid.build_sample(domains=["news"], per_class=10, include_adversarial=True)
    assert len(ai_rows) > 0

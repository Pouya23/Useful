"""Contract tests for OmniVoice planning; no model packages or GPU required."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("omnivoice_samples.py")
SPEC = importlib.util.spec_from_file_location("tested_omni_samples", SCRIPT)
omni = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(omni)


@pytest.mark.parametrize(
    "weights,total,expected",
    [
        ([1], 123, [123]),
        ([1, 1, 1], 5, [2, 2, 1]),
        ([1, 3], 6, [2, 4]),
        ([1000, 1], 2, [1, 1]),
    ],
)
def test_fixed_duration_allocation(weights, total, expected):
    assert omni.allocate_exact(weights, total) == expected


@pytest.mark.parametrize("weights,total", [([], 5), ([0], 5), ([-1, 2], 5), ([1, 2], 1)])
def test_impossible_allocation_is_rejected(weights, total):
    with pytest.raises(ValueError):
        omni.allocate_exact(weights, total)


def test_allocation_conserves_total_over_many_chunks():
    for count in range(1, 101):
        weights = [1 + ((index * 31) % 83) for index in range(count)]
        for total in (count, count + 1, count * 57):
            allocated = omni.allocate_exact(weights, total)
            assert sum(allocated) == total
            assert len(allocated) == count
            assert all(length >= 1 for length in allocated)


def test_bounded_chunking_preserves_entire_10_to_500_word_range():
    for count in range(10, 501):
        text = " ".join(f"word{index}" for index in range(count))
        chunks = omni.split_bounded(text, lambda value: len(value.split()) * 5, 65)
        assert " ".join(chunks) == text
        assert all(len(chunk.split()) * 5 <= 65 for chunk in chunks)


def test_phoneme_control_group_remains_intact():
    text = "They play the [B EY1 S] guitar beside the quiet river today."
    chunks = omni.split_bounded(text, len, 30)
    assert " ".join(chunks) == text
    assert any("[B EY1 S]" in chunk for chunk in chunks)
    assert all(len(chunk) <= 30 for chunk in chunks)


def test_bounded_chunking_prefers_sentence_context():
    text = "One two three four five. Six seven eight nine ten. Eleven twelve."
    chunks = omni.split_bounded(text, len, 40)
    assert chunks[0] == "One two three four five."
    assert " ".join(chunks) == text
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_short_tail_rebalances_without_changing_words_or_exceeding_bound():
    text = " ".join(f"word{index}" for index in range(22))

    def estimate(value):
        return len(value.split()) * 5

    chunks = omni.split_bounded(text, estimate, 50, minimum=15)
    assert " ".join(chunks) == text
    assert all(15 <= estimate(chunk) <= 50 for chunk in chunks)


def test_short_tail_reference_rebalance_keeps_control_group_intact():
    text = "one two three four five six seven eight nine ten [B EY1 S] guitar"

    def estimate(value):
        return len(re.findall(r"\[[^\]]+\]|\S+", value)) * 5

    chunks = omni.split_bounded(text, estimate, 50, minimum=15)
    assert " ".join(chunks) == text
    assert any("[B EY1 S]" in chunk for chunk in chunks)
    assert all(15 <= estimate(chunk) <= 50 for chunk in chunks)


@pytest.mark.parametrize("text", ["", "   ", "an_indivisible_extremely_long_word"])
def test_empty_or_unbounded_token_is_rejected(text):
    with pytest.raises(ValueError):
        omni.split_bounded(text, len, 5)


@pytest.mark.parametrize(
    "samples,rate,expected",
    [
        (960, 24000, 960),
        (960, 44100, 1764),
        (961, 44100, 1766),
        (601, 16000, 401),
        (1, 192000, 8),
        (1, 8000, 1),
        (10**18 + 1, 48000, 2 * (10**18 + 1)),
    ],
)
def test_exact_resampling_counts(samples, rate, expected):
    assert omni.ceil_resampled(samples, 24000, rate) == expected


def test_resampling_is_integer_ceiling_without_float_precision_loss():
    for rate in (8000, 16000, 22050, 24000, 44100, 48000, 96000, 192000):
        for samples in (1, 959, 960, 961, 10001, 1000000000000000001):
            result = omni.ceil_resampled(samples, 24000, rate)
            assert (result - 1) * 24000 < samples * rate <= result * 24000


@pytest.mark.parametrize(
    "config",
    [
        [],
        {"unknown_key": True},
        {"n": 2.5},
        {"n": True},
        {"speed": "fast"},
        {"speed": False},
        {"text": 42},
        {"output_dir": 17},
        {"estimate_only": "false"},
        {"dtype": "int8"},
    ],
)
def test_malformed_configuration_is_rejected(config, tmp_path, monkeypatch):
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["omni", "--config", str(path), "--text", "A valid text input"])
    with pytest.raises(SystemExit) as error:
        omni.parse_arguments()
    assert error.value.code == 2


def test_cli_overrides_json_defaults_and_bom_is_accepted(tmp_path, monkeypatch):
    path = tmp_path / "configuration.json"
    path.write_text(
        json.dumps(
            {
                "text": "one two three four five six seven eight nine ten",
                "n": 2,
                "speed": 0.9,
                "sample_rate": 44100,
                "estimate_only": True,
            }
        ),
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(sys, "argv", ["omni", "--config", str(path), "--n", "4"])
    args = omni.parse_arguments()
    assert args.n == 4 and args.speed == 0.9 and args.sample_rate == 44100 and args.estimate_only


def test_no_reference_defaults_to_supported_deep_male_attributes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["omni", "--text", "A valid text input"])
    assert omni.parse_arguments().instruct == "male, middle-aged, low pitch, british accent"


def test_reference_does_not_add_default_voice_design_instruction(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "omni",
            "--text",
            "A valid text input",
            "--ref-audio",
            "voice.wav",
            "--ref-text",
            "Exact transcript.",
        ],
    )
    assert omni.parse_arguments().instruct is None


def test_existing_reference_cache_does_not_add_default_instruction(tmp_path, monkeypatch):
    cache = tmp_path / "voice.pt"
    cache.touch()
    monkeypatch.setattr(sys, "argv", ["omni", "--text", "A valid text input", "--prompt-cache", str(cache)])
    assert omni.parse_arguments().instruct is None


def test_explicit_voice_instruction_overrides_default(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["omni", "--text", "A valid text input", "--instruct", "female, moderate pitch"]
    )
    assert omni.parse_arguments().instruct == "female, moderate pitch"


def test_peak_limit_cannot_change_duration_or_waveform_relative_shape():
    np = pytest.importorskip("numpy")
    original = np.linspace(-2, 2, 1000, dtype=np.float32)
    limited, metadata = omni.limit_peak(original, 0.98, np)
    assert len(limited) == len(original)
    assert metadata["gain"] == 0.49
    assert np.allclose(limited, original * 0.49)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--text", "hello", "--max-vram-gb", "5.001"],
        ["--text", "hello", "--speed", "nan"],
        ["--text", "hello", "--pause-ms", "-1"],
        ["--text", "hello", "--fade-ms", "100"],
        ["--text", "hello", "--ref-audio", "reference.wav"],
        ["--text", "hello", "--text-file", "narration.txt"],
    ],
)
def test_invalid_cli_contracts(arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["omni", *arguments])
    with pytest.raises(SystemExit) as error:
        omni.parse_arguments()
    assert error.value.code == 2

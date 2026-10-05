"""Focused invariants for the standalone Kokoro planner; no model download."""

import json

import kokoro_samples as kokoro
import numpy as np
import pytest


def test_preserve_every_word_in_requested_range():
    for count in range(10, 501):
        words = [f"word{i}" for i in range(count)]
        text = "  \n ".join(" ".join(words[i : i + 17]) for i in range(0, count, 17))
        chunks = kokoro.initial_chunks(text, 35)
        assert " ".join(chunks).split() == words
        assert all(len(chunk.split()) <= 35 for chunk in chunks)


def test_chunks_choose_sentences_and_clauses_before_hard_word_boundary():
    assert kokoro.initial_chunks("We finish here. The next sentence continues for several words.", 6) == [
        "We finish here.",
        "The next sentence continues for several",
        "words.",
    ]
    assert kokoro.initial_chunks("One two three, four five six seven eight", 6)[0] == "One two three,"
    assert kokoro.initial_chunks("Dr. Smith met J. Doe and then left.", 6)[0] == "Dr. Smith met J. Doe and"
    assert kokoro.initial_chunks("First short paragraph.\nAnother short paragraph.", 35) == [
        "First short paragraph.",
        "Another short paragraph.",
    ]


def test_cosine_edge_fades_reduce_seam_jumps_without_changing_length_or_core_audio():
    signal = np.full(1000, 0.6, dtype=np.float32)
    result = kokoro.edge_fade(signal, 24000, 3)
    assert len(result) == len(signal)
    assert result[0] == result[-1] == 0
    assert np.array_equal(result[72:-72], signal[72:-72])
    assert np.array_equal(signal, np.full(1000, 0.6, dtype=np.float32))
    assert np.array_equal(kokoro.edge_fade(signal, 24000, 0), signal)


@pytest.mark.parametrize(
    "settings", [{"edge_fade_ms": -1}, {"edge_fade_ms": 21}, {"edge_fade_ms": float("nan")}]
)
def test_bad_fade_settings_rejected(settings):
    with pytest.raises(SystemExit):
        kokoro.parse_args(["--text", "replacement text", "--edge-fade-ms", str(settings["edge_fade_ms"])])


def test_exact_ceil_for_duration_pause_and_resampling():
    from fractions import Fraction

    for count in range(10, 501):
        native = count * 7 * 600 + (count // 35) * 887
        for rate in (8000, 16000, 22050, 24000, 44100, 48000, 96000, 192000):
            exact = Fraction(native * rate, 24000)
            assert kokoro.export_length(native, rate) == -(-exact.numerator // exact.denominator)


def test_real_resample_matches_preview():
    signal = pytest.importorskip("scipy.signal")
    for native in (1, 599, 600, 601, 12887):
        audio = np.zeros(native, dtype=np.float32)
        for rate in (16000, 22050, 44100, 48000):
            import math

            divisor = math.gcd(rate, 24000)
            result = signal.resample_poly(audio, rate // divisor, 24000 // divisor)
            assert len(result) == kokoro.export_length(native, rate)


@pytest.mark.parametrize(
    "settings",
    [
        {"n": True},
        {"n": 1.5},
        {"estimate_only": "false"},
        {"speed": "1.0"},
        {"max_vram_gb": 5.1},
        {"speed": float("nan")},
        {"seed": -1},
        {"unknown_setting": 1},
        {"text_file": 10},
    ],
)
def test_invalid_config_rejected(tmp_path, settings):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(SystemExit):
        kokoro.parse_args(["--config", str(config), "--text", "ten words"])


def test_cli_overrides_json_and_opposite_text_source(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"text_file": "unused.txt", "n": 2, "estimate_only": True}), encoding="utf-8"
    )
    args = kokoro.parse_args(
        ["--config", str(config), "--text", "Replacement text", "--n", "3", "--no-estimate-only"]
    )
    assert args.text == "Replacement text"
    assert args.text_file is None
    assert args.n == 3
    assert args.estimate_only is False


def test_supported_decoder_temporal_geometry():
    """Duration doubles before x60 upsampling, then centered ISTFT hop5."""
    for duration in range(1, 501):
        length = duration * 2
        for stride, kernel in ((10, 20), (6, 12)):
            length = (length - 1) * stride - 2 * ((kernel - stride) // 2) + kernel
        length += 1  # Generator's left reflection pad before spectral projection.
        centered_istft_length = (length - 1) * 5
        assert centered_istft_length == duration * kokoro.SAMPLES_PER_DURATION

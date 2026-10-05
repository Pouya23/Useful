"""Focused duration-contract tests; do not require either model's large weights.

Run: python -m pytest -q test_f5_style_planning.py
Torch is needed only for marked architecture tests; scipy/numpy only for
resampling. If _source_checks/f5_cfm.py is present, an additional test executes
the actual pinned CFM.sample duration/clamp code with a dummy flow integrator.
These tests validate length mechanics, not model quality, CUDA peak VRAM or
end-to-end inference with downloaded checkpoints.
"""

from __future__ import annotations
import __future__

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F5, STYLE = load_script("f5_samples"), load_script("styletts2_samples")


@pytest.mark.parametrize("module", [F5, STYLE])
@pytest.mark.parametrize(
    "lengths,pause,overlap,expected",
    [
        ([1000], 500, 400, 1000),
        ([1000, 500, 400], 100, 0, 2100),
        ([1000, 500, 400], 0, 200, 1500),
        ([100, 50, 40], 0, 200, 100),
    ],
)
def test_join_length_includes_exported_pauses_and_clamped_overlaps(module, lengths, pause, overlap, expected):
    assert module.joined_length(lengths, pause, overlap) == expected


def cli(module):
    result = [
        "--text",
        "one two three four five six seven eight nine ten",
        "--repo",
        "repo",
        "--reference-audio",
        "reference.wav",
    ]
    if module is F5:
        result.extend(["--reference-text", "Exact reference transcript."])
    return result


@pytest.mark.parametrize("module", [F5, STYLE])
@pytest.mark.parametrize(
    "settings",
    [
        {"n": "3"},
        {"n": True},
        {"speed": False},
        {"device": 3},
        {"estimate_only": "yes"},
        {"unknown": 1},
        {"wav_subtype": "invalid"},
    ],
)
def test_json_config_rejects_wrong_types_and_unknown_choices(module, settings, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        module.parse_args(cli(module) + ["--config", str(config)])
    assert error.value.code == 2


@pytest.mark.parametrize("module", [F5, STYLE])
def test_config_is_overridden_by_cli_and_preserves_decimal_budget(module, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"n": 2, "speed": 1.2, "max_vram_gb": 5, "estimate_only": True}), encoding="utf-8"
    )
    settings = module.parse_args(cli(module) + ["--config", str(config), "--n", "3"])
    assert (settings.n, settings.speed, settings.max_vram_gb, settings.estimate_only) == (3, 1.2, 5, True)


@pytest.mark.parametrize("module", [F5, STYLE])
def test_explicit_cli_text_and_boolean_override_opposite_config_source(module, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"text_file": "unused.txt", "estimate_only": True}), encoding="utf-8")
    settings = module.parse_args(cli(module) + ["--config", str(config), "--no-estimate-only"])
    assert settings.text_file is None
    assert settings.text is not None
    assert settings.estimate_only is False


@pytest.mark.parametrize("module", [F5, STYLE])
@pytest.mark.parametrize(
    "extra",
    [
        ["--max-vram-gb", "5.001"],
        ["--max-vram-gb", "0"],
        ["--speed", "nan"],
        ["--pause-ms", "5", "--crossfade-ms", "5"],
    ],
)
def test_invalid_numerical_settings_fail_before_model_loading(module, extra):
    with pytest.raises(SystemExit):
        module.parse_args(cli(module) + extra)


def test_chunking_preserves_all_500_words_and_utf8_budget():
    text = " ".join(["naïve", "speech", "duration", "estimate"] * 125)
    chunks = F5.split_text(text, 60)
    assert " ".join(chunks) == text
    assert all(len(chunk.encode("utf-8")) <= 60 for chunk in chunks)
    with pytest.raises(ValueError):
        F5.split_text("word" * 100, 60)


def test_chunking_prefers_complete_sentences_and_preserves_punctuation():
    text = "One two three four five. Six seven eight nine ten. Eleven twelve."
    for chunks in (F5.split_text(text, 40), STYLE.split_text(text, 8)):
        assert chunks[0] == "One two three four five."
        assert " ".join(chunks) == text


@pytest.mark.parametrize("count", [10, 100, 500])
def test_style_sentence_chunking_retains_every_word_within_bound(count):
    text = " ".join((f"word{i}." if i % 9 == 8 else f"word{i}") for i in range(count))
    chunks = STYLE.split_text(text, 35)
    assert " ".join(chunks) == text
    assert all(1 <= len(chunk.split()) <= 35 for chunk in chunks)


def test_f5_short_terminal_chunk_gets_context_when_budget_permits():
    text = "one two three four end"
    chunks = F5.split_text(text, 15)
    assert " ".join(chunks) == text
    assert all(len(chunk.encode("utf-8")) <= 15 for chunk in chunks)
    assert len(chunks[-1].encode("utf-8")) >= 10


def test_f5_tiny_tail_uses_requested_speed_and_same_frozen_frame_contract(monkeypatch):
    torch = pytest.importorskip("torch")
    utility = types.ModuleType("f5_tts.model.utils")
    utility.convert_char_to_pinyin = lambda text: text
    utility.list_str_to_idx = lambda text, vocabulary: torch.ones(1, len(text[0]), dtype=torch.long)
    monkeypatch.setitem(sys.modules, "f5_tts.model.utils", utility)
    arguments = types.SimpleNamespace(max_chunk_bytes=8, max_chunk_seconds=6, speed=1, duration_seconds=None)
    reference = torch.zeros(1, 24000)
    ref_text = "Exact reference transcript. "
    plans = F5.build_plan(arguments, "Example. A.", reference, ref_text, {})
    assert plans[-1]["text"] == "A."
    reference_frames = 24000 // F5.HOP
    requested = reference_frames + int(reference_frames / len(ref_text.encode()) * 2)
    expected_frames = max(requested, max(len(ref_text + "A."), reference_frames + 1) + 1)
    assert plans[-1]["duration_frames"] == expected_frames
    assert plans[-1]["native_samples"] == (expected_frames - reference_frames - 1) * F5.HOP


@pytest.mark.parametrize("module", [F5, STYLE])
@pytest.mark.parametrize("length,fade", [(1000, 120), (5, 120), (1000, 0)])
def test_edge_fades_remove_join_discontinuities_without_changing_length(module, length, fade):
    np = pytest.importorskip("numpy")
    original = np.ones(length, dtype=np.float32)
    result = module.fade_edges(original, fade, np)
    assert result.shape == original.shape
    assert result.dtype == np.float32
    assert np.all(original == 1)
    if fade:
        assert result[0] == result[-1] == 0
    else:
        assert np.array_equal(result, original)


@pytest.mark.parametrize("module", [F5, STYLE])
def test_long_edge_fades_rejected_before_model_loading(module):
    with pytest.raises(SystemExit):
        module.parse_args(cli(module) + ["--fade-ms", "100"])


@pytest.mark.parametrize("module", [F5, STYLE])
@pytest.mark.parametrize("amplitude,expected_gain", [(2.0, 0.49), (0.5, 1.0), (0.0, 1.0)])
def test_pcm_clipping_prevention_is_global_gain_only(module, amplitude, expected_gain):
    np = pytest.importorskip("numpy")
    original = np.linspace(-amplitude, amplitude, 1000, dtype=np.float32)
    limited, metadata = module.limit_peak(original, 0.98, np)
    assert len(limited) == len(original)
    assert metadata["gain"] == pytest.approx(expected_gain)
    assert np.allclose(limited, original * expected_gain)
    assert float(np.max(np.abs(limited))) <= 0.980001


@pytest.mark.parametrize("frames", [2, 3, 100, 511])
def test_real_torch_centered_vocos_istft_length(frames):
    torch = pytest.importorskip("torch")
    spec = torch.complex(torch.randn(1, 513, frames), torch.randn(1, 513, frames))
    result = torch.istft(spec, 1024, 256, 1024, torch.hann_window(1024), center=True)
    assert result.shape[-1] == (frames - 1) * 256


@pytest.mark.parametrize("frames", [1, 2, 17, 100, 501])
def test_real_torch_style_hifigan_temporal_topology(frames):
    torch = pytest.importorskip("torch")
    wave = torch.nn.ConvTranspose1d(1, 1, 3, 2, 1, output_padding=1)(torch.zeros(1, 1, frames))
    for stride, kernel in zip((10, 5, 3, 2), (20, 10, 6, 4)):
        wave = torch.nn.ConvTranspose1d(
            1, 1, kernel, stride, padding=stride // 2 + stride % 2, output_padding=stride % 2
        )(wave)
    assert wave.shape[-1] == frames * 600


def test_actual_pinned_cfm_sample_duration_clamps_without_flow_generation():
    torch = pytest.importorskip("torch")
    source = ROOT / "_source_checks/f5_cfm.py"
    if not source.exists():
        pytest.skip("Optional upstream source snapshot absent")
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "CFM")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "sample")
    namespace = {
        "torch": torch,
        "F": torch.nn.functional,
        "exists": lambda value: value is not None,
        "lens_to_mask": lambda lengths: torch.arange(int(lengths.max()))[None, :] < lengths[:, None],
        "pad_sequence": torch.nn.utils.rnn.pad_sequence,
        "get_epss_timesteps": lambda steps, device, dtype: torch.linspace(
            0, 1, steps + 1, device=device, dtype=dtype
        ),
        "odeint": lambda fn, initial, times, **kwargs: torch.stack([initial for _ in times]),
    }
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            "official_cfm_sample",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )

    class Dummy:
        device, num_channels, vocab_char_map, odeint_kwargs = torch.device("cpu"), 100, None, {}
        transformer = types.SimpleNamespace(clear_cache=lambda: None)

        def eval(self):
            return self

        def parameters(self):
            return iter([torch.empty(1)])

    for conditioning, tokens, request, expected in [
        (94, 15, 150, 150),
        (94, 120, 100, 121),
        (94, 15, 10, 95),
        (94, 15, 70000, 65536),
    ]:
        result, _ = namespace["sample"](
            Dummy(),
            torch.zeros(1, conditioning, 100),
            torch.zeros(1, tokens, dtype=torch.long),
            request,
            steps=2,
        )
        assert result.shape[1] == expected


@pytest.mark.parametrize(
    "source_rate,target_rate,length",
    [(24000, 44100, 257), (24000, 48000, 257), (24000, 16000, 257), (24000, 22050, 501)],
)
def test_export_resampling_ceil_length(source_rate, target_rate, length):
    np = pytest.importorskip("numpy")
    signal = pytest.importorskip("scipy.signal")
    import math

    factor = math.gcd(source_rate, target_rate)
    result = signal.resample_poly(
        np.zeros(length, dtype=np.float32), target_rate // factor, source_rate // factor
    )
    assert len(result) == (length * target_rate + source_rate - 1) // source_rate

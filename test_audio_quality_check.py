"""Read-only acoustic/ASR report contracts; no recognition weights or network."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

SPEC = importlib.util.spec_from_file_location(
    "audio_quality_tested", Path(__file__).with_name("audio_quality_check.py")
)
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


def test_normalization_ignores_case_punctuation_without_inventing_number_expansions():
    assert quality.words("THE cat's 23 apples; naïve speech.") == [
        "the",
        "cat's",
        "23",
        "apples",
        "naïve",
        "speech",
    ]
    assert quality.words("They’re here.") == ["they're", "here"]


@pytest.mark.parametrize(
    "reference,hypothesis,deletions,insertions,substitutions",
    [
        ("One two three", "one two three", 0, 0, 0),
        ("one two three", "one three", 1, 0, 0),
        ("one two three", "one two two three", 0, 1, 0),
        ("one two three", "one four three", 0, 0, 1),
        ("one two three", "", 3, 0, 0),
        ("", "one two", 0, 2, 0),
    ],
)
def test_word_edit_alignment_recovers_omissions_repetitions_and_substitutions(
    reference, hypothesis, deletions, insertions, substitutions
):
    result = quality.word_alignment(reference, hypothesis)
    assert (result["deletions"], result["insertions"], result["substitutions"]) == (
        deletions,
        insertions,
        substitutions,
    )
    assert len(result["word_edits"]) == deletions + insertions + substitutions
    expected_rate = (deletions + insertions + substitutions) / len(reference.split()) if reference else None
    assert result["word_error_rate"] == expected_rate


def test_500_word_alignment_keeps_missing_final_word_inspectable():
    reference = " ".join(f"word{index}" for index in range(500))
    result = quality.word_alignment(reference, " ".join(reference.split()[:-1]))
    assert result["deletions"] == 1
    assert result["word_error_rate"] == 1 / 500
    assert result["word_edits"][-1]["expected"] == "word499"


def test_acoustic_silence_excludes_both_edges_and_measures_internal_pause():
    rate = 1000
    wave = np.concatenate(
        (np.zeros(100), np.ones(200) * 0.1, np.zeros(800), np.ones(200) * 0.1, np.zeros(300))
    )
    result = quality.acoustic_stats(wave, rate, np)
    assert result["frames"] == 1600
    assert result["actual_duration_seconds"] == 1.6
    assert result["leading_silence_seconds"] == 0.1
    assert result["trailing_silence_seconds"] == 0.3
    assert result["longest_interior_silence_seconds"] == 0.8
    assert result["interior_silence_spans"] == [
        {"start_seconds": 0.3, "end_seconds": 1.1, "duration_seconds": 0.8}
    ]


def test_stereo_opposite_phase_does_not_falsely_become_silence():
    wave = np.tile([0.1, -0.1], (1000, 1))
    result = quality.acoustic_stats(wave, 1000, np)
    assert result["quiet_sample_fraction"] == 0
    assert result["channels"] == 2


def test_nonfinite_and_clipping_stats_are_finite_serializable_diagnostics():
    result = quality.acoustic_stats(np.array([0, 1, -1, np.nan, np.inf]), 1000, np)
    assert result["clipped_sample_fraction"] == 2 / 5
    assert result["nonfinite_sample_fraction"] == pytest.approx(2 / 5)
    assert result["largest_adjacent_sample_delta"] == 2
    json.dumps(result, allow_nan=False)


def test_missing_file_retains_a_failure_row_and_does_not_mutate_other_audio(tmp_path):
    path = tmp_path / "speech.wav"
    sf.write(path, np.ones(1000, dtype=np.float32) * 0.1, 1000, subtype="FLOAT")
    original = path.read_bytes()
    rows = quality.analyze_requests(
        [
            {"job_id": "missing", "audio_file": str(tmp_path / "missing.wav"), "text": "Original words"},
            {"job_id": "valid", "audio_file": str(path), "text": "Original words", "seed": 42},
        ],
        None,
        "disabled",
        "en",
        np,
        sf,
    )
    assert [row["status"] for row in rows] == ["failed", "acoustic_checked"]
    assert rows[1]["seed"] == 42
    assert rows[1]["word_error_rate"] is None
    assert path.read_bytes() == original


def test_fake_cpu_asr_does_not_receive_reference_prompt_and_reports_missing_word(tmp_path):
    path = tmp_path / "speech.wav"
    sf.write(path, np.ones(1000, dtype=np.float32) * 0.1, 1000, subtype="FLOAT")
    calls = []

    class Recognizer:
        def transcribe(self, audio_file, **kwargs):
            calls.append(kwargs)
            segments = [SimpleNamespace(start=0, end=1, text="One three", words=[])]
            return iter(segments), SimpleNamespace(language="en")

    rows = quality.analyze_requests(
        [{"audio_file": str(path), "text": "One two three"}], Recognizer(), "ready", "en", np, sf
    )
    assert rows[0]["asr_status"] == "completed"
    assert rows[0]["transcript"] == "One three"
    assert rows[0]["deletions"] == 1
    assert "initial_prompt" not in calls[0] and "prefix" not in calls[0]
    assert calls[0]["vad_filter"] is False


def test_failed_recognizer_retains_valid_acoustic_metrics(tmp_path):
    path = tmp_path / "speech.wav"
    sf.write(path, np.ones(1000, dtype=np.float32) * 0.1, 1000, subtype="FLOAT")

    class Recognizer:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("deliberate recognition failure")

    row = quality.analyze_requests(
        [{"audio_file": str(path), "text": "One two three"}], Recognizer(), "ready", "en", np, sf
    )[0]
    assert row["status"] == "acoustic_checked"
    assert row["frames"] == 1000
    assert row["asr_status"] == "failed"
    assert row["word_error_rate"] is None


def test_cli_produces_report_and_hides_gpu_without_touching_wav(tmp_path, monkeypatch):
    path, requests, report = tmp_path / "speech.wav", tmp_path / "requests.json", tmp_path / "quality.json"
    sf.write(path, np.ones(1000, dtype=np.float32) * 0.1, 1000, subtype="FLOAT")
    requests.write_text(json.dumps([{"audio_file": str(path), "text": "One two three"}]), encoding="utf-8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-visible")
    assert quality.main(["--requests-json", str(requests), "--output-json", str(report)]) == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    payload = json.loads(report.read_text())
    assert payload["asr_status"] == "disabled"
    assert payload["rows"][0]["status"] == "acoustic_checked"


def test_asr_loading_failure_retains_acoustic_report(tmp_path, monkeypatch):
    path, requests, report = tmp_path / "speech.wav", tmp_path / "requests.json", tmp_path / "quality.json"
    sf.write(path, np.ones(1000, dtype=np.float32) * 0.1, 1000, subtype="FLOAT")
    requests.write_text(json.dumps([{"audio_file": str(path), "text": "One two three"}]), encoding="utf-8")

    def unavailable(*args, **kwargs):
        raise RuntimeError("No network/model available")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=unavailable))
    assert quality.main(["--requests-json", str(requests), "--output-json", str(report), "--asr"]) == 0
    payload = json.loads(report.read_text())
    assert payload["asr_status"] == "unavailable"
    assert payload["rows"][0]["frames"] == 1000

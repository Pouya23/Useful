#!/usr/bin/env python3
"""Read-only CPU checks for TTS recordings and optional independent ASR.

Install numpy and soundfile. For --asr, the tested optional dependencies are:
  pip install faster-whisper==1.2.1 ctranslate2==4.6.0 av==15.1.0
  python audio_quality_check.py --requests-json requests.json --output-json quality.json --asr

requests.json is a list of objects with audio_file and text, plus optional
job_id, model, seed, and other identifying fields. Files are never modified.
Silence/delta/clipping warnings are acoustic heuristics, not judgments of
intelligibility or naturalness. ASR word errors are comparisons against an
independent recognizer's transcript, not proof of omitted spoken words.
Punctuation/case are ignored; numbers and contractions are not rewritten.
The recognizer receives no reference-text prompt. CUDA is hidden and ASR
uses CPU int8. Model-loading or per-file ASR failures retain acoustic rows.

API: https://github.com/SYSTRAN/faster-whisper
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any


def finite_number(value: Any) -> float | None:
    """Keep optional recognizer timing/confidence metadata JSON-safe."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def words(text: str) -> list[str]:
    """Unicode words with case/punctuation normalization only."""
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", normalized, flags=re.UNICODE)


def word_alignment(reference: str, hypothesis: str) -> dict[str, Any]:
    """Minimum word edit distance with deterministic, inspectable alignment."""
    expected, actual = words(reference), words(hypothesis)
    rows, columns = len(expected) + 1, len(actual) + 1
    costs = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        costs[i][0] = i
    for j in range(columns):
        costs[0][j] = j
    for i in range(1, rows):
        for j in range(1, columns):
            costs[i][j] = min(
                costs[i - 1][j - 1] + (expected[i - 1] != actual[j - 1]),
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
            )
    i, j, alignment = len(expected), len(actual), []
    counts = {"deletions": 0, "insertions": 0, "substitutions": 0}
    while i or j:
        if i and j and expected[i - 1] == actual[j - 1] and costs[i][j] == costs[i - 1][j - 1]:
            i, j = i - 1, j - 1
            continue
        if i and j and costs[i][j] == costs[i - 1][j - 1] + 1:
            alignment.append(
                {
                    "type": "substitution",
                    "reference_index": i - 1,
                    "hypothesis_index": j - 1,
                    "expected": expected[i - 1],
                    "actual": actual[j - 1],
                }
            )
            counts["substitutions"] += 1
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            alignment.append(
                {
                    "type": "deletion",
                    "reference_index": i - 1,
                    "hypothesis_index": j,
                    "expected": expected[i - 1],
                    "actual": None,
                }
            )
            counts["deletions"] += 1
            i -= 1
        else:
            alignment.append(
                {
                    "type": "insertion",
                    "reference_index": i,
                    "hypothesis_index": j - 1,
                    "expected": None,
                    "actual": actual[j - 1],
                }
            )
            counts["insertions"] += 1
            j -= 1
    alignment.reverse()
    errors = sum(counts.values())
    return {
        "reference_word_count": len(expected),
        "transcript_word_count": len(actual),
        "word_errors": errors,
        "word_error_rate": errors / len(expected) if expected else None,
        **counts,
        "word_edits": alignment,
        "normalization": "Unicode NFKC/casefold; punctuation ignored; numbers and contractions retained",
        "interpretation": "ASR transcript differences are diagnostic evidence, not proof of speech omissions.",
    }


def acoustic_stats(samples: Any, sample_rate: int, np: Any) -> dict[str, Any]:
    """20 ms RMS silence spans, exact peak/clipping counts, and adjacent delta."""
    if sample_rate < 1:
        raise ValueError("Sample rate must be positive")
    wave = np.asarray(samples, dtype=np.float64)
    if wave.ndim == 1:
        wave = wave[:, None]
    if wave.ndim != 2 or not wave.shape[0] or not wave.shape[1]:
        raise ValueError("Audio must have nonempty frames and one or more channels")
    finite = np.isfinite(wave)
    clean = np.where(finite, wave, 0)
    frame_count = wave.shape[0]
    duration = frame_count / sample_rate
    hop = max(1, round(sample_rate * 0.020))
    spans = [(start, min(frame_count, start + hop)) for start in range(0, frame_count, hop)]
    quiet = [float(np.sqrt(np.mean(clean[start:stop] ** 2))) < 10 ** (-50 / 20) for start, stop in spans]
    regions, start = [], None
    for index, is_quiet in enumerate(quiet + [False]):
        if is_quiet and start is None:
            start = index
        elif not is_quiet and start is not None:
            regions.append((spans[start][0], spans[index - 1][1]))
            start = None
    leading = next((stop / sample_rate for begin, stop in regions if begin == 0), 0.0)
    trailing = next(
        ((frame_count - begin) / sample_rate for begin, stop in regions if stop == frame_count), 0.0
    )
    interior = [
        {
            "start_seconds": begin / sample_rate,
            "end_seconds": stop / sample_rate,
            "duration_seconds": (stop - begin) / sample_rate,
        }
        for begin, stop in regions
        if begin > 0 and stop < frame_count
    ]
    peak = float(np.max(np.abs(clean)))
    clipped = float(np.count_nonzero(np.abs(clean) >= 0.999) / wave.size)
    nonfinite = float(1 - np.count_nonzero(finite) / wave.size)
    valid_adjacent = finite[:-1] & finite[1:]
    delta = np.abs(np.diff(clean, axis=0))
    largest_delta = float(np.max(np.where(valid_adjacent, delta, 0))) if frame_count > 1 else 0.0
    longest = max((span["duration_seconds"] for span in interior), default=0.0)
    rms = float(np.sqrt(np.mean(clean**2)))
    warnings = []
    if nonfinite:
        warnings.append("nonfinite_samples")
    if clipped > 0.0001:
        warnings.append("possible_clipping_or_saturation")
    if longest >= 0.6:
        warnings.append("long_interior_quiet_span; compare against intended punctuation pauses")
    if largest_delta > 0.5:
        warnings.append("large_adjacent_sample_delta; possible click, not an audibility verdict")
    if rms < 10 ** (-50 / 20):
        warnings.append("near_silent_audio")
    return {
        "sample_rate": sample_rate,
        "frames": frame_count,
        "channels": wave.shape[1],
        "actual_duration_seconds": duration,
        "peak_amplitude": peak,
        "rms_amplitude": rms,
        "clipped_sample_fraction": clipped,
        "nonfinite_sample_fraction": nonfinite,
        "largest_adjacent_sample_delta": largest_delta,
        "leading_silence_seconds": leading,
        "trailing_silence_seconds": trailing,
        "longest_interior_silence_seconds": longest,
        "interior_silence_spans": interior,
        "quiet_sample_fraction": sum(stop - begin for begin, stop in regions) / frame_count,
        "silence_analysis": "20 ms nonoverlapping RMS windows below -50 dBFS; channel energies retained",
        "acoustic_warnings": warnings,
    }


def analyze_requests(
    requests: list[Any], asr_model: Any, asr_status: str, language: str, np: Any, sf: Any
) -> list[dict[str, Any]]:
    """Keep one result per request, including unusable audio and ASR failures."""
    rows = []
    for index, request in enumerate(requests):
        started = time.perf_counter()
        row = dict(request) if isinstance(request, dict) else {"request": request}
        row.update(
            status="pending",
            asr_status=asr_status,
            transcript=None,
            word_error_rate=None,
            deletions=None,
            insertions=None,
            substitutions=None,
        )
        print(
            f"Quality check {index + 1}/{len(requests)}: {row.get('audio_file', '<missing path>')}",
            flush=True,
        )
        try:
            if not isinstance(request, dict) or not isinstance(request.get("audio_file"), str):
                raise ValueError("Every request needs an audio_file path string")
            if not isinstance(request.get("text"), str):
                raise ValueError("Every request needs its original text string")
            audio_path = Path(request["audio_file"]).resolve()
            wave, rate = sf.read(audio_path, dtype="float32", always_2d=True)
            row["audio_file"] = str(audio_path)
            row.update(acoustic_stats(wave, rate, np))
            row["status"] = "acoustic_checked"
            if asr_model is not None:
                if row["nonfinite_sample_fraction"]:
                    row["asr_status"] = "skipped_nonfinite_audio"
                else:
                    try:
                        asr_started = time.perf_counter()
                        segments, info = asr_model.transcribe(
                            str(audio_path),
                            language=language or None,
                            beam_size=5,
                            temperature=0.0,
                            condition_on_previous_text=False,
                            vad_filter=False,
                            word_timestamps=True,
                        )
                        segment_rows = []
                        for segment in segments:
                            segment_rows.append(
                                {
                                    "start_seconds": finite_number(segment.start),
                                    "end_seconds": finite_number(segment.end),
                                    "text": segment.text,
                                    "words": [
                                        {
                                            "start_seconds": finite_number(word.start),
                                            "end_seconds": finite_number(word.end),
                                            "word": word.word,
                                            "probability": finite_number(word.probability),
                                        }
                                        for word in (segment.words or [])
                                    ],
                                }
                            )
                        transcript = " ".join(segment["text"].strip() for segment in segment_rows).strip()
                        row.update(word_alignment(request["text"], transcript))
                        row.update(
                            transcript=transcript,
                            asr_segments=segment_rows,
                            asr_status="completed",
                            asr_seconds=time.perf_counter() - asr_started,
                            recognized_language=str(info.language),
                        )
                    except Exception as error:
                        row.update(asr_status="failed", asr_error=f"{type(error).__name__}: {error}")
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        row["quality_check_seconds"] = time.perf_counter() - started
        rows.append(row)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--requests-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--asr", action="store_true", help="Run optional CPU independent Whisper transcription"
    )
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--language", default="en", help="Empty string enables automatic language detection")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--download-root", type=Path)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Set before any optional engine import, even if the parent exported GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import soundfile as sf

    requests = json.loads(args.requests_json.read_text(encoding="utf-8-sig"))
    if not isinstance(requests, list):
        raise ValueError("requests-json must contain a list")
    asr_model, asr_status, asr_error = None, "disabled", None
    if args.asr:
        try:
            from faster_whisper import WhisperModel

            asr_model = WhisperModel(
                args.model,
                device="cpu",
                compute_type="int8",
                cpu_threads=args.threads,
                num_workers=1,
                download_root=str(args.download_root) if args.download_root else None,
            )
            asr_status = "ready"
        except Exception as error:
            asr_status, asr_error = "unavailable", f"{type(error).__name__}: {error}"
            print(f"ASR unavailable; retaining acoustic checks: {asr_error}", flush=True)
    rows = analyze_requests(requests, asr_model, asr_status, args.language, np, sf)
    report = {
        "status": "completed_with_errors" if any(row["status"] == "failed" for row in rows) else "completed",
        "asr_status": asr_status,
        "asr_error": asr_error,
        "model": args.model,
        "asr_device": "cpu",
        "asr_compute_type": "int8",
        "rows": rows,
        "interpretation": "Acoustic warnings and independent ASR comparisons require listening; exact duration does not establish quality.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_name(args.output_json.name + ".part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output_json)
    print(f"Saved {len(rows)} quality rows to {args.output_json.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

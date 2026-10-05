#!/usr/bin/env python3
"""OmniVoice samples with an exact, pre-synthesis waveform-length plan.

Install in a separate Python 3.10+ environment (prefer Python 3.11/3.12):
  python -m pip install torch==2.8.0 torchaudio==2.8.0
  python -m pip install "git+https://github.com/k2-fsa/OmniVoice.git@08be0b4ccbac3e13e374e86fbfead4b4cac343e2" "transformers==5.3.0" scipy tqdm
For CUDA use the matching PyTorch CUDA wheel index. FFmpeg may be necessary
for compressed reference audio. Optional normalization needs OmniVoice's [tn].

Examples:
  python omnivoice_samples.py --text-file narration.txt --n 3 --estimate-only
  python omnivoice_samples.py --text-file narration.txt --n 3 --instruct "male, middle-aged, low pitch, british accent"
  python omnivoice_samples.py --text-file narration.txt --ref-audio ref.wav --ref-text "Exact reference transcript." --prompt-cache voice.pt
  python omnivoice_samples.py --text-file narration.txt --prompt-cache voice.pt --estimate-only
  python omnivoice_samples.py --self-test

The upstream rule estimator allocates audio tokens. This script calculates that
allocation BEFORE generating speech and supplies exactly those target_lens to
upstream iterative inference. It is not an independently validated prediction
of unconstrained natural speaking time. No waveform is stretched, trimmed, or
padded to conceal a duration mismatch. Silence removal and upstream long-form
chunking are disabled; explicit inter-chunk pauses are counted. Manual chunks
are bounded and sequential. In auto/design mode chunk one's generated tokens
condition subsequent chunks for consistency without re-estimating frozen lengths.
Chunks prefer sentences/clauses and short terminal chunks are rebalanced when
the configured bound permits. Without a reference or explicit instruction the
voice design uses a middle-aged, low-pitched male British speaker. These are
supported attributes, not a guarantee of a particular identity or serious mood.

Native output is 24 kHz, 960 samples/token. Export resampling changes sample
rate, not acoustic bandwidth. Exact sample-count checks are stricter than 1%.
Planning time excludes imports, download/loading, and reference preparation.
Warm allocation is fast; cold-start latency is separately reported. A new
reference needs preparation; an existing --prompt-cache avoids it.
CPU is default and uses zero GPU VRAM. Optional CUDA bounds the PyTorch
allocator and reports its measured peak, not a guaranteed whole-driver peak.
An allocator cap cannot constrain driver/library allocations. Use CPU when a
hard <=5 GB GPU ceiling is indispensable. Generation quality, intelligibility,
and near-instant latency require measurement on your hardware.

Pinned source/API and codec length mapping:
https://github.com/k2-fsa/OmniVoice/tree/08be0b4ccbac3e13e374e86fbfead4b4cac343e2
https://huggingface.co/k2-fsa/OmniVoice/tree/c5fdb5ccb189668d56333f77ba2629f4cd7535f4
https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py
https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/dac/modeling_dac.py
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Union

SOURCE_COMMIT = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
MODEL_ID = "k2-fsa/OmniVoice"
MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
NATIVE_RATE = 24000
TOKEN_HOP = 960
FRAME_RATE = 25
DEFAULT_VOICE_INSTRUCT = "male, middle-aged, low pitch, british accent"
SOURCE_HASHES = {
    "omnivoice.utils.voice_design": "23fbbbe31641b6e23c54fd46d7c64956e032239a6dfd96dab2e8615e598f46ab",
    "omnivoice.utils.lang_map": "e7924def0990d215aa94356f2ab17512a7b77e9cd43bc9bded093f757dfc1cd9",
    "omnivoice.models.omnivoice": "631050ba94775b9c8a72ec3b2d84777c38317e2e502b1998d3128fedf88846ff",
    "omnivoice.utils.duration": "3a0b468e89eb2a109a270fe4386ce1afe57305c08375a524a1af08399dfaf772",
    "omnivoice.utils.text": "ec833278ea7dcb22dfd3d27c3a932f8fb64948826242abb721a63b33b1c0a45a",
}


def ceil_resampled(samples: int, original: int, exported: int) -> int:
    return (samples * exported + original - 1) // original


def allocate_exact(weights: list[int], total: int) -> list[int]:
    """Largest-remainder allocation with at least one frame per chunk."""
    if not weights or total < len(weights) or any(w <= 0 for w in weights):
        raise ValueError("Duration must allocate at least one token per text chunk.")
    remaining = total - len(weights)
    denominator = sum(weights)
    numerators = [remaining * w for w in weights]
    lengths = [1 + value // denominator for value in numerators]
    leftover = total - sum(lengths)
    order = sorted(range(len(weights)), key=lambda i: (-(numerators[i] % denominator), i))
    for i in order[:leftover]:
        lengths[i] += 1
    return lengths


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument("--text")
    text_group.add_argument("--text-file", type=Path)
    parser.add_argument(
        "--config", type=Path, help="JSON object of CLI destinations; CLI values override it."
    )
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda[:index]")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/omnivoice"))
    parser.add_argument("--sample-rate", type=int, default=None, help="Export Hz; native default 24000.")
    parser.add_argument("--pause-ms", type=float, default=0.0, help="Configured gap between manual chunks.")
    parser.add_argument(
        "--max-vram-gb", type=float, default=5.0, help="Decimal GB, <=5; CUDA allocator budget only."
    )
    parser.add_argument(
        "--estimate-only", action="store_true", help="No model download/loading or speech generation."
    )
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--instruct", default=None, help="Comma-separated supported attributes, e.g. 'male, whisper'."
    )
    parser.add_argument("--ref-audio", type=Path, default=None)
    parser.add_argument(
        "--ref-text", default=None, help="Required exact reference transcript; ASR is never loaded."
    )
    parser.add_argument(
        "--prompt-cache", type=Path, default=None, help="Load an existing VoiceClonePrompt or save a new one."
    )
    parser.add_argument("--preprocess-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Total speech-token seconds, before inter-chunk pauses; overrides speed.",
    )
    parser.add_argument("--max-chunk-seconds", type=float, default=8.0)
    parser.add_argument("--max-reference-seconds", type=float, default=10.0)
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--t-shift", type=float, default=0.1)
    parser.add_argument("--layer-penalty-factor", type=float, default=5.0)
    parser.add_argument("--position-temperature", type=float, default=5.0)
    parser.add_argument("--class-temperature", type=float, default=0.0)
    parser.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fade-ms", type=float, default=10.0, help="Length-preserving fades per manual chunk."
    )
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument(
        "--peak-limit", type=float, default=0.98, help="Peak ceiling via amplitude-only attenuation, (0,1]"
    )
    parser.add_argument(
        "--local-model", type=Path, default=None, help="Local snapshot of the pinned official model."
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--self-test", action="store_true", help="Dependency-free arithmetic/chunking checks."
    )
    early, _ = parser.parse_known_args()
    if early.config:
        config = json.loads(early.config.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict):
            parser.error("--config must contain a JSON object.")
        actions = {action.dest: action for action in parser._actions}
        for key, value in config.items():
            key = key.replace("-", "_")
            if key not in actions or key in {"help", "config", "self_test"}:
                parser.error(f"Unknown or disallowed config key: {key}")
            action = actions[key]
            if isinstance(
                action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)
            ) and not isinstance(value, bool):
                parser.error(f"Config {key} must be boolean.")
            if (
                value is not None
                and action.type is int
                and (not isinstance(value, int) or isinstance(value, bool))
            ):
                parser.error(f"Config {key} must be an integer.")
            if (
                value is not None
                and action.type is float
                and (not isinstance(value, (int, float)) or isinstance(value, bool))
            ):
                parser.error(f"Config {key} must be a number.")
            if value is not None and action.type is Path and not isinstance(value, str):
                parser.error(f"Config {key} must be a path string.")
            if (
                value is not None
                and action.type is None
                and not isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction))
                and not isinstance(value, str)
            ):
                parser.error(f"Config {key} must be a string.")
            if action.type and value is not None:
                try:
                    value = action.type(value)
                except (TypeError, ValueError) as error:
                    parser.error(f"Invalid config {key}: {error}")
            if action.choices and value not in action.choices:
                parser.error(f"Invalid config {key}: {value}")
            parser.set_defaults(**{key: value})
    args = parser.parse_args()
    if args.self_test:
        return args
    if (args.text is None) == (args.text_file is None):
        parser.error("Provide exactly one of --text or --text-file, including configuration defaults.")
    if args.n < 1 or args.seed < 0 or args.seed + args.n - 1 >= 2**32:
        parser.error("--n must be positive and every seed must be within [0, 2**32).")
    finite_positive = ("speed", "max_chunk_seconds", "max_reference_seconds", "max_vram_gb")
    finite_nonnegative = (
        "pause_ms",
        "fade_ms",
        "guidance_scale",
        "layer_penalty_factor",
        "position_temperature",
        "class_temperature",
    )
    for key in finite_positive + finite_nonnegative:
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key in finite_positive and value == 0):
            parser.error(
                f"--{key.replace('_', '-')} must be finite and {'positive' if key in finite_positive else 'nonnegative'}."
            )
    if args.max_vram_gb > 5:
        parser.error("--max-vram-gb cannot exceed the requested 5 GB ceiling.")
    if not math.isfinite(args.peak_limit) or not 0 < args.peak_limit <= 1:
        parser.error("--peak-limit must be finite and in (0,1].")
    if args.fade_ms > 20:
        parser.error("--fade-ms must be in [0,20]; long fades can erase initial/final speech.")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite.")
    if not math.isfinite(args.t_shift) or args.t_shift <= 0 or args.num_step < 1:
        parser.error("--t-shift and --num-step must be positive.")
    if args.sample_rate is not None and not 8000 <= args.sample_rate <= 192000:
        parser.error("--sample-rate must be between 8000 and 192000 Hz.")
    if args.max_chunk_seconds * FRAME_RATE < 1:
        parser.error("--max-chunk-seconds must allow at least one audio token.")
    if args.device != "cpu" and re.fullmatch(r"cuda(?::\d+)?", args.device) is None:
        parser.error("Supported --device values: cpu or cuda[:index].")
    if args.device == "cpu" and args.dtype == "float16":
        parser.error("Use float32 or bfloat16 on CPU; float16 CPU kernels are not supported consistently.")
    if args.ref_audio is not None and not args.ref_text:
        parser.error("--ref-audio requires --ref-text; automatic transcription is disabled.")
    if args.ref_text and args.ref_audio is None:
        parser.error("--ref-text requires --ref-audio. Cached prompts contain their transcript.")
    if args.prompt_cache and args.prompt_cache.exists() and args.ref_audio is not None:
        parser.error("Use either an existing --prompt-cache or --ref-audio/--ref-text.")
    if args.prompt_cache and not args.prompt_cache.exists() and args.ref_audio is None:
        parser.error("A missing --prompt-cache requires --ref-audio and --ref-text.")
    if args.instruct is None and args.ref_audio is None and not args.prompt_cache:
        args.instruct = DEFAULT_VOICE_INSTRUCT
    return args


def split_bounded(text: str, estimate: Callable[[str], int], maximum: int, minimum: int = 0) -> list[str]:
    """Bound allocations, preserve controls, and favor sentence/clause endings."""
    if maximum < 1 or minimum < 0:
        raise ValueError("Require a positive chunk bound and a nonnegative preferred minimum.")
    units = re.findall(r"\[[^\]]+\]|\S+", text)
    chunks: list[str] = []
    for unit in units:
        if estimate(unit) > maximum:
            raise ValueError(f"One indivisible word/control group exceeds the chunk limit: {unit!r}")
    start = 0
    while start < len(units):
        end = start + 1
        while end < len(units) and estimate(" ".join(units[start : end + 1])) <= maximum:
            end += 1
        if end < len(units):
            lower = start + max(1, (end - start) // 2)
            for pattern in (r"[.!?。！？][\"'”’\])}]*$", r"[,;:，；：][\"'”’\])}]*$"):
                candidates = [i for i in range(lower, end + 1) if re.search(pattern, units[i - 1])]
                if candidates:
                    end = candidates[-1]
                    break
        chunks.append(" ".join(units[start:end]))
        start = end
    if not units:
        raise ValueError("Normalized text contains no speech units.")
    # Very short unreferenced speech is less stable. Rebalance a small final
    # chunk with its predecessor when both allocations can remain bounded.
    # This only changes the text/length plan; no audio is cropped or padded.
    if minimum and len(chunks) > 1 and estimate(chunks[-1]) < minimum:
        combined = chunks[-2] + " " + chunks[-1]
        if estimate(combined) <= maximum:
            chunks[-2:] = [combined]
        else:
            combined_units = re.findall(r"\[[^\]]+\]|\S+", combined)
            original_left = estimate(chunks[-2])
            candidates = []
            for cut in range(1, len(combined_units)):
                left, right = " ".join(combined_units[:cut]), " ".join(combined_units[cut:])
                left_length, right_length = estimate(left), estimate(right)
                if minimum <= left_length <= maximum and minimum <= right_length <= maximum:
                    boundary = bool(re.search(r"[.!?。！？,;:，；：][\"'”’\])}]*$", combined_units[cut - 1]))
                    candidates.append((not boundary, abs(original_left - left_length), cut, left, right))
            if candidates:
                selected = min(candidates)
                chunks[-2:] = [selected[-2], selected[-1]]
    return chunks


def limit_peak(wave: Any, ceiling: float, np: Any) -> tuple[Any, dict[str, float]]:
    """Avoid playback overdrive with one amplitude-only, duration-preserving gain."""
    original_peak = float(np.max(np.abs(wave)))
    gain = min(1.0, ceiling / original_peak) if original_peak else 1.0
    return wave * gain, {"original_peak": original_peak, "gain": gain, "ceiling": ceiling}


def lightweight_preflight() -> tuple[Any, Any, Any]:
    """Load hash-verified stdlib utilities without importing the neural package.

    The upstream package initializer imports Torch. Loading its pure utility
    files directly and the two verified resolver functions avoids that cold
    start in auto/design preview mode. Model inference still imports and checks
    the complete pinned package before any generation.
    """
    spec = importlib.util.find_spec("omnivoice")
    if spec is None or spec.origin is None:
        raise ImportError("OmniVoice is not installed.")
    root = Path(spec.origin).parent
    sources = {}
    for name, expected in SOURCE_HASHES.items():
        filename = root.joinpath(*name.split(".")[1:]).with_suffix(".py")
        code = filename.read_text(encoding="utf-8").replace("\r\n", "\n")
        if hashlib.sha256(code.encode("utf-8")).hexdigest() != expected:
            raise RuntimeError(f"Upstream source changed: {name}. Install {SOURCE_COMMIT}.")
        sources[name] = (filename, code)
    modules = {}
    for name in (
        "omnivoice.utils.voice_design",
        "omnivoice.utils.lang_map",
        "omnivoice.utils.duration",
        "omnivoice.utils.text",
    ):
        filename, code = sources[name]
        module = SimpleNamespace(__file__=str(filename), __name__=name)
        exec(compile(code, str(filename), "exec"), module.__dict__)
        modules[name] = module
    namespace = {
        "Optional": Optional,
        "Union": Union,
        "re": re,
        "difflib": difflib,
        "logger": logging.getLogger("omnivoice_preview"),
    }
    namespace.update(modules["omnivoice.utils.voice_design"].__dict__)
    namespace.update(modules["omnivoice.utils.lang_map"].__dict__)
    filename, code = sources["omnivoice.models.omnivoice"]
    tree = ast.parse(code)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_resolve_language", "_resolve_instruct"}
    ]
    if len(functions) != 2:
        raise RuntimeError("Pinned language/instruct resolvers are unavailable.")
    for node in functions:
        # The same constants are already supplied above; importing their package
        # would unnecessarily initialize Torch. No other resolver code changes.
        node.body = [statement for statement in node.body if not isinstance(statement, ast.ImportFrom)]
    selected = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(selected, str(filename), "exec"), namespace)
    upstream = SimpleNamespace(
        _resolve_language=namespace["_resolve_language"],
        _resolve_instruct=namespace["_resolve_instruct"],
        _ZH_RE=namespace["_ZH_RE"],
    )
    return upstream, modules["omnivoice.utils.duration"], modules["omnivoice.utils.text"]


def verify_sources() -> tuple[Any, Any, Any]:
    if importlib.metadata.version("transformers") != "5.3.0":
        raise RuntimeError("Requires transformers==5.3.0 for the verified codec mapping.")
    modules = {}
    for name, expected in SOURCE_HASHES.items():
        module = importlib.import_module(name)
        source = Path(module.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != expected:
            raise RuntimeError(f"Upstream source changed: {name}. Install OmniVoice at {SOURCE_COMMIT}.")
        modules[name] = module
    return (
        modules["omnivoice.models.omnivoice"],
        modules["omnivoice.utils.duration"],
        modules["omnivoice.utils.text"],
    )


def prepare_reference(args: argparse.Namespace, upstream: Any) -> tuple[Any, dict[str, Any]]:
    """Read cached tokens or prepare source audio without generating speech."""
    if args.prompt_cache and args.prompt_cache.exists():
        prompt = upstream.VoiceClonePrompt.load(str(args.prompt_cache), map_location="cpu")
        count = int(prompt.ref_audio_tokens.shape[-1])
        if (
            prompt.ref_audio_tokens.ndim != 2
            or prompt.ref_audio_tokens.shape[0] != 8
            or count <= 0
            or not prompt.ref_text
        ):
            raise ValueError("Invalid cached voice prompt shape or transcript.")
        if count / FRAME_RATE > args.max_reference_seconds:
            raise ValueError("Cached reference exceeds --max-reference-seconds.")
        return prompt, {
            "mode": "cached_clone",
            "ref_text": prompt.ref_text,
            "ref_token_count": count,
            "ref_rms": float(prompt.ref_rms),
        }
    if args.ref_audio:
        import numpy as np

        audio = importlib.import_module("omnivoice.utils.audio")
        wav = audio.load_audio(str(args.ref_audio), NATIVE_RATE)
        if wav.size == 0 or not np.isfinite(wav).all():
            raise ValueError("Reference audio must contain finite, nonempty samples.")
        rms = float(np.sqrt(np.mean(wav**2)))
        if 0 < rms < 0.1:
            wav = wav * 0.1 / rms
        if args.preprocess_prompt:
            wav = audio.remove_silence(wav, NATIVE_RATE, mid_sil=200, lead_sil=100, trail_sil=200)
        count = wav.shape[-1] // TOKEN_HOP
        if count <= 0 or count / FRAME_RATE > args.max_reference_seconds:
            raise ValueError(
                "Prepared reference must fit --max-reference-seconds and contain an audio token."
            )
        transcript = args.ref_text
        if args.preprocess_prompt:
            transcript = importlib.import_module("omnivoice.utils.text").add_punctuation(transcript)
        return None, {
            "mode": "new_clone",
            "ref_text": transcript,
            "ref_token_count": int(count),
            "ref_rms": rms,
        }
    return None, {
        "mode": "voice_design" if args.instruct else "auto",
        "ref_text": None,
        "ref_token_count": None,
        "ref_rms": None,
    }


def make_plan(
    text: str,
    args: argparse.Namespace,
    upstream: Any,
    duration_module: Any,
    text_module: Any,
    reference: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    language = upstream._resolve_language(args.language)
    if args.language and args.language.lower() != "none" and language is None:
        raise ValueError(f"Unsupported language: {args.language}")
    if args.normalize_text:
        text = text_module.normalize_text(text, language)
    instruct = upstream._resolve_instruct(args.instruct, use_zh=bool(upstream._ZH_RE.search(text)))
    estimator = duration_module.RuleDurationEstimator()
    ref_text = reference["ref_text"] or "Nice to meet you."
    ref_count = reference["ref_token_count"] if reference["ref_text"] else 25

    def raw(part: str) -> int:
        return max(1, int(estimator.estimate_duration(part, ref_text, ref_count)))

    target_total = max(1, int(args.duration * FRAME_RATE)) if args.duration is not None else None
    effective_speed = raw(text) / target_total if target_total is not None else args.speed

    def planned(part: str) -> int:
        # Match upstream order: divide unrounded estimate, then int().
        estimate = estimator.estimate_duration(part, ref_text, ref_count)
        return max(1, int(estimate / effective_speed))

    maximum = int(args.max_chunk_seconds * FRAME_RATE)
    split_limit = max(1, int(maximum * 0.85)) if target_total is not None else maximum
    chunks = split_bounded(text, planned, split_limit, minimum=min(2 * FRAME_RATE, split_limit // 2))
    lengths = (
        allocate_exact([raw(chunk) for chunk in chunks], target_total)
        if target_total is not None
        else [planned(chunk) for chunk in chunks]
    )
    while any(length > maximum for length in lengths):
        index = next(i for i, length in enumerate(lengths) if length > maximum)
        words = re.findall(r"\[[^\]]+\]|\S+", chunks[index])
        if len(words) < 2:
            raise ValueError("Requested duration cannot fit the configured bounded chunks.")
        middle = len(words) // 2
        chunks[index : index + 1] = [" ".join(words[:middle]), " ".join(words[middle:])]
        lengths = allocate_exact([raw(chunk) for chunk in chunks], target_total)
    pause_samples = round(args.pause_ms * NATIVE_RATE / 1000)
    native_samples = sum(lengths) * TOKEN_HOP + max(0, len(chunks) - 1) * pause_samples
    output_rate = args.sample_rate or NATIVE_RATE
    output_samples = ceil_resampled(native_samples, NATIVE_RATE, output_rate)
    return {
        "normalized_text": text,
        "language": language,
        "instruct": instruct,
        "chunks": [
            {"text": chunk, "target_tokens": length, "native_samples": length * TOKEN_HOP}
            for chunk, length in zip(chunks, lengths)
        ],
        "native_sample_rate": NATIVE_RATE,
        "sample_rate": output_rate,
        "pause_native_samples": pause_samples,
        "predicted_native_samples": native_samples,
        "predicted_export_samples": output_samples,
        "predicted_seconds": output_samples / output_rate,
        "planner_ms": (time.perf_counter() - started) * 1000,
        "effective_speed": effective_speed,
        "interpretation": "Exact final length of reused token allocation; not an unconstrained natural-duration prediction.",
    }


def validate_codec(model: Any, torch: Any) -> None:
    codec = model.audio_tokenizer
    if (
        model.sampling_rate != NATIVE_RATE
        or codec.config.hop_length != TOKEN_HOP
        or codec.config.frame_rate != FRAME_RATE
    ):
        raise RuntimeError("Loaded codec disagrees with the pinned duration mapping.")
    decoder = codec.acoustic_decoder
    convs = [module for module in decoder.modules() if isinstance(module, torch.nn.ConvTranspose1d)]
    if [module.stride[0] for module in convs] != [8, 5, 4, 2, 3]:
        raise RuntimeError("Unexpected codec decoder upsampling strides.")
    for module in convs:
        stride = module.stride[0]
        offset = (
            -stride
            - 2 * module.padding[0]
            + module.dilation[0] * (module.kernel_size[0] - 1)
            + module.output_padding[0]
            + 1
        )
        if offset != 0:
            raise RuntimeError("Codec transposed convolution is not exactly length-multiplying.")
    for module in decoder.modules():
        if isinstance(module, torch.nn.Conv1d):
            if module.stride[0] != 1 or 2 * module.padding[0] != module.dilation[0] * (
                module.kernel_size[0] - 1
            ):
                raise RuntimeError("Codec convolution unexpectedly changes temporal length.")


def memory_snapshot(torch: Any, device: Any, limit_bytes: int) -> dict[str, Any]:
    if device.type == "cpu":
        return {
            "gpu_vram_bytes": 0,
            "torch_peak_allocated_bytes": 0,
            "torch_peak_reserved_bytes": 0,
            "scope": "CPU execution; no CUDA context created by this script",
        }
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    if reserved > limit_bytes:
        raise RuntimeError(f"PyTorch reserved peak exceeded budget: {reserved} > {limit_bytes} bytes.")
    return {
        "torch_peak_allocated_bytes": allocated,
        "torch_peak_reserved_bytes": reserved,
        "scope": "PyTorch allocator only; total driver VRAM peak is not guaranteed",
        "limit_bytes": limit_bytes,
    }


def self_test() -> None:
    for size in (1, 5, 30):
        weights = list(range(1, size + 1))
        for total in (size, size + 1, 999):
            allocated = allocate_exact(weights, total)
            assert sum(allocated) == total and len(allocated) == size and min(allocated) >= 1
    for original, exported in ((24000, 16000), (24000, 44100), (24000, 48000)):
        for samples in (1, 960, 10001, 8_000_000):
            assert ceil_resampled(samples, original, exported) == math.ceil(samples * exported / original)
    text = "A test [B EY1 S] guitar has many beautiful sounds."
    chunks = split_bounded(text, lambda value: len(value), 24)
    assert " ".join(chunks) == text and any("[B EY1 S]" in chunk for chunk in chunks)
    assert all(len(chunk) <= 24 for chunk in chunks)
    for count in (10, 50, 500):
        test_text = " ".join("word" for _ in range(count))
        test_chunks = split_bounded(test_text, lambda value: len(value.split()) * 4, 40)
        assert " ".join(test_chunks) == test_text and len(test_chunks) == (count + 9) // 10
    print(
        "Self-test passed: exact frame allocation, resampling counts, control-preserving bounded chunking (10–500 words)."
    )


def main() -> int:
    cold_started = time.perf_counter()
    args = parse_arguments()
    if args.self_test:
        self_test()
        return 0
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8-sig")
    text = text.strip()
    if not text or len(re.findall(r"\S+", text)) > 500:
        raise ValueError("Provide nonempty text of at most 500 whitespace-delimited words.")
    try:
        upstream, duration_module, text_module = (
            lightweight_preflight()
            if args.ref_audio is None and args.prompt_cache is None
            else verify_sources()
        )
    except ImportError as error:
        raise RuntimeError(
            "Missing dependencies. Follow the pinned installation commands in this file's docstring."
        ) from error
    prompt, reference = prepare_reference(args, upstream)
    plan = make_plan(text, args, upstream, duration_module, text_module, reference)
    plan["cold_preflight_ms"] = (time.perf_counter() - cold_started) * 1000
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    manifest = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_commit": SOURCE_COMMIT,
        "configuration": configuration,
        "reference": reference,
        "plan": plan,
        "samples": [],
        "status": "estimated",
        "live_inference_verified": False,
    }
    atomic_json(args.output_dir / "estimate.json", manifest)
    print(
        json.dumps(
            {
                "predicted_seconds": plan["predicted_seconds"],
                "predicted_samples": plan["predicted_export_samples"],
                "sample_rate": plan["sample_rate"],
                "chunks": len(plan["chunks"]),
                "planner_ms": plan["planner_ms"],
                "cold_preflight_ms": plan["cold_preflight_ms"],
            },
            indent=2,
        )
    )
    if args.estimate_only:
        return 0

    upstream, _, _ = verify_sources()
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from scipy.signal import resample_poly
    from tqdm import tqdm

    device = torch.device(args.device)
    limit_bytes = int(args.max_vram_gb * 1_000_000_000)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu.")
        torch.cuda.set_device(device)
        # Context/library peak cannot be capped here; leave a conservative reserve.
        budget = limit_bytes - 384 * 1024 * 1024
        if budget <= 0:
            raise ValueError("CUDA budget must exceed the context reserve.")
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(budget / total_bytes, 1.0), device)
        torch.cuda.reset_peak_memory_stats(device)
    dtype_name = args.dtype if args.dtype != "auto" else ("float32" if device.type == "cpu" else "float16")
    dtype = getattr(torch, dtype_name)
    loading_started = time.perf_counter()
    snapshot = (
        str(args.local_model)
        if args.local_model
        else snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=args.local_files_only)
    )
    if not (Path(snapshot) / "audio_tokenizer" / "config.json").is_file():
        raise RuntimeError(
            "Snapshot must include official audio_tokenizer; unpinned external fallback is disabled."
        )
    model = upstream.OmniVoice.from_pretrained(
        snapshot, device_map=str(device), dtype=dtype, load_asr=False, attn_implementation="sdpa"
    )
    model.eval()
    model.audio_tokenizer.eval()
    validate_codec(model, torch)
    manifest["model_loading_ms"] = (time.perf_counter() - loading_started) * 1000
    manifest["dtype"] = dtype_name
    if reference["mode"] == "new_clone":
        with torch.inference_mode():
            prompt = model.create_voice_clone_prompt(
                str(args.ref_audio), ref_text=args.ref_text, preprocess_prompt=args.preprocess_prompt
            )
        if (
            prompt.ref_text != reference["ref_text"]
            or prompt.ref_audio_tokens.shape[-1] != reference["ref_token_count"]
        ):
            raise RuntimeError(
                "Actual reference tokenization disagrees with preflight; no speech was generated."
            )
        if args.prompt_cache:
            args.prompt_cache.parent.mkdir(parents=True, exist_ok=True)
            prompt.save(str(args.prompt_cache))
    manifest["memory_after_loading"] = memory_snapshot(torch, device, limit_bytes)
    generation_config = upstream.OmniVoiceGenerationConfig(
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        layer_penalty_factor=args.layer_penalty_factor,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
        denoise=args.denoise,
        preprocess_prompt=False,
        postprocess_output=False,
        pad_duration=0.0,
        fade_duration=args.fade_ms / 1000,
        audio_chunk_duration=0.0,
        audio_chunk_threshold=float("inf"),
    )
    outputs = [args.output_dir / f"sample_{i + 1:03d}_seed_{args.seed + i}.wav" for i in range(args.n)]
    if any(path.exists() for path in outputs):
        raise FileExistsError("A sample output already exists. Choose a fresh --output-dir.")
    count = 0
    progress_total = args.n * (len(plan["chunks"]) * (args.num_step + 1) + 1)
    manifest["status"] = "generating"
    atomic_json(args.output_dir / "manifest.json", manifest)
    with tqdm(total=progress_total, desc="OmniVoice: inference / decode / export", unit="unit") as bar:

        def step_hook(_module: Any, _inputs: Any, _output: Any) -> None:
            nonlocal count
            count += 1
            bar.update(1)

        hook = model.register_forward_hook(step_hook)
        try:
            for i, destination in enumerate(outputs):
                seed = args.seed + i
                torch.manual_seed(seed)
                np.random.seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                started = time.perf_counter()
                wave_parts = []
                first_tokens = None
                for chunk_index, chunk in enumerate(plan["chunks"]):
                    ref_tokens = prompt.ref_audio_tokens if prompt else first_tokens
                    ref_text = (
                        prompt.ref_text
                        if prompt
                        else (plan["chunks"][0]["text"] if first_tokens is not None else None)
                    )
                    ref_rms = prompt.ref_rms if prompt else None
                    task = upstream.GenerationTask(
                        batch_size=1,
                        texts=[chunk["text"]],
                        target_lens=[chunk["target_tokens"]],
                        langs=[plan["language"]],
                        instructs=[plan["instruct"]],
                        ref_texts=[ref_text],
                        ref_audio_tokens=[ref_tokens],
                        ref_rms=[ref_rms],
                        speed=[plan["effective_speed"]],
                    )
                    before_steps = count
                    with torch.inference_mode():
                        tokens = model._generate_iterative(task, generation_config)[0]
                        if tuple(tokens.shape) != (8, chunk["target_tokens"]):
                            raise RuntimeError("Generated token shape disagrees with frozen duration plan.")
                        waveform = model._decode_and_post_process(tokens, ref_rms, generation_config)
                    if count - before_steps != args.num_step:
                        raise RuntimeError("Unexpected upstream denoising-step count.")
                    if (
                        waveform.ndim != 1
                        or waveform.size != chunk["native_samples"]
                        or not np.isfinite(waveform).all()
                    ):
                        raise RuntimeError(
                            "Decoded waveform disagrees with plan or contains nonfinite values."
                        )
                    if prompt is None and first_tokens is None:
                        first_tokens = tokens.detach().cpu()
                    wave_parts.append(waveform.astype(np.float32, copy=False))
                    if chunk_index + 1 < len(plan["chunks"]) and plan["pause_native_samples"]:
                        wave_parts.append(np.zeros(plan["pause_native_samples"], dtype=np.float32))
                    del tokens, waveform, task
                    memory_snapshot(torch, device, limit_bytes)
                    bar.update(1)
                waveform = np.concatenate(wave_parts)
                if waveform.size != plan["predicted_native_samples"]:
                    raise RuntimeError("Concatenation length disagrees with preflight.")
                if plan["sample_rate"] != NATIVE_RATE:
                    divisor = math.gcd(NATIVE_RATE, plan["sample_rate"])
                    waveform = resample_poly(
                        waveform, plan["sample_rate"] // divisor, NATIVE_RATE // divisor
                    ).astype(np.float32)
                if waveform.size != plan["predicted_export_samples"]:
                    raise RuntimeError("Resampling output length disagrees with rational mapping.")
                waveform, peak_normalization = limit_peak(waveform, args.peak_limit, np)
                temporary = destination.with_name(destination.stem + ".part.wav")
                sf.write(str(temporary), waveform, plan["sample_rate"], subtype="FLOAT")
                info = sf.info(str(temporary))
                if info.frames != plan["predicted_export_samples"] or info.samplerate != plan["sample_rate"]:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError("WAV metadata does not match duration plan.")
                error = abs(info.frames - plan["predicted_export_samples"]) / max(info.frames, 1)
                if error >= 0.01:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError("Final exported duration error is >=1%.")
                os.replace(temporary, destination)
                manifest["samples"].append(
                    {
                        "file": str(destination.resolve()),
                        "seed": seed,
                        "peak_normalization": peak_normalization,
                        "predicted_samples": plan["predicted_export_samples"],
                        "actual_samples": info.frames,
                        "predicted_seconds": plan["predicted_seconds"],
                        "actual_seconds": info.frames / info.samplerate,
                        "relative_duration_error": error,
                        "generation_seconds": time.perf_counter() - started,
                        "memory": memory_snapshot(torch, device, limit_bytes),
                    }
                )
                manifest["live_inference_verified"] = True
                atomic_json(args.output_dir / "manifest.json", manifest)
                bar.update(1)
        except BaseException as error:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(error).__name__}: {error}"
            atomic_json(args.output_dir / "manifest.json", manifest)
            raise
        finally:
            hook.remove()
    manifest["status"] = "completed"
    manifest["memory"] = memory_snapshot(torch, device, limit_bytes)
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(f"Saved {args.n} verified sample(s) and manifest.json to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

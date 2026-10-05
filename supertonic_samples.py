#!/usr/bin/env python3
"""Standalone Supertonic 3 sampler with a duration-only, reusable inference plan.

Python 3.11+; install numpy, onnxruntime==1.23.1, soundfile, scipy for CPU.
For CUDA install onnxruntime-gpu[cuda,cudnn]==1.23.2 instead of onnxruntime;
use the matching CUDA 12.8+ and cuDNN 9 runtimes in an isolated environment.
Download the complete open ONNX weights before running:
  hf download supertone-oss-archive/supertonic-3 \
    --revision aafc6e32416a594460b32413efc49d7fe4ce6d46 --local-dir supertonic_assets
Clone https://github.com/supertone-oss-archive/supertonic.git and check out
1e9799e964ea4c0dad7cde993b65c3c813a7b373 (the text frontend is imported).

Example:
  python supertonic_samples.py --repo /path/to/supertonic \
    --assets-dir /path/to/supertonic_assets --voice M2 --lang en \
    --text "A sufficiently long demonstration sentence to test the speech duration." --n 3

CPU is the default and allocates zero GPU VRAM. Optional --device cuda:0 uses
four ONNX Runtime CUDA sessions with conservative per-session arena caps whose
sum is below --max-vram-gb minus --cuda-reserve-mb. These limits cover arenas,
not every driver/context/workspace allocation. CUDA therefore requires explicit
--allow-unverified-cuda-budget; measure total VRAM externally, as the Kaggle
notebook does. A missing or failed CUDA provider is an error; unsupported graph
operators may run on the explicitly configured CPU provider. A preview
runs the duration predictor, not the text encoder, flow model, or vocoder. Its
planned sample count includes the official SDK's removal of allocated latent
padding, explicit pauses, and deterministic resampling. Speech is never stretched
or padded to make an estimate correct. Models and sessions must be loaded first;
cold startup and preview latency are reported separately. Preview latency is
measured, not promised to be below a particular time on every CPU.
M2 is the official deep, calm, serious male preset. This open checkpoint has no
reference-WAV cloning encoder; custom voices require an already compatible style
JSON. Punctuation-aware chunks and short edge fades reduce mechanical resets and
joining clicks; the fades and peak protection preserve every planned sample.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_REVISION = "1e9799e964ea4c0dad7cde993b65c3c813a7b373"
WEIGHTS_REVISION = "aafc6e32416a594460b32413efc49d7fe4ce6d46"
WEIGHT_HASHES = {
    "duration_predictor.onnx": "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
    "text_encoder.onnx": "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
    "vector_estimator.onnx": "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c",
    "vocoder.onnx": "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, help="JSON defaults; CLI values override them; keys use underscores"
    )
    text = parser.add_mutually_exclusive_group()
    text.add_argument("--text")
    text.add_argument("--text-file", type=Path)
    parser.add_argument("--repo", type=Path, help="Pinned official Supertonic source checkout")
    parser.add_argument(
        "--assets-dir", type=Path, help="Downloaded snapshot containing onnx/ and voice_styles/"
    )
    parser.add_argument("--voice", default="M2", help="Preset basename; M2 is deep/serious male")
    parser.add_argument(
        "--voice-style", type=Path, help="Explicit compatible voice-style JSON instead of --voice"
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Higher is faster; modifies the cached duration plan"
    )
    parser.add_argument("--steps", type=int, default=16, help="Flow generation steps; duration is unaffected")
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=300,
        help="Bounded sequential chunks; Korean/Japanese default is 120",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]; respects CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "--allow-unverified-cuda-budget",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Acknowledge that the ORT arena cap cannot guarantee a whole-process VRAM ceiling",
    )
    parser.add_argument(
        "--cuda-reserve-mb",
        type=int,
        default=768,
        help="Decimal MB reserved outside the four CUDA arenas",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("supertonic_samples"))
    parser.add_argument(
        "--sample-rate", type=int, default=0, help="Export sample rate; 0 uses native 44100 Hz"
    )
    parser.add_argument("--pause-ms", type=float, default=0.0, help="Explicit silence between chunks only")
    parser.add_argument(
        "--edge-fade-ms",
        type=float,
        default=3.0,
        help="Cosine fade at each chunk edge; preserves sample count; 0 disables it",
    )
    parser.add_argument("--gain-db", type=float, default=0.0)
    parser.add_argument(
        "--normalize-peak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce peaks above 0.98 to avoid PCM clipping; preserves sample count",
    )
    parser.add_argument(
        "--max-vram-gb", type=float, default=5.0, help="Decimal GB, at most 5; CUDA arena budget target"
    )
    parser.add_argument("--estimate-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wav-subtype", choices=["PCM_16", "PCM_24", "FLOAT"], default="PCM_16")
    early, _ = parser.parse_known_args(argv)
    if early.config:
        defaults = json.loads(early.config.read_text(encoding="utf-8"))
        if not isinstance(defaults, dict):
            parser.error("--config must contain one JSON object")
        actions = {action.dest: action for action in parser._actions}
        unknown = set(defaults) - set(actions)
        if unknown:
            parser.error(f"Unknown JSON keys: {sorted(unknown)}")
        for key, value in defaults.items():
            if key in {"help", "config"}:
                parser.error(f"JSON key {key!r} is not a setting")
            action = actions[key]
            if value is None and action.default is not None:
                parser.error(f"JSON value for {key} cannot be null")
            if value is not None:
                if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
                    valid = type(value) is bool
                elif action.type is int:
                    valid = type(value) is int
                elif action.type is float:
                    valid = type(value) in {int, float}
                else:
                    valid = isinstance(value, str)
                if not valid:
                    parser.error(f"Invalid JSON type for {key}")
            if action.type and value is not None:
                try:
                    defaults[key] = action.type(value)
                except (ValueError, TypeError) as exc:
                    parser.error(f"Invalid JSON value for {key}: {exc}")
        if early.text is not None:
            defaults["text_file"] = None
        elif early.text_file is not None:
            defaults["text"] = None
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if (args.text is None) == (args.text_file is None):
        parser.error("Specify exactly one of --text or --text-file, including JSON defaults")
    if args.repo is None or args.assets_dir is None:
        parser.error("--repo and --assets-dir are required")
    for name in ("n", "steps", "threads", "max_chunk_chars"):
        if not isinstance(getattr(args, name), int) or getattr(args, name) < 1:
            parser.error(f"{name} must be a positive integer")
    if args.max_chunk_chars < 20:
        parser.error("--max-chunk-chars must be at least 20")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not math.isfinite(args.speed) or not 0.25 <= args.speed <= 4.0:
        parser.error("--speed must be between 0.25 and 4 (extreme speeds may impair quality)")
    if not math.isfinite(args.pause_ms) or args.pause_ms < 0:
        parser.error("--pause-ms must be finite and nonnegative")
    if not math.isfinite(args.edge_fade_ms) or not 0 <= args.edge_fade_ms <= 20:
        parser.error("--edge-fade-ms must be between 0 and 20")
    if not math.isfinite(args.gain_db) or abs(args.gain_db) > 120:
        parser.error("--gain-db must be finite and between -120 and 120")
    if not math.isfinite(args.max_vram_gb) or not 0 < args.max_vram_gb <= 5:
        parser.error("--max-vram-gb must be in (0, 5] decimal GB")
    if args.device != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", args.device):
        parser.error("--device must be cpu or cuda[:index]")
    if args.device != "cpu" and not args.allow_unverified_cuda_budget:
        parser.error(
            "CUDA arena limits cannot guarantee total peak VRAM <=5 GB; explicitly "
            "--allow-unverified-cuda-budget and monitor total VRAM, or use cpu"
        )
    if args.cuda_reserve_mb < 0:
        parser.error("--cuda-reserve-mb must be nonnegative")
    if args.device != "cpu" and args.cuda_reserve_mb * 1_000_000 >= args.max_vram_gb * 1_000_000_000:
        parser.error("--cuda-reserve-mb must leave a positive CUDA arena budget")
    if args.sample_rate and not 8000 <= args.sample_rate <= 192000:
        parser.error("--sample-rate must be 0 or 8000..192000 Hz")
    for action in parser._actions:
        value = getattr(args, action.dest, None)
        if action.choices is not None and value not in action.choices:
            parser.error(f"{action.dest} must be one of {action.choices}")
        if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)) and not isinstance(
            value, bool
        ):
            parser.error(f"{action.dest} must be a JSON boolean")
    return args


def boundary_strength(word: str) -> int:
    """Punctuation preference, avoiding common English abbreviations/initials."""
    stripped = word.rstrip("\"\u201d\u2019')]}")
    if re.search(r"[!?\u2026\u3002\uff01\uff1f]+$", stripped):
        return 2
    if stripped.endswith("."):
        abbreviation = stripped.lower() in {
            "mr.",
            "mrs.",
            "ms.",
            "dr.",
            "prof.",
            "sr.",
            "jr.",
            "st.",
            "vs.",
            "etc.",
            "e.g.",
            "i.e.",
            "a.m.",
            "p.m.",
        }
        initial = bool(re.fullmatch(r"(?:[A-Za-z]\.)+", stripped))
        if not abbreviation and not initial:
            return 2
    return 1 if re.search(r"[,;:\u2014\u2013\uff0c\uff1b\uff1a]+$", stripped) else 0


def chunk_text(text: str, limit: int) -> list[str]:
    """Keep strict character bounds while preferring complete sentences/clauses."""
    if limit < 1:
        raise ValueError("Chunk character limit must be positive")
    if not text.strip():
        raise ValueError("The text is empty")
    chunks: list[str] = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if any(len(word) > limit for word in words):
            raise ValueError(f"A token exceeds the chunk limit {limit}")
        while words:
            end, length = 0, 0
            for word in words:
                candidate = length + len(word) + bool(end)
                if candidate > limit:
                    break
                length, end = candidate, end + 1
            if end < len(words):
                candidates = [
                    index + 1 for index, word in enumerate(words[:end]) if boundary_strength(word) == 2
                ]
                if not candidates:
                    candidates = [
                        index + 1 for index, word in enumerate(words[:end]) if boundary_strength(word) == 1
                    ]
                if candidates:
                    end = candidates[-1]
            chunks.append(" ".join(words[:end]))
            words = words[end:]
    return chunks


def edge_fade(audio: Any, sample_rate: int, milliseconds: float) -> Any:
    """Suppress hard chunk joins without trimming, stretching, or padding."""
    import numpy as np

    result = np.asarray(audio, dtype=np.float32).copy()
    count = min(round(sample_rate * milliseconds / 1000), len(result) // 2)
    if count > 1:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, count))).astype(np.float32)
        result[:count] *= ramp
        result[-count:] *= ramp[::-1]
    return result


def protect_peak(audio: Any, gain_db: float, normalize_peak: bool) -> tuple[Any, dict[str, float]]:
    """Apply one global gain; attenuate overshoots rather than clip them."""
    import numpy as np

    result = np.asarray(audio, dtype=np.float32) * (10 ** (gain_db / 20))
    peak_before = float(np.max(np.abs(result)))
    scale = min(1.0, 0.98 / peak_before) if normalize_peak and peak_before else 1.0
    result *= scale
    return result, {
        "peak_before_protection": peak_before,
        "peak_scale": scale,
        "peak_after_protection": float(np.max(np.abs(result))),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frontend(repo: Path) -> Any:
    command = ["git", "-C", str(repo), "rev-parse", "HEAD"]
    revision = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(
            f"Expected source revision {SOURCE_REVISION}; got {revision}. Check out the pinned revision."
        )
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", "py/helper.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty.strip():
        raise RuntimeError(
            "py/helper.py has local changes; the verified inference contract requires the pinned frontend"
        )
    spec = importlib.util.spec_from_file_location("supertonic_frontend", repo / "py" / "helper.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load official Supertonic frontend")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class ChunkPlan:
    text: str
    text_ids: Any
    text_mask: Any
    duration: Any
    native_samples: int
    latent_frames: int


def execution_providers(args: argparse.Namespace, session_count: int) -> list[Any]:
    """Give independent ORT sessions disjoint portions of the aggregate arena budget.

    CUDA device IDs are visible ordinals; a process masked to one GPU uses cuda:0.
    gpu_mem_limit is an arena limit, not a complete device-memory usage limit.
    """
    if args.device == "cpu":
        return ["CPUExecutionProvider"]
    if session_count < 1:
        raise ValueError("At least one ONNX session is required")
    device_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
    budget = int(args.max_vram_gb * 1_000_000_000) - args.cuda_reserve_mb * 1_000_000
    arena_limit = budget // session_count
    if arena_limit < 1:
        raise ValueError("The requested budget leaves no room for CUDA session arenas")
    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": device_index,
                "gpu_mem_limit": arena_limit,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "cudnn_conv_use_max_workspace": "0",
                "do_copy_in_default_stream": "1",
                "use_tf32": "0",
            },
        ),
        "CPUExecutionProvider",
    ]


def validate_session_providers(session: Any, device: str, graph: str) -> None:
    active = session.get_providers()
    expected = (
        ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    if active != expected:
        message = f"Unexpected execution providers for {graph}: {active}; expected {expected}."
        if device != "cpu":
            message += " CUDA was requested, so a failed CUDA provider cannot silently fall back to CPU."
        raise RuntimeError(message)


class Engine:
    """One resident model; plans never call the waveform-generating graphs."""

    def __init__(self, args: argparse.Namespace):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.args = args
        self.providers = execution_providers(args, len(WEIGHT_HASHES))
        if args.device != "cpu":
            # Prefer explicitly installed NVIDIA runtime packages rather than an
            # older CUDA runtime bundled by an unrelated notebook environment.
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(directory="")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(
                    "CUDAExecutionProvider is unavailable. Install onnxruntime-gpu[cuda,cudnn]==1.23.2 "
                    "with CUDA 12.8+ and cuDNN 9; do not install the CPU and GPU wheels together."
                )
        self.assets = args.assets_dir.resolve()
        onnx_dir = self.assets / "onnx"
        self.cfg = json.loads((onnx_dir / "tts.json").read_text(encoding="utf-8"))
        self.sample_rate = int(self.cfg["ae"]["sample_rate"])
        self.chunk_size = int(self.cfg["ae"]["base_chunk_size"]) * int(
            self.cfg["ttl"]["chunk_compress_factor"]
        )
        self.latent_dim = int(self.cfg["ttl"]["latent_dim"]) * int(self.cfg["ttl"]["chunk_compress_factor"])
        if (self.sample_rate, self.chunk_size, self.latent_dim) != (44100, 3072, 144):
            raise RuntimeError(
                "This adapter requires the pinned Supertonic 3 configuration (44100, 3072, 144)"
            )
        self.hashes = {name: sha256(onnx_dir / name) for name in WEIGHT_HASHES}
        if self.hashes != WEIGHT_HASHES:
            raise RuntimeError("ONNX hashes do not match the pinned complete Supertonic 3 weights")
        frontend = load_frontend(args.repo.resolve())
        self.processor = frontend.UnicodeProcessor(str(onnx_dir / "unicode_indexer.json"))
        style_path = args.voice_style or self.assets / "voice_styles" / f"{args.voice}.json"
        self.style_path = style_path.resolve()
        self.style = frontend.load_voice_style([str(self.style_path)])
        if args.lang not in frontend.AVAILABLE_LANGS:
            raise ValueError(f"Unsupported language {args.lang!r}; choices: {frontend.AVAILABLE_LANGS}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = args.threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sessions = {
            name: ort.InferenceSession(str(onnx_dir / name), sess_options=opts, providers=self.providers)
            for name in WEIGHT_HASHES
        }
        for name, session in self.sessions.items():
            validate_session_providers(session, args.device, name)
            # Disable ORT's automatic provider retry on subsequent execution errors.
            # Individual CPU-assigned operators remain part of the explicit provider list.
            session.disable_fallback()
        self.validate_decoder()
        # Warm only the duration graph: initialization is excluded from preview latency.
        ids, mask = self.processor(["Warm up."], [args.lang])
        self.sessions["duration_predictor.onnx"].run(
            None, {"text_ids": ids, "style_dp": self.style.dp, "text_mask": mask}
        )

    def memory_report(self) -> dict[str, Any]:
        if self.args.device == "cpu":
            return {
                "peak_gpu_vram_bytes": 0,
                "gpu_budget_guarantee": "CPUExecutionProvider only: zero GPU allocations",
                "providers": self.providers,
            }
        cuda_options = self.providers[0][1]
        return {
            "peak_gpu_vram_bytes": None,
            "gpu_budget_guarantee": "No whole-process guarantee; monitor total process VRAM externally",
            "cuda_arena_limit_per_session_bytes": cuda_options["gpu_mem_limit"],
            "cuda_arena_limit_aggregate_bytes": cuda_options["gpu_mem_limit"] * len(self.sessions),
            "cuda_overhead_reserve_bytes": self.args.cuda_reserve_mb * 1_000_000,
            "requested_total_budget_bytes": int(self.args.max_vram_gb * 1_000_000_000),
            "providers": self.providers,
            "provider_options_by_graph": {
                name: session.get_provider_options() for name, session in self.sessions.items()
            },
            "cpu_operator_fallback": "Explicit CPU provider allowed for unsupported graph operators",
            "vram_measurement": "ORT exposes arena limits, not a whole-process peak; use notebook NVML telemetry",
        }

    def validate_decoder(self) -> None:
        """Reject incompatible decoder-length contracts before a text preview is offered."""
        for frames in (2, 3):
            latent = self.np.zeros((1, self.latent_dim, frames), dtype=self.np.float32)
            wav = self.sessions["vocoder.onnx"].run(None, {"latent": latent})[0]
            if wav.shape != (1, frames * self.chunk_size):
                raise RuntimeError(
                    f"Vocoder length contract changed: {wav.shape}; expected {(1, frames * self.chunk_size)}"
                )

    def plan(self, text: str) -> list[ChunkPlan]:
        limit = self.args.max_chunk_chars
        if self.args.lang in {"ko", "ja"} and limit == 300:
            limit = 120
        plans: list[ChunkPlan] = []
        for chunk in chunk_text(text, limit):
            ids, mask = self.processor([chunk], [self.args.lang])
            dur = self.sessions["duration_predictor.onnx"].run(
                None, {"text_ids": ids, "style_dp": self.style.dp, "text_mask": mask}
            )[0]
            dur = self.np.asarray(dur) / self.args.speed
            if dur.shape != (1,) or not self.np.isfinite(dur).all() or dur[0] <= 0:
                raise RuntimeError(f"Invalid predicted duration: {dur!r}")
            # These are exactly the SDK's float32 duration->sample and latent-allocation operations.
            samples_array = (dur * self.sample_rate).astype(self.np.int64)
            samples = int(samples_array[0])
            frames = int(
                ((dur.max() * self.sample_rate + self.chunk_size - 1) / self.chunk_size).astype(self.np.int32)
            )
            mask_frames = (samples + self.chunk_size - 1) // self.chunk_size
            if samples < 1 or frames != mask_frames:
                raise RuntimeError(
                    "Duration-to-latent allocation is inconsistent; refusing an inaccurate preview"
                )
            plans.append(ChunkPlan(chunk, ids, mask, dur.copy(), samples, frames))
        return plans

    def synthesize(self, plan: ChunkPlan, rng: Any, progress: Any) -> Any:
        np = self.np
        embedding = self.sessions["text_encoder.onnx"].run(
            None, {"text_ids": plan.text_ids, "style_ttl": self.style.ttl, "text_mask": plan.text_mask}
        )[0]
        progress()
        latent = rng.standard_normal((1, self.latent_dim, plan.latent_frames)).astype(np.float32)
        mask = np.ones((1, 1, plan.latent_frames), dtype=np.float32)
        total = np.array([self.args.steps], dtype=np.float32)
        for step in range(self.args.steps):
            latent = self.sessions["vector_estimator.onnx"].run(
                None,
                {
                    "noisy_latent": latent,
                    "text_emb": embedding,
                    "style_ttl": self.style.ttl,
                    "text_mask": plan.text_mask,
                    "latent_mask": mask,
                    "current_step": np.array([step], dtype=np.float32),
                    "total_step": total,
                },
            )[0]
            progress()
        wav = self.sessions["vocoder.onnx"].run(None, {"latent": latent})[0]
        expected_allocation = plan.latent_frames * self.chunk_size
        if wav.shape != (1, expected_allocation):
            raise RuntimeError(
                f"Vocoder output length {wav.shape} violates the pre-generation allocation {expected_allocation}"
            )
        progress()
        # Official model export: remove only the predetermined final latent allocation padding.
        # This is independent of audio content; there is no silence detection or target-length fitting.
        return edge_fade(wav[0, : plan.native_samples], self.sample_rate, self.args.edge_fade_ms)


def predicted_export_samples(native_samples: int, native_rate: int, export_rate: int) -> int:
    return (native_samples * export_rate + native_rate - 1) // native_rate


def resample_audio(wav: Any, native_rate: int, export_rate: int) -> Any:
    if native_rate == export_rate:
        return wav
    from scipy.signal import resample_poly

    divisor = math.gcd(native_rate, export_rate)
    return resample_poly(wav, export_rate // divisor, native_rate // divisor).astype("float32", copy=False)


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8-sig")
    text = text.strip()
    if not text:
        raise ValueError("The input text is empty")
    words = len(re.findall(r"\S+", text))
    if not 10 <= words <= 500:
        raise ValueError(f"Expected the requested 10..500 whitespace-delimited words; got {words}")
    import numpy as np
    import soundfile as sf

    # Resolve optional export dependencies before initialization or expensive generation.
    if args.sample_rate and args.sample_rate != 44100:
        import scipy.signal  # noqa: F401
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "supertonic_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Output already exists: {manifest_path}. Choose a fresh --output-dir")
    startup_start = time.perf_counter()
    engine = Engine(args)
    startup_s = time.perf_counter() - startup_start
    plan_start = time.perf_counter()
    plans = engine.plan(text)
    preview_s = time.perf_counter() - plan_start
    native_rate = engine.sample_rate
    export_rate = args.sample_rate or native_rate
    pause_samples = round(args.pause_ms * native_rate / 1000)
    native_samples = sum(plan.native_samples for plan in plans) + pause_samples * (len(plans) - 1)
    expected_samples = predicted_export_samples(native_samples, native_rate, export_rate)
    manifest: dict[str, Any] = {
        "model": "Supertonic 3",
        "source_revision": SOURCE_REVISION,
        "weights_revision": WEIGHTS_REVISION,
        "weight_sha256": engine.hashes,
        "voice_style_sha256": sha256(engine.style_path),
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "text": text,
        "word_count": words,
        "native_rate": native_rate,
        "export_rate": export_rate,
        "startup_seconds": startup_s,
        "warm_preview_seconds": preview_s,
        "predicted_samples": expected_samples,
        "predicted_seconds": expected_samples / export_rate,
        "pause_native_samples": pause_samples,
        "duration_method": "Cached duration-predictor plan, official allocation-padding removal, no content-based trimming",
        **engine.memory_report(),
        "chunks": [
            {
                "text": p.text,
                "native_samples": p.native_samples,
                "latent_frames": p.latent_frames,
                "allocated_samples": p.latent_frames * engine.chunk_size,
                "removed_allocation_padding_samples": p.latent_frames * engine.chunk_size - p.native_samples,
            }
            for p in plans
        ],
        "runs": [],
    }
    print(
        f"Duration preview: {expected_samples / export_rate:.6f} s ({expected_samples} exported samples)",
        flush=True,
    )
    print(
        f"Warm preview: {preview_s * 1000:.2f} ms; cold startup: {startup_s:.3f} s; chunks: {len(plans)}",
        flush=True,
    )
    write_json(manifest_path, manifest)
    if args.estimate_only:
        return 0
    units_total = args.n * len(plans) * (args.steps + 2)
    units_done = 0

    def progress() -> None:
        nonlocal units_done
        units_done += 1
        fraction = units_done / units_total
        bar = "#" * int(30 * fraction) + "-" * (30 - int(30 * fraction))
        print(
            f"\rGeneration [{bar}] {100 * fraction:6.2f}% ({units_done}/{units_total} stages)",
            end="",
            flush=True,
        )

    for index in range(args.n):
        seed = args.seed + index
        if seed < 0:
            raise ValueError("Seeds must be nonnegative")
        path = args.output_dir / f"supertonic_{index + 1:03d}_seed{seed}.wav"
        if path.exists():
            raise FileExistsError(path)
        started = time.perf_counter()
        pieces = []
        rng = np.random.default_rng(seed)
        for chunk_index, plan in enumerate(plans):
            if chunk_index and pause_samples:
                pieces.append(np.zeros(pause_samples, dtype=np.float32))
            pieces.append(engine.synthesize(plan, rng, progress))
        audio = resample_audio(np.concatenate(pieces), native_rate, export_rate)
        audio, output_levels = protect_peak(audio, args.gain_db, args.normalize_peak)
        error = abs(len(audio) - expected_samples) / len(audio)
        if not np.isfinite(audio).all() or len(audio) != expected_samples or error >= 0.01:
            raise RuntimeError(
                f"Generation failed duration/finite-audio validation: predicted={expected_samples}, actual={len(audio)}, error={error:.8%}"
            )
        temporary = path.with_suffix(".tmp.wav")
        sf.write(temporary, audio, export_rate, subtype=args.wav_subtype)
        info = sf.info(temporary)
        if (info.frames, info.samplerate, info.channels) != (expected_samples, export_rate, 1):
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Exported WAV header violates the duration plan")
        os.replace(temporary, path)
        manifest["runs"].append(
            {
                "seed": seed,
                "file": str(path.resolve()),
                "generation_seconds": time.perf_counter() - started,
                "actual_samples": info.frames,
                "actual_seconds": info.frames / export_rate,
                "output_levels": output_levels,
                "duration_error_fraction": error,
                **engine.memory_report(),
            }
        )
        write_json(manifest_path, manifest)
    print(f"\nSaved {args.n} samples and {manifest_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

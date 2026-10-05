#!/usr/bin/env python3
"""Standalone F5-TTS v1 samples with an exact, reused duration plan.

Python 3.10-3.12 recommended. Install matching torch/torchaudio, scipy,
soundfile, tqdm, safetensors, huggingface_hub, vocos==0.1.0 and the requirements
of the official F5 repository. Clone its source and check out SOURCE_REVISION:
  git clone https://github.com/SWivid/F5-TTS.git F5-TTS
  git -C F5-TTS checkout 283252563dbf91be625e0c27926acfaac449186c
  python -m pip install -e ./F5-TTS
  python f5_samples.py --repo ./F5-TTS --text-file passage.txt \
      --reference-audio reference.wav --reference-text "Exact transcript." --n 3

F5 generator weights: CC-BY-NC-4.0 (official pretrained checkpoint).
Native rate is 24 kHz; --sample-rate resamples the exported WAV.
Reference/style controls are supported. There is no independent emotion or
word-emphasis API in this adapter. Speed is duration allocation, not an
assurance of identical perceptual tempo for every voice/text.

The reference WAV is used as supplied (mono/resampled, without silence
trimming); provide a clean 1-8 second reference and its exact transcript.
Chunks prefer sentence/clause boundaries and keep tiny tails with context.
Short chunks use the requested speed rather than an implicit 0.3 slowdown.
Short edge fades suppress discontinuities without changing any sample counts.
The preview is an exact *allocated output length*, not a learned assessment
of naturally appropriate speaking time. F5 still has to fit the text into
that allocation. Planning does not run the flow model or vocoder. Cold
imports, downloads and reference preprocessing are timed separately.

CPU is the default and uses zero GPU VRAM. CUDA's PyTorch allocator is capped
below --max-vram-gb, with a reserve for nonallocator allocations. That cap is
not a hard bound on total process VRAM: CUDA libraries allocate separately.
Measured peaks and optional sampled NVML process usage are reported; a GPU
run exceeding the observed budget fails. CPU is the guaranteed GPU-safe mode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

SOURCE_REVISION = "283252563dbf91be625e0c27926acfaac449186c"
WEIGHT_REVISION = "84e5a410d9cead4de2f847e7c9369a6440bdfaca"
VOCOS_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
NATIVE_RATE, HOP = 24000, 256
SOURCE_HASHES = {
    "src/f5_tts/model/cfm.py": "88cea5f87b78a39dce292910ccac4b9e59d62171f245e2bf6a17c09426a35208",
    "src/f5_tts/model/modules.py": "92a1d97756bc7773a75c20e0e2fc913b3220f2ae2e6f042126f3ad8a57da9ce8",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="JSON object of argparse destination names; CLI overrides it")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    p.add_argument("--repo", type=Path, help="Official F5-TTS checkout at the pinned revision")
    p.add_argument("--reference-audio", type=Path)
    p.add_argument("--reference-text")
    p.add_argument("--checkpoint", type=Path, help="Optional local standard F5TTS_v1_Base EMA safetensors")
    p.add_argument("--vocab", type=Path, help="Optional matching local vocabulary")
    p.add_argument("--vocos-dir", type=Path, help="Local pinned Vocos config.yaml and pytorch_model.bin")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="cpu or cuda:N")
    p.add_argument("--output-dir", type=Path, default=Path("f5_samples"))
    p.add_argument("--sample-rate", type=int, default=NATIVE_RATE)
    p.add_argument("--pause-ms", type=float, default=0)
    p.add_argument("--crossfade-ms", type=float, default=0)
    p.add_argument(
        "--fade-ms", type=float, default=5, help="Length-preserving per-chunk edge fades, 0..20 ms"
    )
    p.add_argument("--max-vram-gb", type=float, default=5, help="Decimal GB, maximum 5")
    p.add_argument("--estimate-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--speed", type=float, default=1)
    p.add_argument(
        "--duration-seconds",
        type=float,
        help="Target generated seconds, excluding pauses/overlaps; quantized to frames",
    )
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--cfg-strength", type=float, default=2)
    p.add_argument("--sway", type=float, default=-1)
    p.add_argument("--ode-method", choices=("euler", "midpoint", "rk4"), default="euler")
    p.add_argument("--no-epss", action="store_true", help="Use uniformly spaced flow sampling times")
    p.add_argument("--target-rms", type=float, default=0.1)
    p.add_argument(
        "--max-chunk-seconds", type=float, default=6, help="Bound requested generated duration per chunk"
    )
    p.add_argument("--max-chunk-bytes", type=int, default=160)
    p.add_argument("--wav-subtype", choices=("PCM_16", "PCM_24", "FLOAT"), default="PCM_24")
    p.add_argument(
        "--peak-limit", type=float, default=0.98, help="Peak ceiling via amplitude-only attenuation, (0,1]"
    )
    known, _ = p.parse_known_args(argv)
    if known.config:
        settings = json.loads(known.config.read_text(encoding="utf-8"))
        actions = {action.dest: action for action in p._actions if action.dest not in ("help", "config")}
        allowed = set(actions)
        if not isinstance(settings, dict) or set(settings) - allowed:
            p.error("Config must be an object containing only documented argument destination names")
        for name, value in settings.items():
            action = actions[name]
            if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
                valid = isinstance(value, bool)
            elif action.type is int:
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif action.type is float:
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            else:
                valid = isinstance(value, str)
            if not valid or action.choices and value not in action.choices:
                p.error(f"Invalid JSON type/value for {name!r}")
        arguments = sys.argv[1:] if argv is None else argv
        if any(value == "--text" or value.startswith("--text=") for value in arguments):
            settings["text_file"] = None
        if any(value == "--text-file" or value.startswith("--text-file=") for value in arguments):
            settings["text"] = None
        p.set_defaults(**settings)
    a = p.parse_args(argv)
    for name in ("repo", "reference_audio", "checkpoint", "vocab", "vocos_dir", "output_dir", "text_file"):
        value = getattr(a, name)
        if value is not None:
            setattr(a, name, Path(value))
    if (a.text is None) == (a.text_file is None):
        p.error("Supply exactly one of --text or --text-file")
    if not a.repo or not a.reference_audio or not a.reference_text or not a.reference_text.strip():
        p.error("--repo, --reference-audio and a nonempty --reference-text are required")
    numeric = (
        a.speed,
        a.max_vram_gb,
        a.target_rms,
        a.max_chunk_seconds,
        a.pause_ms,
        a.crossfade_ms,
        a.fade_ms,
        a.peak_limit,
        a.cfg_strength,
        a.sway,
    )
    if not all(math.isfinite(v) for v in numeric):
        p.error("All numerical settings must be finite")
    if not 0 < a.max_vram_gb <= 5 or a.n < 1 or a.steps < 2 or a.speed <= 0 or a.target_rms <= 0:
        p.error("Require 0 < max-vram-gb <= 5, n >= 1, steps >= 2, speed > 0 and target-rms > 0")
    if (
        a.sample_rate < 8000
        or a.sample_rate > 192000
        or not 0.2 <= a.max_chunk_seconds <= 12
        or a.max_chunk_bytes < 8
    ):
        p.error("Require sample-rate 8000..192000, max-chunk-seconds .2..12 and max-chunk-bytes >= 8")
    if a.pause_ms < 0 or a.crossfade_ms < 0 or (a.pause_ms and a.crossfade_ms):
        p.error("Pauses and crossfades must be nonnegative and cannot both be enabled")
    if not 0 <= a.fade_ms <= 20:
        p.error("fade-ms must be in [0,20]; long fades can erase initial/final speech")
    if not 0 < a.peak_limit <= 1:
        p.error("peak-limit must be in (0,1]")
    if a.duration_seconds is not None and (not math.isfinite(a.duration_seconds) or a.duration_seconds <= 0):
        p.error("duration-seconds must be positive and finite")
    if a.cfg_strength < 0 or not -1 <= a.sway <= 1:
        p.error("Require cfg-strength >= 0 and sway in [-1, 1]")
    return a


def check_source(repo: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(f"F5 source must be at {SOURCE_REVISION}; found {revision}")
    for relative, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((repo / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Modified/incompatible duration source: {relative}")
    changed = subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", "src/f5_tts/model"], check=False
    )
    if changed.returncode:
        raise RuntimeError("The pinned model source has local modifications")
    sys.path.insert(0, str((repo / "src").resolve()))


class MemoryMonitor:
    """Allocator cap plus measured allocator peaks and optional sampled NVML usage."""

    def __init__(self, torch: Any, device: str, limit_gb: float):
        self.torch, self.device, self.limit = torch, torch.device(device), int(limit_gb * 1e9)
        self.stop_event = threading.Event()
        self.nvml_peak: int | None = None
        self.thread: threading.Thread | None = None
        self.fraction: float | None = None
        if self.device.type == "cpu":
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Only cpu and available cuda:N devices are supported")
        torch.cuda.set_device(self.device)
        total = torch.cuda.get_device_properties(self.device).total_memory
        available_budget = min(self.limit, total) - int(0.75e9)
        if available_budget <= 0:
            raise ValueError("VRAM budget must leave 0.75 GB for CUDA overhead")
        self.fraction = available_budget / total
        torch.cuda.set_per_process_memory_fraction(self.fraction, self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            import pynvml

            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]

            def observe() -> None:
                while not self.stop_event.is_set():
                    usage = 0
                    supported = False
                    for handle in handles:
                        try:
                            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                            for proc in procs:
                                if (
                                    proc.pid == os.getpid()
                                    and isinstance(proc.usedGpuMemory, int)
                                    and proc.usedGpuMemory < (1 << 60)
                                ):
                                    usage += proc.usedGpuMemory
                                    supported = True
                        except pynvml.NVMLError:
                            pass
                    if supported:
                        self.nvml_peak = max(self.nvml_peak or 0, usage)
                    self.stop_event.wait(0.05)

            self.thread = threading.Thread(target=observe, daemon=True)
            self.thread.start()
        except Exception:
            # NVML is optional; Windows WDDM may not expose process allocation.
            pass

    def snapshot(self) -> dict[str, Any]:
        if self.device.type == "cpu":
            return {"gpu_peak_bytes": 0, "device": "cpu", "hard_total_vram_guarantee": True}
        self.torch.cuda.synchronize(self.device)
        allocated = self.torch.cuda.max_memory_allocated(self.device)
        reserved = self.torch.cuda.max_memory_reserved(self.device)
        result = {
            "torch_peak_allocated_bytes": allocated,
            "torch_peak_reserved_bytes": reserved,
            "nvml_sampled_process_peak_bytes": self.nvml_peak,
            "allocator_fraction": self.fraction,
            "budget_bytes": self.limit,
            "hard_total_vram_guarantee": False,
        }
        if max(allocated, reserved, self.nvml_peak or 0) > self.limit:
            raise RuntimeError(f"Observed GPU peak exceeds budget: {result}")
        return result

    def close(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)


def split_text(text: str, byte_budget: int) -> list[str]:
    """Bound UTF-8 bytes while preferring nearby sentence and clause endings."""
    if byte_budget < 1:
        raise ValueError("Chunk byte budget must be positive")
    words, result, start = text.split(), [], 0
    for word in words:
        if len(word.encode("utf-8")) > byte_budget:
            raise ValueError(f"A word exceeds chunk byte budget: {word!r}; increase --max-chunk-bytes")
    while start < len(words):
        end = start + 1
        while end < len(words) and len(" ".join(words[start : end + 1]).encode("utf-8")) <= byte_budget:
            end += 1
        if end < len(words):
            # A nearby complete sentence provides better prosodic context than
            # cutting solely when a byte counter fills up. Do not create tiny
            # punctuation chunks by selecting a boundary near the chunk start.
            minimum = start + max(1, (end - start) // 2)
            for pattern in (r"[.!?。！？][\"'”’\])}]*$", r"[,;:，；：][\"'”’\])}]*$"):
                candidates = [i for i in range(minimum, end + 1) if re.search(pattern, words[i - 1])]
                if candidates:
                    end = candidates[-1]
                    break
        result.append(" ".join(words[start:end]))
        start = end
    if len(result) > 1 and len(result[-1].encode("utf-8")) < 10:
        combined = result[-2] + " " + result[-1]
        if len(combined.encode("utf-8")) <= byte_budget:
            result[-2:] = [combined]
        else:
            previous = result[-2].split()
            for cut in range(len(previous) - 1, 0, -1):
                tail = " ".join(previous[cut:]) + " " + result[-1]
                if len(tail.encode("utf-8")) > byte_budget:
                    break
                if len(tail.encode("utf-8")) >= 10:
                    result[-2:] = [" ".join(previous[:cut]), tail]
                    break
    return result


def fade_edges(wave: Any, samples: int, np: Any) -> Any:
    """Remove hard edge discontinuities while preserving the waveform length."""
    count = min(max(0, samples), len(wave) // 2)
    if not count:
        return wave
    result = wave.copy()
    ramp = np.linspace(0, 1, count, dtype=np.float32)
    result[:count] *= ramp
    result[-count:] *= ramp[::-1]
    return result


def limit_peak(wave: Any, ceiling: float, np: Any) -> tuple[Any, dict[str, float]]:
    """Prevent integer-WAV clipping through one length-preserving global gain."""
    original_peak = float(np.max(np.abs(wave)))
    gain = min(1.0, ceiling / original_peak) if original_peak else 1.0
    return wave * gain, {"original_peak": original_peak, "gain": gain, "ceiling": ceiling}


def joined_length(lengths: list[int], pause: int, overlap: int) -> int:
    total = lengths[0]
    for length in lengths[1:]:
        total += pause + length - min(overlap, total, length)
    return total


def join_audio(waves: list[Any], pause: int, overlap: int, np: Any) -> Any:
    final = waves[0]
    for wave in waves[1:]:
        count = min(overlap, len(final), len(wave))
        if count:
            ramp = np.linspace(0, 1, count, dtype=np.float32)
            final = np.concatenate(
                (final[:-count], final[-count:] * (1 - ramp) + wave[:count] * ramp, wave[count:])
            )
        else:
            final = np.concatenate((final, np.zeros(pause, dtype=np.float32), wave))
    return final


def build_plan(
    a: argparse.Namespace, text: str, ref_wave: Any, ref_text: str, vocab_map: dict[str, int]
) -> list[dict[str, Any]]:
    from f5_tts.model.utils import convert_char_to_pinyin, list_str_to_idx

    ref_frames = ref_wave.shape[-1] // HOP
    # Centered F5 mel has floor(reference_samples / hop) + 1 frames.
    conditioning_frames = ref_frames + 1
    rate_bytes = len(ref_text.encode("utf-8")) / (ref_wave.shape[-1] / NATIVE_RATE)
    byte_budget = min(a.max_chunk_bytes, max(8, int(rate_bytes * a.max_chunk_seconds * a.speed)))
    chunks = split_text(text, byte_budget)
    weights = [len(chunk.encode("utf-8")) for chunk in chunks]
    total_weight = sum(weights)
    plans = []
    for chunk, weight in zip(chunks, weights):
        phonemes = convert_char_to_pinyin([ref_text + chunk])
        tokens = list_str_to_idx(phonemes, vocab_map)
        token_count = int((tokens != -1).sum().item())
        if a.duration_seconds is None:
            # Freeze an explicit allocation at the user's requested speed.
            # The upstream <10-byte special case uses speed=0.3, which can
            # turn a tiny tail into a conspicuously prolonged final word/pause.
            requested = ref_frames + int(ref_frames / len(ref_text.encode("utf-8")) * weight / a.speed)
        else:
            allocated_samples = a.duration_seconds * NATIVE_RATE * weight / total_weight
            requested = ref_frames + max(2, int(allocated_samples / HOP) + 1)
        frames = min(65536, max(requested, max(token_count, conditioning_frames) + 1))
        if frames <= ref_frames + 1 or frames * HOP / NATIVE_RATE > 22:
            raise ValueError(
                "Chunk is empty or exceeds 22 total conditioning/generation seconds; shorten the chunk/reference"
            )
        sample_count = (frames - ref_frames - 1) * HOP
        if sample_count > a.max_chunk_seconds * NATIVE_RATE + HOP:
            raise ValueError(
                "Final allocated chunk exceeds --max-chunk-seconds after the CFM clamp; use a smaller byte budget or larger chunk limit"
            )
        plans.append(
            {
                "text": chunk,
                "phonemes": phonemes,
                "duration_frames": frames,
                "reference_slice_frames": ref_frames,
                "conditioning_frames": conditioning_frames,
                "native_samples": sample_count,
                "token_count": token_count,
            }
        )
    return plans


def load_models(a: argparse.Namespace, vocab_path: Path, torch: Any) -> tuple[Any, Any, dict[str, str]]:
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from vocos import Vocos

    if importlib.metadata.version("vocos") != "0.1.0":
        raise RuntimeError("The validated Vocos decoder is vocos==0.1.0")
    checkpoint = a.checkpoint or Path(
        hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors", revision=WEIGHT_REVISION)
    )
    if checkpoint.suffix != ".safetensors":
        raise ValueError("This adapter accepts only official-layout EMA .safetensors checkpoints")
    vocab_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    model = CFM(
        transformer=DiT(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=True,
            qk_norm=None,
            conv_layers=4,
            pe_attn_head=None,
            attn_backend="torch",
            attn_mask_enabled=False,
            text_num_embeds=vocab_size,
            mel_dim=100,
        ),
        mel_spec_kwargs={
            "n_fft": 1024,
            "hop_length": HOP,
            "win_length": 1024,
            "n_mel_channels": 100,
            "target_sample_rate": NATIVE_RATE,
            "mel_spec_type": "vocos",
        },
        odeint_kwargs={"method": a.ode_method},
        vocab_char_map=vocab_map,
    )
    state = load_file(str(checkpoint), device="cpu")
    state = {
        key.removeprefix("ema_model."): value
        for key, value in state.items()
        if key not in ("initted", "step")
    }
    for key in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
        state.pop(key, None)
    model.load_state_dict(state, strict=True)
    del state
    dtype = torch.float16 if str(a.device).startswith("cuda") else torch.float32
    model = model.eval().to(device=a.device, dtype=dtype)
    if a.vocos_dir:
        config_file, vocos_file = a.vocos_dir / "config.yaml", a.vocos_dir / "pytorch_model.bin"
    else:
        config_file = Path(
            hf_hub_download("charactr/vocos-mel-24khz", "config.yaml", revision=VOCOS_REVISION)
        )
        vocos_file = Path(
            hf_hub_download("charactr/vocos-mel-24khz", "pytorch_model.bin", revision=VOCOS_REVISION)
        )
    vocoder = Vocos.from_hparams(str(config_file))
    vocoder.load_state_dict(torch.load(vocos_file, map_location="cpu", weights_only=True), strict=True)
    vocoder = vocoder.eval().to(a.device)
    istft = vocoder.head.istft
    if (istft.padding, istft.hop_length, istft.n_fft) != ("center", HOP, 1024):
        raise RuntimeError(
            "Unsupported Vocos length mapping; require official centered 24 kHz/256-hop decoder"
        )
    return (
        model,
        vocoder,
        {
            "checkpoint": str(checkpoint),
            "vocab": str(vocab_path),
            "vocos_config": str(config_file),
            "vocos_weights": str(vocos_file),
        },
    )


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    a = parse_args(argv)
    text = a.text if a.text is not None else a.text_file.read_text(encoding="utf-8-sig")
    text = text.strip()
    word_count = len(re.findall(r"\S+", text))
    if not 10 <= word_count <= 500:
        raise ValueError(
            f"This requested evaluation range is 10..500 whitespace-separated words; got {word_count}"
        )
    check_source(a.repo.resolve())
    import numpy as np
    import soundfile as sf
    import torch
    from f5_tts.model.utils import get_tokenizer
    from huggingface_hub import hf_hub_download
    from scipy.signal import resample_poly
    from tqdm.auto import tqdm

    if (a.output_dir / "manifest.json").exists() or any(a.output_dir.glob("sample_*.wav")):
        raise FileExistsError(
            f"Output directory already contains run artifacts: {a.output_dir}; choose a fresh --output-dir"
        )
    monitor = MemoryMonitor(torch, a.device, a.max_vram_gb)
    try:
        a.output_dir.mkdir(parents=True, exist_ok=True)
        wave, sr = sf.read(a.reference_audio, dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
        if sr != NATIVE_RATE:
            factor = math.gcd(sr, NATIVE_RATE)
            wave = resample_poly(wave, NATIVE_RATE // factor, sr // factor).astype(np.float32)
        if not 1 <= len(wave) / NATIVE_RATE <= 8 or not np.isfinite(wave).all():
            raise ValueError("Provide a finite, clean reference between 1 and 8 seconds")
        rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
        if rms <= 1e-6:
            raise ValueError("Reference is effectively silent")
        ref_wave = torch.from_numpy(wave.copy()).unsqueeze(0)
        if rms < a.target_rms:
            ref_wave = ref_wave * (a.target_rms / rms)
        ref_text = a.reference_text.strip()
        if not ref_text.endswith((".", "。")):
            ref_text += "."
        ref_text += " "
        vocab_path = a.vocab or Path(
            hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/vocab.txt", revision=WEIGHT_REVISION)
        )
        vocab_map, _ = get_tokenizer(str(vocab_path), "custom")
        startup_seconds = time.perf_counter() - started
        planning_started = time.perf_counter()
        plans = build_plan(a, text, ref_wave, ref_text, vocab_map)
        preflight_seconds = time.perf_counter() - planning_started
        pause = round(a.pause_ms * NATIVE_RATE / 1000)
        overlap = int(a.crossfade_ms * NATIVE_RATE / 1000)
        native_count = joined_length([p["native_samples"] for p in plans], pause, overlap)
        exported_count = (native_count * a.sample_rate + NATIVE_RATE - 1) // NATIVE_RATE
        preview = {
            "native_samples": native_count,
            "exported_samples": exported_count,
            "seconds": exported_count / a.sample_rate,
            "chunks": len(plans),
            "startup_seconds": startup_seconds,
            "preflight_seconds": preflight_seconds,
            "estimator": "F5 allocated frames; CFM text/reference clamp; centered Vocos; deterministic joins/resampling",
            "samples_share_duration": True,
        }
        print(json.dumps({"preflight": preview}, indent=2), flush=True)
        manifest = {
            "model": "F5TTS_v1_Base",
            "source_revision": SOURCE_REVISION,
            "weight_revision": WEIGHT_REVISION,
            "vocos_revision": VOCOS_REVISION,
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            "text": text,
            "word_count": word_count,
            "preview": preview,
            "plan": plans,
            "outputs": [],
            "memory": monitor.snapshot(),
        }
        manifest_path = a.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if a.estimate_only:
            return 0
        load_started = time.perf_counter()
        model, vocoder, weights = load_models(a, vocab_path, torch)
        manifest["weights"] = weights
        manifest["model_load_seconds"] = time.perf_counter() - load_started
        ref_wave = ref_wave.to(a.device)
        # Validate the arithmetic against the actual feature frontend before synthesis.
        actual_reference_frames = model.mel_spec(ref_wave).shape[-1]
        if actual_reference_frames != plans[0]["conditioning_frames"]:
            raise RuntimeError(
                "Reference mel frontend length differs from the validated centered-STFT mapping"
            )
        with tqdm(total=a.n * len(plans), desc="F5 generated chunks", unit="chunk") as progress:
            for sample in range(a.n):
                sample_started = time.perf_counter()
                waves = []
                sample_seed = a.seed + sample
                for index, plan in enumerate(plans):
                    with torch.inference_mode():
                        evaluations = a.steps * {"euler": 1, "midpoint": 2, "rk4": 4}[a.ode_method]
                        with tqdm(
                            total=evaluations,
                            desc=f"Flow sample {sample + 1}, chunk {index + 1}",
                            unit="eval",
                            leave=False,
                        ) as flow_progress:

                            def advance_flow(*_: Any) -> None:
                                flow_progress.update(1)

                            hook = model.transformer.register_forward_hook(advance_flow)
                            try:
                                generated, trajectory = model.sample(
                                    cond=ref_wave,
                                    text=plan["phonemes"],
                                    duration=plan["duration_frames"],
                                    steps=a.steps,
                                    cfg_strength=a.cfg_strength,
                                    sway_sampling_coef=a.sway,
                                    seed=sample_seed + index * 1000003,
                                    use_epss=not a.no_epss,
                                )
                            finally:
                                hook.remove()
                        del trajectory
                        if generated.shape[1] != plan["duration_frames"]:
                            raise RuntimeError("CFM returned an unexpected frame count")
                        mel = generated[:, plan["reference_slice_frames"] :, :].float().permute(0, 2, 1)
                        audio = vocoder.decode(mel).reshape(-1)
                        if rms < a.target_rms:
                            audio = audio * (rms / a.target_rms)
                        chunk_wave = audio.cpu().numpy().copy()
                        del generated, mel, audio
                    if len(chunk_wave) != plan["native_samples"]:
                        raise RuntimeError("Vocos length mismatch; refusing to pad or trim speech")
                    if not np.isfinite(chunk_wave).all():
                        raise RuntimeError("Model generated NaN/Inf audio")
                    chunk_wave = fade_edges(chunk_wave, round(a.fade_ms * NATIVE_RATE / 1000), np)
                    waves.append(chunk_wave)
                    progress.update(1)
                    monitor.snapshot()
                final = join_audio(waves, pause, overlap, np)
                if a.sample_rate != NATIVE_RATE:
                    factor = math.gcd(a.sample_rate, NATIVE_RATE)
                    final = resample_poly(final, a.sample_rate // factor, NATIVE_RATE // factor).astype(
                        np.float32
                    )
                measured_count = len(final)
                relative_error = abs(measured_count - exported_count) / measured_count
                if measured_count != exported_count:
                    raise RuntimeError(
                        f"Exact duration contract failed ({exported_count} planned versus {measured_count} produced; error {relative_error:.6%}); no sample exported"
                    )
                final, peak_normalization = limit_peak(final, a.peak_limit, np)
                monitor.snapshot()
                destination = a.output_dir / f"sample_{sample + 1:03d}_seed_{sample_seed}.wav"
                sf.write(destination, final, a.sample_rate, subtype=a.wav_subtype)
                info = sf.info(destination)
                if info.frames != measured_count or info.samplerate != a.sample_rate:
                    raise RuntimeError("WAV export length validation failed")
                manifest["outputs"].append(
                    {
                        "path": str(destination.resolve()),
                        "seed": sample_seed,
                        "peak_normalization": peak_normalization,
                        "predicted_samples": exported_count,
                        "measured_samples": info.frames,
                        "duration_seconds": info.duration,
                        "relative_error": relative_error,
                        "generation_seconds": time.perf_counter() - sample_started,
                        "absolute_peak_amplitude": float(np.max(np.abs(final))),
                    }
                )
                manifest["memory"] = monitor.snapshot()
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0
    finally:
        monitor.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

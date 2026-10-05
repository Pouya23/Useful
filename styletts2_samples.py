#!/usr/bin/env python3
"""Standalone StyleTTS2 LibriTTS samples with cached exact duration plans.

Python 3.10-3.12 recommended. Obtain the official source at SOURCE_REVISION:
  git clone https://github.com/yl4579/StyleTTS2.git StyleTTS2
  git -C StyleTTS2 checkout 5cedc71c333f8d8b8551ca59378bdcc7af4c9529
Install the repository requirements, phonemizer, espeak-ng, scipy, soundfile,
huggingface_hub and matching torch/torchaudio. Espeak must be discoverable by
phonemizer; on Windows set PHONEMIZER_ESPEAK_LIBRARY to its libespeak-ng DLL.
  python styletts2_samples.py --repo ./StyleTTS2 --text-file passage.txt \
      --reference-audio reference.wav --n 3

Uses the fully downloadable official yl4579/StyleTTS2-LibriTTS config and
checkpoint at WEIGHT_REVISION. This checkpoint uses HifiGAN. Native output
is 24 kHz; exported sampling rates are deterministic resampling. Voice and
style come from reference audio and alpha/beta/style diffusion controls.
Speed is applied to predicted phoneme durations before rounding. There is
no separate reliable emotion label or lexical emphasis API in this adapter.

Exact prediction needs BERT, style diffusion and the duration predictor, but
does not run F0/energy prediction or the waveform decoder. Thus warm preview
latency is measured and is not promised to be near-instant on all hardware.
The sampled styles, duration-conditioned encoding and integer durations are
cached on CPU and reused in synthesis, even when --n produces different
styles/durations. --estimate-only writes this plan; --reuse-plan resumes it.
The default fixed 50-sample tail trim is the official notebook's pulse
workaround; set --tail-trim-samples 0 to retain the complete decoder output.
No audio-content-dependent trimming or duration-fitting padding is used.
Chunks prefer complete sentences/clauses; short length-preserving edge fades
reduce clicks at joins. Reference timbre/prosody are favored by default.

CPU is the default: zero GPU VRAM. CUDA gets a conservative PyTorch allocator
cap and measured peaks, but a universal total-process 5 GB guarantee cannot
be made because CUDA libraries allocate outside that allocator. Optional
sampled NVML usage is also reported. An observed budget violation fails.
"""

from __future__ import annotations

import argparse
import hashlib
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

SOURCE_REVISION = "5cedc71c333f8d8b8551ca59378bdcc7af4c9529"
WEIGHT_REVISION = "3aa7ba7f8f275ec13dce21682a61494c35089e2a"
NATIVE_RATE = 24000
SOURCE_HASHES = {
    "models.py": "d9f54b98514d142dd20663318784a63805b2b718ace7a44b738431c6d0f48daf",
    "Modules/hifigan.py": "1ce902afae4882c9dfb9f8a85141eeb4ae3e6de7002ae9212efb484a68d00ad7",
    "Utils/PLBERT/util.py": "f9e4ed98d2f05f0961141aa262f87d887b1665e9dfa99564e36d4ecb21d9ce73",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config", type=Path, help="JSON object of argparse destination names; CLI takes precedence"
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--text")
    g.add_argument("--text-file", type=Path)
    p.add_argument("--repo", type=Path)
    p.add_argument("--reference-audio", type=Path)
    p.add_argument("--checkpoint", type=Path, help="Optional local official-layout LibriTTS .pth checkpoint")
    p.add_argument("--model-config", type=Path, help="Optional local official LibriTTS config.yml")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="cpu or cuda:N")
    p.add_argument("--output-dir", type=Path, default=Path("styletts2_samples"))
    p.add_argument("--sample-rate", type=int, default=NATIVE_RATE)
    p.add_argument("--pause-ms", type=float, default=0)
    p.add_argument("--crossfade-ms", type=float, default=0)
    p.add_argument(
        "--fade-ms", type=float, default=5, help="Length-preserving per-chunk edge fades, 0..20 ms"
    )
    p.add_argument("--max-vram-gb", type=float, default=5, help="Decimal GB, maximum 5")
    p.add_argument("--estimate-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--reuse-plan", type=Path, help="A duration_plan.pt produced with identical synthesis settings"
    )
    p.add_argument("--speed", type=float, default=1)
    p.add_argument(
        "--alpha", type=float, default=0.1, help="Weight of sampled acoustic style versus reference [0,1]"
    )
    p.add_argument(
        "--beta", type=float, default=0.3, help="Weight of sampled prosodic style versus reference [0,1]"
    )
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument("--embedding-scale", type=float, default=1)
    p.add_argument(
        "--style-continuity",
        type=float,
        default=0.7,
        help="Previous sampled style weight between chunks [0,1]",
    )
    p.add_argument("--pitch-scale", type=float, default=1, help="Multiply predicted F0 before decoding")
    p.add_argument("--max-chunk-words", type=int, default=35)
    p.add_argument("--max-chunk-tokens", type=int, default=400)
    p.add_argument(
        "--max-chunk-seconds",
        type=float,
        default=20,
        help="Reject overly long planned chunks before decoding",
    )
    p.add_argument(
        "--tail-trim-samples", type=int, default=50, help="Fixed per-chunk trim for notebook's end pulse"
    )
    p.add_argument("--language", choices=("en-us", "en-gb"), default="en-us")
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
            p.error("Config must contain only documented argument destination names")
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
    for name in (
        "repo",
        "reference_audio",
        "checkpoint",
        "model_config",
        "output_dir",
        "text_file",
        "reuse_plan",
    ):
        if getattr(a, name) is not None:
            setattr(a, name, Path(getattr(a, name)))
    if (a.text is None) == (a.text_file is None):
        p.error("Supply exactly one of --text or --text-file")
    if not a.repo or not a.reference_audio:
        p.error("--repo and --reference-audio are required")
    numbers = (
        a.max_vram_gb,
        a.speed,
        a.alpha,
        a.beta,
        a.embedding_scale,
        a.style_continuity,
        a.pitch_scale,
        a.pause_ms,
        a.crossfade_ms,
        a.fade_ms,
        a.peak_limit,
        a.max_chunk_seconds,
    )
    if not all(math.isfinite(value) for value in numbers):
        p.error("Numeric settings must be finite")
    if not 0 < a.max_vram_gb <= 5 or a.n < 1 or a.speed <= 0 or a.pitch_scale <= 0:
        p.error("Require 0 < max-vram-gb <= 5, n >= 1, speed > 0 and pitch-scale > 0")
    if not all(0 <= v <= 1 for v in (a.alpha, a.beta, a.style_continuity)):
        p.error("alpha, beta and style-continuity must be in [0,1]")
    if a.embedding_scale <= 0 or a.diffusion_steps < 2 or not 1 <= a.max_chunk_words <= 80:
        p.error("Require embedding-scale > 0, diffusion-steps >= 2 and max-chunk-words 1..80")
    if not 16 <= a.max_chunk_tokens <= 500 or not 1 <= a.max_chunk_seconds <= 30:
        p.error("Require max-chunk-tokens 16..500 and max-chunk-seconds 1..30")
    if not 8000 <= a.sample_rate <= 192000 or a.tail_trim_samples < 0:
        p.error("Require sample-rate 8000..192000 and tail-trim-samples >= 0")
    if a.pause_ms < 0 or a.crossfade_ms < 0 or (a.pause_ms and a.crossfade_ms):
        p.error("Pauses and crossfades must be nonnegative and cannot both be enabled")
    if not 0 <= a.fade_ms <= 20:
        p.error("fade-ms must be in [0,20]; long fades can erase initial/final speech")
    if not 0 < a.peak_limit <= 1:
        p.error("peak-limit must be in (0,1]")
    return a


def check_source(repo: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    if result.stdout.strip() != SOURCE_REVISION:
        raise RuntimeError(f"StyleTTS2 source must be at {SOURCE_REVISION}")
    for relative, expected in SOURCE_HASHES.items():
        digest = hashlib.sha256((repo / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Modified/incompatible duration or decoder source: {relative}")
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "models.py",
            "Modules",
            "text_utils.py",
            "Utils/PLBERT",
        ],
        check=False,
    )
    if changed.returncode:
        raise RuntimeError("The pinned inference source has local modifications")
    sys.path.insert(0, str(repo.resolve()))


class MemoryMonitor:
    def __init__(self, torch: Any, device: str, limit_gb: float):
        self.torch, self.device, self.limit = torch, torch.device(device), int(limit_gb * 1e9)
        self.stop_event, self.thread = threading.Event(), None
        self.nvml_peak, self.fraction = None, None
        if self.device.type == "cpu":
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Only cpu and available cuda:N devices are supported")
        torch.cuda.set_device(self.device)
        total = torch.cuda.get_device_properties(self.device).total_memory
        allocator_budget = min(total, self.limit) - int(0.75e9)
        if allocator_budget <= 0:
            raise ValueError("VRAM budget must leave .75 GB for CUDA overhead")
        self.fraction = allocator_budget / total
        torch.cuda.set_per_process_memory_fraction(self.fraction, self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            import pynvml

            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]

            def observe() -> None:
                while not self.stop_event.is_set():
                    usage, supported = 0, False
                    for handle in handles:
                        try:
                            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                                if (
                                    proc.pid == os.getpid()
                                    and isinstance(proc.usedGpuMemory, int)
                                    and proc.usedGpuMemory < 1 << 60
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
            # NVML is optional and may be unavailable under Windows WDDM.
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
            raise RuntimeError(f"Observed GPU memory exceeds budget: {result}")
        return result

    def close(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)


def load_models(a: argparse.Namespace, torch: Any) -> tuple[Any, Any, dict[str, str]]:
    import yaml
    from huggingface_hub import hf_hub_download
    from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
    from munch import Munch
    from transformers import AlbertConfig
    from Utils.PLBERT.util import CustomAlbert

    from models import build_model

    model_config = a.model_config or Path(
        hf_hub_download("yl4579/StyleTTS2-LibriTTS", "Models/LibriTTS/config.yml", revision=WEIGHT_REVISION)
    )
    checkpoint = a.checkpoint or Path(
        hf_hub_download(
            "yl4579/StyleTTS2-LibriTTS", "Models/LibriTTS/epochs_2nd_00020.pth", revision=WEIGHT_REVISION
        )
    )
    config = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    parameters = Munch.fromDict(config["model_params"])
    decoder = parameters.decoder
    if (
        decoder.type,
        list(decoder.upsample_rates),
        list(decoder.upsample_kernel_sizes),
        parameters.style_dim,
        config["preprocess_params"]["sr"],
    ) != ("hifigan", [10, 5, 3, 2], [20, 10, 6, 4], 128, NATIVE_RATE):
        raise RuntimeError("Require the official LibriTTS HifiGAN/128-style/24 kHz model configuration")
    bert_config = yaml.safe_load((a.repo / "Utils/PLBERT/config.yml").read_text(encoding="utf-8"))
    bert = CustomAlbert(AlbertConfig(**bert_config["model_params"]))
    # ASR, pitch extractor and discriminators are only used for training.
    model = build_model(parameters, None, None, bert)
    inference_keys = (
        "bert",
        "bert_encoder",
        "predictor",
        "decoder",
        "text_encoder",
        "predictor_encoder",
        "style_encoder",
        "diffusion",
    )
    for name in list(model.keys()):
        if name not in inference_keys:
            del model[name]
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = weights["net"]
    for name in inference_keys:
        if name not in weights:
            raise RuntimeError(f"Checkpoint is missing inference module {name}")
        state = {key.removeprefix("module."): value for key, value in weights[name].items()}
        # Transformers no longer register this deterministic buffer persistently.
        expected = model[name].state_dict()
        for key in list(state):
            if key.endswith("embeddings.position_ids") and key not in expected:
                del state[key]
        model[name].load_state_dict(state, strict=True)
        model[name].eval()
    del weights
    for name in ("bert", "bert_encoder", "predictor", "predictor_encoder", "style_encoder", "diffusion"):
        model[name].to(a.device)
    # Text encoding and the decoder stay on CPU until the duration preview has printed.
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3, rho=9),
        clamp=False,
    )
    for upsample in model.decoder.generator.ups:
        stride, kernel, padding, out = (
            upsample.stride[0],
            upsample.kernel_size[0],
            upsample.padding[0],
            upsample.output_padding[0],
        )
        if kernel - 2 * padding + out != stride:
            raise RuntimeError(
                "Decoder transpose convolution no longer multiplies length exactly by its stride"
            )
    return (
        model,
        sampler,
        {
            "checkpoint": str(checkpoint),
            "model_config": str(model_config),
            "decoder_samples_per_duration_unit": "600",
        },
    )


def reference_style(a: argparse.Namespace, model: Any, torch: Any, np: Any) -> Any:
    import soundfile as sf
    import torchaudio
    from scipy.signal import resample_poly

    wave, rate = sf.read(a.reference_audio, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if rate != NATIVE_RATE:
        factor = math.gcd(rate, NATIVE_RATE)
        wave = resample_poly(wave, NATIVE_RATE // factor, rate // factor).astype(np.float32)
    if not 0.5 <= len(wave) / NATIVE_RATE <= 15 or not np.isfinite(wave).all():
        raise ValueError("Provide a clean finite reference of .5..15 seconds")
    if float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))) < 1e-6:
        raise ValueError("Reference is effectively silent")
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=80, n_fft=2048, win_length=1200, hop_length=300
    )
    # The official notebook leaves sample_rate at torchaudio's 16000 default
    # while supplying a 24 kHz reference: preserve that trained inference path.
    mel = mel_transform(torch.from_numpy(wave.copy()))
    mel = ((torch.log(1e-5 + mel.unsqueeze(0)) + 4) / 4).unsqueeze(1).to(a.device)
    with torch.inference_mode():
        result = torch.cat((model.style_encoder(mel), model.predictor_encoder(mel)), dim=1)
    return result


def split_text(text: str, word_budget: int) -> list[str]:
    """Prefer sentence/clause endings within a strict word-count bound."""
    if word_budget < 1:
        raise ValueError("Chunk word budget must be positive")
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        end = min(start + word_budget, len(words))
        if end < len(words):
            minimum = start + max(1, (end - start) // 2)
            for pattern in (r"[.!?。！？][\"'”’\])}]*$", r"[,;:，；：][\"'”’\])}]*$"):
                candidates = [i for i in range(minimum, end + 1) if re.search(pattern, words[i - 1])]
                if candidates:
                    end = candidates[-1]
                    break
        chunks.append(" ".join(words[start:end]))
        start = end
    return chunks


def fade_edges(wave: Any, samples: int, np: Any) -> Any:
    """Suppress hard chunk-edge discontinuities without changing sample count."""
    count = min(max(0, samples), len(wave) // 2)
    if not count:
        return wave
    result = wave.copy()
    ramp = np.linspace(0, 1, count, dtype=np.float32)
    result[:count] *= ramp
    result[-count:] *= ramp[::-1]
    return result


def limit_peak(wave: Any, ceiling: float, np: Any) -> tuple[Any, dict[str, float]]:
    """Apply only attenuation needed to prevent PCM export clipping."""
    original_peak = float(np.max(np.abs(wave)))
    gain = min(1.0, ceiling / original_peak) if original_peak else 1.0
    return wave * gain, {"original_peak": original_peak, "gain": gain, "ceiling": ceiling}


def tokenize_chunks(a: argparse.Namespace, text: str, torch: Any) -> list[dict[str, Any]]:
    from nltk.tokenize import TreebankWordTokenizer
    from phonemizer.backend import EspeakBackend
    from text_utils import TextCleaner

    phonemizer = EspeakBackend(language=a.language, preserve_punctuation=True, with_stress=True)
    tokenizer, cleaner = TreebankWordTokenizer(), TextCleaner()

    def encode(chunk: str) -> list[int]:
        phonemes = phonemizer.phonemize([chunk])[0]
        phonemes = " ".join(tokenizer.tokenize(phonemes)).replace("``", '"').replace("''", '"')
        return [0] + cleaner(phonemes)

    def split(words: list[str]) -> list[dict[str, Any]]:
        chunk = " ".join(words)
        tokens = encode(chunk)
        if len(tokens) <= a.max_chunk_tokens:
            return [{"text": chunk, "tokens": torch.tensor(tokens, dtype=torch.long).unsqueeze(0)}]
        if len(words) == 1:
            raise ValueError("One word exceeds the phoneme token limit")
        half = len(words) // 2
        for pattern in (r"[.!?。！？][\"'”’\])}]*$", r"[,;:，；：][\"'”’\])}]*$"):
            candidates = [i for i in range(1, len(words)) if re.search(pattern, words[i - 1])]
            if candidates:
                half = min(candidates, key=lambda i: abs(i - half))
                break
        return split(words[:half]) + split(words[half:])

    result = []
    for chunk in split_text(text, a.max_chunk_words):
        result.extend(split(chunk.split()))
    return result


def joined_length(lengths: list[int], pause: int, overlap: int) -> int:
    total = lengths[0]
    for count in lengths[1:]:
        total += count + pause - min(total, count, overlap)
    return total


def build_plans(
    a: argparse.Namespace,
    chunks: list[dict[str, Any]],
    model: Any,
    sampler: Any,
    reference: Any,
    torch: Any,
    tqdm: Any,
) -> list[list[dict[str, Any]]]:
    # BERT content embeddings are independent of sampled speaker/style, so compute once.
    embeddings = []
    with torch.inference_mode():
        for chunk in chunks:
            tokens = chunk["tokens"].to(a.device)
            lengths = torch.tensor([tokens.shape[-1]], dtype=torch.long, device=a.device)
            mask = torch.zeros((1, tokens.shape[-1]), dtype=torch.bool, device=a.device)
            bert = model.bert(tokens, attention_mask=(~mask).int())
            encoded = model.bert_encoder(bert).transpose(-1, -2)
            embeddings.append((bert.cpu(), encoded.cpu(), lengths.cpu(), mask.cpu()))
    plans = []
    with tqdm(total=a.n * len(chunks), desc="Style/duration preflight", unit="chunk") as bar:
        for sample in range(a.n):
            previous, sample_plan = None, []
            for index, (chunk, cached) in enumerate(zip(chunks, embeddings)):
                chunk_seed = a.seed + sample + index * 1000003
                torch.manual_seed(chunk_seed)
                bert, encoded, lengths, mask = (value.to(a.device) for value in cached)
                with torch.inference_mode():
                    style = sampler(
                        noise=torch.randn((1, 1, 256), device=a.device),
                        embedding=bert,
                        embedding_scale=a.embedding_scale,
                        features=reference,
                        num_steps=a.diffusion_steps,
                    ).squeeze(1)
                    if previous is not None:
                        style = a.style_continuity * previous + (1 - a.style_continuity) * style
                    prosody = a.beta * style[:, 128:] + (1 - a.beta) * reference[:, 128:]
                    acoustic = a.alpha * style[:, :128] + (1 - a.alpha) * reference[:, :128]
                    # Cache the *blended* style for the official long-form continuity path.
                    previous = torch.cat((acoustic, prosody), dim=-1)
                    duration_encoding = model.predictor.text_encoder(encoded, prosody, lengths, mask)
                    logits, _ = model.predictor.lstm(duration_encoding)
                    duration = torch.sigmoid(model.predictor.duration_proj(logits)).sum(dim=-1)
                    integers = torch.round(duration.reshape(-1) / a.speed).clamp(min=1).long()
                native_samples = int(integers.sum().item()) * 600 - a.tail_trim_samples
                if native_samples <= 0 or native_samples > a.max_chunk_seconds * NATIVE_RATE:
                    raise ValueError(
                        "Planned chunk is empty/too long; use smaller --max-chunk-words or faster --speed"
                    )
                sample_plan.append(
                    {
                        "text": chunk["text"],
                        "tokens": chunk["tokens"],
                        "duration_encoding": duration_encoding.cpu(),
                        "durations": integers.cpu(),
                        "prosody_style": prosody.cpu(),
                        "acoustic_style": acoustic.cpu(),
                        "native_samples": native_samples,
                        "chunk_seed": chunk_seed,
                    }
                )
                bar.update(1)
            plans.append(sample_plan)
    return plans


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(a: argparse.Namespace, text: str, weights: dict[str, str]) -> str:
    settings = {
        name: getattr(a, name)
        for name in (
            "n",
            "seed",
            "speed",
            "alpha",
            "beta",
            "diffusion_steps",
            "embedding_scale",
            "style_continuity",
            "pitch_scale",
            "max_chunk_words",
            "max_chunk_tokens",
            "tail_trim_samples",
            "language",
            "sample_rate",
            "pause_ms",
            "crossfade_ms",
            "fade_ms",
            "peak_limit",
        )
    }
    settings.update(
        text=text,
        source_revision=SOURCE_REVISION,
        weights_revision=WEIGHT_REVISION,
        reference_sha256=hashlib.sha256(a.reference_audio.read_bytes()).hexdigest(),
        checkpoint_sha256=file_digest(Path(weights["checkpoint"])),
        config_sha256=hashlib.sha256(Path(weights["model_config"]).read_bytes()).hexdigest(),
    )
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def generate_chunk(a: argparse.Namespace, plan: dict[str, Any], model: Any, torch: Any, np: Any) -> Any:
    torch.manual_seed(plan["chunk_seed"] + 2147483647)
    with torch.inference_mode():
        tokens = plan["tokens"].to(a.device)
        lengths = torch.tensor([tokens.shape[-1]], dtype=torch.long, device=a.device)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        text_encoded = model.text_encoder(tokens, lengths, mask)
        d = plan["duration_encoding"].to(a.device)
        # repeat_interleave is equivalent to the notebook's one-hot alignment
        # matrix multiplication while avoiding a quadratic alignment allocation.
        repeats = plan["durations"].to(a.device)
        prosody = torch.repeat_interleave(d.transpose(-1, -2), repeats, dim=2)
        asr = torch.repeat_interleave(text_encoded, repeats, dim=2)
        # Official HifiGAN adapter shifts conditioning right one frame.
        prosody = torch.cat((prosody[:, :, :1], prosody[:, :, :-1]), dim=2)
        asr = torch.cat((asr[:, :, :1], asr[:, :, :-1]), dim=2)
        style = plan["prosody_style"].to(a.device)
        pitch, energy = model.predictor.F0Ntrain(prosody, style)
        waveform = model.decoder(asr, pitch * a.pitch_scale, energy, plan["acoustic_style"].to(a.device))
        result = waveform.reshape(-1).cpu().numpy().copy()
    # Fixed architecture-level pulse workaround, never duration-fitting padding/cropping.
    if a.tail_trim_samples:
        result = result[: -a.tail_trim_samples]
    if len(result) != plan["native_samples"]:
        raise RuntimeError(
            f"HifiGAN length mismatch ({len(result)} vs {plan['native_samples']}); refusing to fit output length"
        )
    if not np.isfinite(result).all():
        raise RuntimeError("Model generated NaN/Inf audio")
    result = fade_edges(result, round(a.fade_ms * NATIVE_RATE / 1000), np)
    return result


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    a = parse_args(argv)
    text = (a.text if a.text is not None else a.text_file.read_text(encoding="utf-8-sig")).strip()
    word_count = len(re.findall(r"\S+", text))
    if not 10 <= word_count <= 500:
        raise ValueError(
            f"Requested evaluation range is 10..500 whitespace-separated words; got {word_count}"
        )
    check_source(a.repo.resolve())
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from tqdm.auto import tqdm

    if any((a.output_dir / name).exists() for name in ("manifest.json", "duration_plan.pt")) or any(
        a.output_dir.glob("sample_*.wav")
    ):
        raise FileExistsError(
            f"Output directory already contains run artifacts: {a.output_dir}; use a fresh --output-dir (also when reusing a plan)"
        )
    monitor = MemoryMonitor(torch, a.device, a.max_vram_gb)
    try:
        a.output_dir.mkdir(parents=True, exist_ok=True)
        model, sampler, weights = load_models(a, torch)
        reference = reference_style(a, model, torch, np)
        chunks = tokenize_chunks(a, text, torch)
        plan_fingerprint = fingerprint(a, text, weights)
        startup_seconds = time.perf_counter() - started
        planning_started = time.perf_counter()
        if a.reuse_plan:
            saved = torch.load(a.reuse_plan, map_location="cpu", weights_only=True)
            if saved["fingerprint"] != plan_fingerprint:
                raise ValueError("Cached plan settings/reference/checkpoint differ from this invocation")
            plans = saved["plans"]
        else:
            plans = build_plans(a, chunks, model, sampler, reference, torch, tqdm)
        preflight_seconds = time.perf_counter() - planning_started
        pause, overlap = round(a.pause_ms * NATIVE_RATE / 1000), int(a.crossfade_ms * NATIVE_RATE / 1000)
        previews = []
        for sample, plan in enumerate(plans):
            native_count = joined_length([chunk["native_samples"] for chunk in plan], pause, overlap)
            exported_count = (native_count * a.sample_rate + NATIVE_RATE - 1) // NATIVE_RATE
            previews.append(
                {
                    "sample": sample + 1,
                    "seed": a.seed + sample,
                    "native_samples": native_count,
                    "exported_samples": exported_count,
                    "seconds": exported_count / a.sample_rate,
                }
            )
        print(
            json.dumps(
                {
                    "preflight": previews,
                    "startup_seconds": startup_seconds,
                    "preflight_seconds": preflight_seconds,
                    "includes_style_diffusion": True,
                },
                indent=2,
            ),
            flush=True,
        )
        plan_file = a.output_dir / "duration_plan.pt"
        torch.save({"fingerprint": plan_fingerprint, "plans": plans}, plan_file)
        json_plans = [
            [
                {
                    k: v.tolist() if isinstance(v, torch.Tensor) and k == "durations" else v
                    for k, v in chunk.items()
                    if not isinstance(v, torch.Tensor) or k == "durations"
                }
                for chunk in plan
            ]
            for plan in plans
        ]
        manifest = {
            "model": "StyleTTS2-LibriTTS",
            "source_revision": SOURCE_REVISION,
            "weight_revision": WEIGHT_REVISION,
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            "weights": weights,
            "text": text,
            "word_count": word_count,
            "startup_seconds": startup_seconds,
            "preflight_seconds": preflight_seconds,
            "preview": previews,
            "plan": json_plans,
            "cached_plan": str(plan_file.resolve()),
            "plan_fingerprint": plan_fingerprint,
            "outputs": [],
            "memory": monitor.snapshot(),
        }
        manifest_file = a.output_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if a.estimate_only:
            return 0
        model.text_encoder.to(a.device)
        model.decoder.to(a.device)
        with tqdm(total=sum(map(len, plans)), desc="StyleTTS2 generated chunks", unit="chunk") as progress:
            for sample, plan in enumerate(plans):
                sample_started = time.perf_counter()
                waves = []
                for chunk in plan:
                    waves.append(generate_chunk(a, chunk, model, torch, np))
                    monitor.snapshot()
                    progress.update(1)
                final = waves[0]
                for wave in waves[1:]:
                    overlap_count = min(overlap, len(final), len(wave))
                    if overlap_count:
                        ramp = np.linspace(0, 1, overlap_count, dtype=np.float32)
                        final = np.concatenate(
                            (
                                final[:-overlap_count],
                                final[-overlap_count:] * (1 - ramp) + wave[:overlap_count] * ramp,
                                wave[overlap_count:],
                            )
                        )
                    else:
                        final = np.concatenate((final, np.zeros(pause, dtype=np.float32), wave))
                if a.sample_rate != NATIVE_RATE:
                    factor = math.gcd(NATIVE_RATE, a.sample_rate)
                    final = resample_poly(final, a.sample_rate // factor, NATIVE_RATE // factor).astype(
                        np.float32
                    )
                expected, measured = previews[sample]["exported_samples"], len(final)
                relative_error = abs(expected - measured) / measured
                if measured != expected:
                    raise RuntimeError(
                        f"Exact duration contract failed ({expected} planned versus {measured} produced; error {relative_error:.6%}); sample not exported"
                    )
                final, peak_normalization = limit_peak(final, a.peak_limit, np)
                monitor.snapshot()
                destination = a.output_dir / f"sample_{sample + 1:03d}_seed_{a.seed + sample}.wav"
                sf.write(destination, final, a.sample_rate, subtype=a.wav_subtype)
                info = sf.info(destination)
                if info.frames != measured or info.samplerate != a.sample_rate:
                    raise RuntimeError("WAV export length validation failed")
                manifest["outputs"].append(
                    {
                        "path": str(destination.resolve()),
                        "seed": a.seed + sample,
                        "peak_normalization": peak_normalization,
                        "predicted_samples": expected,
                        "measured_samples": info.frames,
                        "duration_seconds": info.duration,
                        "relative_error": relative_error,
                        "generation_seconds": time.perf_counter() - sample_started,
                        "absolute_peak_amplitude": float(np.max(np.abs(final))),
                    }
                )
                manifest["memory"] = monitor.snapshot()
                manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0
    finally:
        monitor.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

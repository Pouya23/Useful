#!/usr/bin/env python3
"""Independent Kokoro sampling and pre-waveform duration planning.

Python 3.12 is recommended. Install PyTorch for your platform, then:
  python -m pip install "git+https://github.com/hexgrad/kokoro.git@dfb907a02bba8152ca444717ca5d78747ccb4bec" "misaki[en]" transformers==5.3.0 scipy soundfile

Example:
  python kokoro_samples.py --text-file narration.txt --voice am_fenrir --speed 1.0 --n 3
  python kokoro_samples.py --text-file narration.txt --estimate-only

The preview runs G2P and the actual voice-conditioned duration network. It does
not run F0/noise prediction, the acoustic text encoder, or the waveform decoder.
Cached durations and embeddings are reused for synthesis. For the supported
checkpoint architecture, the decoder emits exactly 600 * sum(pred_dur) samples
at 24 kHz, including BOS/EOS silence. No energy-based silence removal is used.
The default is a male preset. Kokoro accepts preset/local voice embeddings, not
a reference WAV; there is no reference-audio speaker encoder in this checkpoint.
Chunk boundaries prefer complete sentences and clauses. Tiny configurable edge
fades suppress joining clicks without adding, removing, or moving samples.

CPU is the default and uses zero GPU VRAM. CUDA is opt-in: an allocator budget
and measured PyTorch peaks are reported. This cannot certify a hard process-wide
VRAM limit, because driver/context/non-PyTorch allocations are not controlled.
No inference library or another script in this directory is imported.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import random
import re
import sys
import time
import uuid
from pathlib import Path

NATIVE_RATE = 24_000
SAMPLES_PER_DURATION = 600
SOURCE_REVISION = "dfb907a02bba8152ca444717ca5d78747ccb4bec"
MODEL_REVISIONS = {
    "hexgrad/Kokoro-82M": "f3ff3571791e39611d31c381e3a41a3af07b4987",
    "hexgrad/Kokoro-82M-v1.1-zh": "01e7505bd6a7a2ac4975463114c3a7650a9f7218",
}
MODEL_FILES = {
    "hexgrad/Kokoro-82M": "kokoro-v1_0.pth",
    "hexgrad/Kokoro-82M-v1.1-zh": "kokoro-v1_1-zh.pth",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    text = p.add_mutually_exclusive_group()
    text.add_argument("--text", help="Literal UTF-8 text.")
    text.add_argument("--text-file", type=Path, help="UTF-8 or UTF-8-BOM text file.")
    p.add_argument("--config", type=Path, help="JSON object of CLI defaults; explicit flags take precedence.")
    p.add_argument("--n", type=int, default=1, help="Number of speech samples, with consecutive seeds.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--voice", default="am_fenrir", help="Preset, local .pt voice pack, or comma-separated presets."
    )
    p.add_argument(
        "--voice-weights", help="Comma-separated nonnegative mixing weights; default equal weights."
    )
    p.add_argument(
        "--language", choices=list("abefhijpz"), default="a", help="Kokoro language code; a=US English."
    )
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--repo-id", choices=list(MODEL_REVISIONS), default="hexgrad/Kokoro-82M")
    p.add_argument("--checkpoint", type=Path, help="Local PyTorch checkpoint instead of the pinned download.")
    p.add_argument(
        "--model-config", type=Path, help="Local model config; architecture must match the supported decoder."
    )
    p.add_argument("--cache-dir", type=Path, default=Path(__file__).parent / ".model_cache")
    p.add_argument("--device", default="cpu", help="cpu or cuda[:index].")
    p.add_argument(
        "--max-vram-gb", type=float, default=5.0, help="Decimal GB, at most 5; CUDA allocator target."
    )
    p.add_argument(
        "--cuda-reserve-mb", type=int, default=768, help="Reserve outside the PyTorch allocator target."
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--chunk-words", type=int, default=35, help="Initial text chunk limit; long chunks are split again."
    )
    p.add_argument("--max-chunk-seconds", type=float, default=20.0)
    p.add_argument(
        "--sample-rate",
        type=int,
        default=NATIVE_RATE,
        help="Export sample rate; native synthesis stays 24 kHz.",
    )
    p.add_argument(
        "--pause-ms", type=float, default=0.0, help="Explicit silence between final planned chunks."
    )
    p.add_argument("--gain-db", type=float, default=0.0)
    p.add_argument(
        "--edge-fade-ms",
        type=float,
        default=3.0,
        help="Cosine fade at each chunk edge; preserves sample count; 0 disables it.",
    )
    p.add_argument(
        "--normalize-peak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce peaks above 0.98 without changing length.",
    )
    p.add_argument("--subtype", choices=["PCM_16", "PCM_24", "FLOAT"], default="PCM_16")
    p.add_argument("--output-dir", type=Path, default=Path("samples"))
    p.add_argument(
        "--estimate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write a preview without generating any waveform.",
    )
    p.add_argument(
        "--self-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run integer-length checks without models or third-party imports.",
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    preliminary, _ = p.parse_known_args(raw)
    if preliminary.config:
        defaults = json.loads(preliminary.config.read_text(encoding="utf-8-sig"))
        if not isinstance(defaults, dict):
            p.error("--config must contain a JSON object")
        actions = {a.dest: a for a in p._actions if a.dest not in {"help", "config"}}
        unknown = set(defaults) - set(actions)
        if unknown:
            p.error(f"Unknown configuration keys: {sorted(unknown)}")
        for key, value in defaults.items():
            action = actions[key]
            is_boolean = isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction))
            if is_boolean and not isinstance(value, bool):
                p.error(f"{key} must be a JSON boolean")
            if action.type is int and (isinstance(value, bool) or not isinstance(value, int)):
                p.error(f"{key} must be a JSON integer")
            if action.type is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
                p.error(f"{key} must be a JSON number")
            if action.type is Path and value is not None and not isinstance(value, str):
                p.error(f"{key} must be a JSON path string")
            if action.type is None and not is_boolean and value is not None and not isinstance(value, str):
                p.error(f"{key} must be a JSON string")
            if value is not None and action.type:
                try:
                    value = action.type(value)
                except (TypeError, ValueError) as exc:
                    p.error(f"Invalid {key}: {exc}")
            if action.choices is not None and value not in action.choices:
                p.error(f"Invalid {key}: {value!r}")
            defaults[key] = value
        p.set_defaults(**defaults)
        if any(flag == "--text" or flag.startswith("--text=") for flag in raw):
            p.set_defaults(text_file=None)
        if any(flag == "--text-file" or flag.startswith("--text-file=") for flag in raw):
            p.set_defaults(text=None)
    args = p.parse_args(raw)
    if args.self_test:
        return args
    if (args.text is None) == (args.text_file is None):
        p.error("Supply exactly one of --text and --text-file")
    if args.n < 1 or args.threads < 1 or args.chunk_words < 1:
        p.error("--n, --threads and --chunk-words must be positive")
    if args.seed < 0 or args.seed + args.n - 1 >= 2**63:
        p.error("Consecutive sample seeds must fit a nonnegative signed 64-bit integer")
    for name in ("speed", "max_vram_gb", "max_chunk_seconds", "pause_ms", "gain_db", "edge_fade_ms"):
        if not math.isfinite(getattr(args, name)):
            p.error(f"{name} must be finite")
    if args.speed <= 0 or args.max_chunk_seconds <= 0 or not 0 < args.max_vram_gb <= 5:
        p.error("Positive speed/chunk duration and 0 < max-vram-gb <= 5 are required")
    if args.pause_ms < 0 or not 8_000 <= args.sample_rate <= 192_000 or args.cuda_reserve_mb < 0:
        p.error("Invalid pause, sample rate, or CUDA reserve")
    if abs(args.gain_db) > 120:
        p.error("--gain-db must be between -120 and 120")
    if not 0 <= args.edge_fade_ms <= 20:
        p.error("--edge-fade-ms must be between 0 and 20")
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", args.device):
        p.error("--device must be cpu or cuda[:index]")
    return args


def export_length(native_samples: int, rate: int) -> int:
    """Match scipy.signal.resample_poly's exact ceiling, using integer arithmetic."""
    return (native_samples * rate + NATIVE_RATE - 1) // NATIVE_RATE


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


def initial_chunks(text: str, limit: int) -> list[str]:
    """Keep the hard word bound while preferring complete sentences/clauses."""
    if limit < 1:
        raise ValueError("Chunk word limit must be positive")
    chunks = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        while words:
            end = min(limit, len(words))
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


def edge_fade(audio, sample_rate: int, milliseconds: float):
    """Suppress hard chunk joins with a short, sample-count-preserving taper."""
    import numpy as np

    result = np.asarray(audio, dtype=np.float32).copy()
    count = min(round(sample_rate * milliseconds / 1000), len(result) // 2)
    if count > 1:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, count))).astype(np.float32)
        result[:count] *= ramp
        result[-count:] *= ramp[::-1]
    return result


def validate_english_tokens(tokens: list, maximum_phonemes: int) -> None:
    """Reject skipped words and indivisible phoneme overflow before chunking.

    Misaki can return empty phonemes when no OOD fallback is available. Kokoro's
    public pipeline wrappers also truncate overlong phoneme strings. The direct
    en_tokenize path used below does not truncate, but an overlong individual
    token must be caught before its empty boundary chunk can be emitted.
    """
    for token in tokens:
        text = token.text or ""
        phonemes = token.phonemes or ""
        if any(character.isalnum() for character in text) and not phonemes.strip():
            raise ValueError(
                f"G2P produced no pronunciation for {text!r}; install/configure eSpeak fallback or correct the text."
            )
        if len(phonemes.strip()) > maximum_phonemes:
            raise ValueError(
                f"One word's pronunciation exceeds the phoneme context: {text!r}; refusing to truncate speech."
            )


def progress(done: int, total: int, label: str) -> None:
    fraction = done / max(1, total)
    width = 24
    bar = "#" * int(width * fraction)
    sys.stderr.write(f"\r{label}: [{bar:<{width}}] {done}/{total}")
    if done == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def self_test() -> None:
    assert export_length(600, 24_000) == 600
    assert export_length(601, 16_000) == 401
    assert export_length(600, 44_100) == 1103
    assert initial_chunks("one two three\nfour five", 2) == ["one two", "three", "four five"]
    for duration in range(1, 1000):
        assert SAMPLES_PER_DURATION * duration / NATIVE_RATE == duration / 40
    print("Kokoro integer sample planning checks passed.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    started = time.perf_counter()
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from huggingface_hub import hf_hub_download
        from kokoro import KModel, KPipeline
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError(
            "Install the dependencies shown in this file's docstring into the active environment."
        ) from exc

    distribution = importlib.metadata.distribution("kokoro")
    direct_url = distribution.read_text("direct_url.json")
    installed_revision = json.loads(direct_url).get("vcs_info", {}).get("commit_id") if direct_url else None
    if installed_revision != SOURCE_REVISION:
        raise RuntimeError(
            f"Install the pinned Kokoro source revision {SOURCE_REVISION}; found {installed_revision!r}."
        )
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    budget_bytes = int(args.max_vram_gb * 1_000_000_000)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
        torch.cuda.set_device(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocator_bytes = min(budget_bytes, total_memory) - args.cuda_reserve_mb * 1_000_000
        if allocator_bytes <= 0:
            raise RuntimeError("CUDA reserve leaves no allocator budget.")
        torch.cuda.set_per_process_memory_fraction(allocator_bytes / total_memory, device)
        torch.cuda.reset_peak_memory_stats(device)

    def memory_report() -> dict:
        if device.type != "cuda":
            return {"device": "cpu", "gpu_vram_bytes": 0, "hard_gpu_limit_satisfied": True}
        torch.cuda.synchronize(device)
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        if max(allocated, reserved) > budget_bytes:
            raise RuntimeError("Measured PyTorch CUDA peak exceeded the requested budget.")
        return {
            "device": str(device),
            "torch_peak_allocated_bytes": allocated,
            "torch_peak_reserved_bytes": reserved,
            "requested_budget_bytes": budget_bytes,
            "hard_process_gpu_limit_certified": False,
            "note": "Allocator cap and PyTorch peaks exclude driver/context and other allocations.",
        }

    revision = MODEL_REVISIONS[args.repo_id]

    def download(filename: str) -> str:
        return hf_hub_download(args.repo_id, filename, revision=revision, cache_dir=args.cache_dir)

    config_path = args.model_config or Path(download("config.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decoder = config.get("istftnet", {})
    if (
        decoder.get("upsample_rates") != [10, 6]
        or decoder.get("upsample_kernel_sizes") != [20, 12]
        or decoder.get("gen_istft_hop_size") != 5
        or decoder.get("gen_istft_n_fft") != 20
    ):
        raise RuntimeError(
            "Unsupported decoder configuration; the 600-sample duration mapping is not certified."
        )
    checkpoint = args.checkpoint or Path(download(MODEL_FILES[args.repo_id]))
    model = KModel(repo_id=args.repo_id, config=config, model=str(checkpoint)).to(device).eval()
    pipeline = KPipeline(lang_code=args.language, repo_id=args.repo_id, model=False)
    voice_names = [name.strip() for name in args.voice.split(",") if name.strip()]
    if not voice_names:
        raise ValueError("Supply at least one voice.")
    weights = (
        [float(x) for x in args.voice_weights.split(",")] if args.voice_weights else [1.0] * len(voice_names)
    )
    if (
        len(weights) != len(voice_names)
        or any(not math.isfinite(x) or x < 0 for x in weights)
        or not math.isfinite(sum(weights))
        or sum(weights) <= 0
    ):
        raise ValueError(
            "Voice weights must match voices, be finite/nonnegative, and sum to a positive value."
        )
    packs, voice_hashes = [], {}
    for name in voice_names:
        voice_path = Path(name) if name.endswith(".pt") else Path(download(f"voices/{name}.pt"))
        pack = torch.load(voice_path, map_location="cpu", weights_only=True)
        if not isinstance(pack, torch.Tensor) or pack.ndim != 3 or pack.shape[1:] != (1, 256):
            raise RuntimeError(f"Unexpected voice pack shape for {name}: {getattr(pack, 'shape', None)}")
        if not torch.isfinite(pack).all():
            raise RuntimeError(f"Nonfinite voice embedding in {name}")
        packs.append(pack)
        voice_hashes[name] = hashlib.sha256(voice_path.read_bytes()).hexdigest()
    if any(p.shape != packs[0].shape for p in packs):
        raise ValueError("Voice packs must have identical shapes to mix them.")
    voice_pack = sum(p * (w / sum(weights)) for p, w in zip(packs, weights)).to(device)
    text = args.text_file.read_text(encoding="utf-8-sig") if args.text_file else args.text
    if not text or not text.strip():
        raise ValueError("Text is empty.")
    startup_seconds = time.perf_counter() - started
    run_dir = args.output_dir / f"kokoro_{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    preflight_start = time.perf_counter()

    @torch.inference_mode()
    def plan_one(graphemes: str, phonemes: str) -> dict:
        unknown = sorted(
            {phoneme for phoneme in phonemes if phoneme not in model.vocab and not phoneme.isspace()}
        )
        if unknown:
            raise ValueError(
                f"G2P returned unsupported phonemes {unknown!r}; refusing to silently drop pronunciation symbols."
            )
        ids = [model.vocab[p] for p in phonemes if p in model.vocab]
        if not ids or len(ids) + 2 > model.context_length or len(phonemes) > voice_pack.shape[0]:
            raise ValueError("Phoneme sequence exceeds the model/voice context; use smaller chunks.")
        input_ids = torch.tensor([[0, *ids, 0]], dtype=torch.long, device=device)
        lengths = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        ref = voice_pack[len(phonemes) - 1]
        bert = model.bert(input_ids, attention_mask=(~mask).int())
        encoded = model.bert_encoder(bert).transpose(-1, -2)
        d = model.predictor.text_encoder(encoded, ref[:, 128:], lengths, mask)
        x, _ = model.predictor.lstm(d)
        durations = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1) / args.speed
        durations = torch.round(durations).clamp(min=1).long().reshape(-1)
        frames = int(durations.sum().item())
        return {
            "text": graphemes,
            "phonemes": phonemes,
            "ids": input_ids.cpu(),
            "durations": durations.cpu(),
            "embedding": d.cpu(),
            "ref": ref.cpu(),
            "samples": frames * SAMPLES_PER_DURATION,
        }

    def plan_text(chunk: str) -> list[dict]:
        if args.language in "ab":
            _, tokens = pipeline.g2p(chunk)
            validate_english_tokens(tokens, min(510, model.context_length - 2, voice_pack.shape[0]))
            phoneme_chunks = list(pipeline.en_tokenize(tokens))
            sequences = [(gs, ps) for gs, ps, _ in phoneme_chunks]
        else:
            ps, _ = pipeline.g2p(chunk)
            sequences = [(chunk, ps)]
        result = []
        for gs, ps in sequences:
            if not ps:
                raise ValueError(
                    f"No phonemes were produced for {gs!r}; check language and pronunciation dependencies."
                )
            if len(ps) > min(model.context_length - 2, voice_pack.shape[0]):
                words = gs.split()
                if len(words) < 2:
                    raise ValueError(
                        "A single token exceeds the phoneme context and cannot be safely truncated."
                    )
                for smaller in initial_chunks(gs, max(1, len(words) // 2)):
                    result.extend(plan_text(smaller))
                continue
            item = plan_one(gs, ps)
            if item["samples"] / NATIVE_RATE > args.max_chunk_seconds:
                words = gs.split()
                if len(words) < 2:
                    raise ValueError(
                        "A single token exceeds --max-chunk-seconds; increase speed or adjust the limit."
                    )
                for smaller in initial_chunks(gs, max(1, len(words) // 2)):
                    result.extend(plan_text(smaller))
            else:
                result.append(item)
        return result

    chunks = initial_chunks(text, args.chunk_words)
    plan = []
    for index, chunk in enumerate(chunks, 1):
        plan.extend(plan_text(chunk))
        progress(index, len(chunks), "Duration preflight")
    if not plan:
        raise RuntimeError("No valid speech chunks were planned.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preflight_seconds = time.perf_counter() - preflight_start
    pause_samples = round(args.pause_ms * NATIVE_RATE / 1000)
    native_samples = sum(item["samples"] for item in plan) + pause_samples * (len(plan) - 1)
    predicted_samples = export_length(native_samples, args.sample_rate)
    preview = {
        "model": args.repo_id,
        "source_revision": SOURCE_REVISION,
        "weights_revision": revision,
        "local_checkpoint_override": str(args.checkpoint) if args.checkpoint else None,
        "settings": vars(args),
        "text": text,
        "voice_sha256": voice_hashes,
        "native_sample_rate": NATIVE_RATE,
        "export_sample_rate": args.sample_rate,
        "predicted_native_samples": native_samples,
        "predicted_export_samples": predicted_samples,
        "predicted_seconds": predicted_samples / args.sample_rate,
        "startup_seconds": startup_seconds,
        "preflight_seconds": preflight_seconds,
        "estimator_kind": "cached learned duration plan; deterministic decoder sample mapping",
        "chunks": [
            {
                "text": i["text"],
                "phonemes": i["phonemes"],
                "samples": i["samples"],
                "durations": i["durations"].tolist(),
            }
            for i in plan
        ],
        "memory": memory_report(),
        "samples": [],
    }
    write_json(run_dir / "preview.json", preview)
    print(
        json.dumps(
            {
                "predicted_seconds": preview["predicted_seconds"],
                "preflight_seconds": preflight_seconds,
                "startup_seconds": startup_seconds,
                "preview": str(run_dir / "preview.json"),
            }
        ),
        flush=True,
    )
    if args.estimate_only:
        return 0

    @torch.inference_mode()
    def synthesize(item: dict):
        ids = item["ids"].to(device)
        durations = item["durations"].to(device)
        d, ref = item["embedding"].to(device), item["ref"].to(device)
        indices = torch.repeat_interleave(torch.arange(ids.shape[1], device=device), durations)
        alignment = torch.zeros((ids.shape[1], indices.numel()), dtype=d.dtype, device=device)
        alignment[indices, torch.arange(indices.numel(), device=device)] = 1
        alignment = alignment.unsqueeze(0)
        en = d.transpose(-1, -2) @ alignment
        f0, noise = model.predictor.F0Ntrain(en, ref[:, 128:])
        lengths = torch.tensor([ids.shape[1]], dtype=torch.long, device=device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        acoustic = model.text_encoder(ids, lengths, mask) @ alignment
        audio = model.decoder(acoustic, f0, noise, ref[:, :128]).reshape(-1).cpu().numpy()
        if len(audio) != item["samples"] or not np.isfinite(audio).all():
            raise RuntimeError(
                "Decoder violated the planned sample count or produced nonfinite audio; no export."
            )
        memory_report()
        return edge_fade(audio, NATIVE_RATE, args.edge_fade_ms)

    manifest = dict(preview)
    progress(0, args.n * len(plan), "Speech generation")
    for sample_index in range(args.n):
        seed = args.seed + sample_index
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        generation_start = time.perf_counter()
        audio_chunks = []
        for index, item in enumerate(plan):
            if index and pause_samples:
                audio_chunks.append(np.zeros(pause_samples, dtype=np.float32))
            audio_chunks.append(synthesize(item))
            progress(sample_index * len(plan) + index + 1, args.n * len(plan), "Speech generation")
        waveform = np.concatenate(audio_chunks).astype(np.float32, copy=False)
        if args.sample_rate != NATIVE_RATE:
            divisor = math.gcd(args.sample_rate, NATIVE_RATE)
            waveform = resample_poly(waveform, args.sample_rate // divisor, NATIVE_RATE // divisor)
        waveform = waveform * (10 ** (args.gain_db / 20))
        peak = float(np.max(np.abs(waveform)))
        if args.normalize_peak and peak > 0.98:
            waveform *= 0.98 / peak
        if len(waveform) != predicted_samples or not np.isfinite(waveform).all():
            raise RuntimeError("Export transformations violated the planned sample count; no export.")
        destination = run_dir / f"sample_{sample_index + 1:03d}_seed_{seed}.wav"
        temporary = destination.with_name(destination.stem + ".temporary.wav")
        sf.write(temporary, waveform, args.sample_rate, subtype=args.subtype, format="WAV")
        info = sf.info(temporary)
        measured_seconds = info.frames / info.samplerate
        error = abs(info.frames - predicted_samples) / predicted_samples
        if info.samplerate != args.sample_rate or error >= 0.01 or info.frames != predicted_samples:
            temporary.unlink()
            raise RuntimeError("Written WAV failed exact duration verification.")
        temporary.replace(destination)
        manifest["samples"].append(
            {
                "path": str(destination),
                "seed": seed,
                "measured_samples": info.frames,
                "measured_seconds": measured_seconds,
                "relative_duration_error": error,
                "generation_seconds": time.perf_counter() - generation_start,
                "output_levels": {
                    "peak_before_protection": peak,
                    "peak_scale": 0.98 / peak if args.normalize_peak and peak > 0.98 else 1.0,
                    "peak_after_protection": float(np.max(np.abs(waveform))),
                },
                "memory": memory_report(),
            }
        )
        write_json(run_dir / "manifest.json", manifest)
    print(f"Saved {args.n} samples and verified exact exported durations in {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error

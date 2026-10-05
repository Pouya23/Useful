#!/usr/bin/env python3
"""Standalone EmotiVoice sampler with cached phoneme durations and exact export timing.

Use Python 3.11 with the official repository's requirements plus scipy. Clone
https://github.com/netease-youdao/EmotiVoice.git and check out
59f0f36de4db12825f4705dd4e0780d79dd6bb01. Download BOTH generator and style encoder
checkpoints according to https://github.com/netease-youdao/EmotiVoice/wiki/Pretrained-models.
The tokenizer/SimBERT model is also required (default WangZeJun/simbert-base-chinese).

Example:
  python emotivoice_samples.py --repo /path/to/EmotiVoice \
    --generator-checkpoint /path/to/g_00140000 \
    --style-checkpoint /path/to/checkpoint_163431 --speaker 9017 \
    --prompt Neutral --speed 1.0 --text-file narration.txt --n 3

--speaker is a label in data/youdao/text/speaker2, not an embedding index.
--speaker-index explicitly chooses an embedding index instead. Emotion/style
prompts are approximate model controls. Speed is implemented by editing the
predicted integer phoneme durations before Gaussian upsampling, because the
upstream inference path ignores its alpha argument. Predictor pitch/energy
scales are neural-space controls, not semitone/dB controls.

Optional --emphasis-json is a list of {"chunk": 0, "phoneme": 2,
"duration_scale": 1.2, "pitch_scale": 1.1, "energy_scale": 1.2}. Indices refer
to the displayed/manifested chunk phonemes, including boundary tokens.
The default speaker is the official male preset 9017 (John Van Stan) with a
neutral style prompt. A reference WAV cannot clone a voice through this pretrained
embedding table; adding a new speaker requires the upstream training workflow.
Punctuation-aware chunks, short edge fades, and whole-recording peak attenuation
avoid avoidable joining/clipping artifacts without altering the duration plan.

CPU is the default and guarantees zero GPU VRAM. CUDA is optional and requires
--allow-unverified-cuda-budget: its conservative PyTorch allocator budget is
not a hard bound on driver/context/external allocations. Only CPU fulfills a
strict hardware-independent <=5 GB guarantee. Preview runs the text/style/
duration predictors only and caches the plan for waveform synthesis. Cold
startup and warm preview timings are reported independently; instant latency
cannot be guaranteed without benchmarking your actual hardware.

The model is deterministic in eval mode: n repeats deliberately have identical
audio for identical configurations. Changing random seeds adds no diversity.
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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SOURCE_REVISION = "59f0f36de4db12825f4705dd4e0780d79dd6bb01"
NATIVE_RATE = 16000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="JSON defaults; CLI overrides; keys use underscores")
    text = parser.add_mutually_exclusive_group()
    text.add_argument("--text")
    text.add_argument("--text-file", type=Path)
    parser.add_argument("--repo", type=Path, help="Pinned official source checkout")
    parser.add_argument("--generator-checkpoint", type=Path)
    parser.add_argument("--style-checkpoint", type=Path)
    parser.add_argument("--model-yaml", type=Path, help="Defaults to repo/config/joint/config.yaml")
    parser.add_argument(
        "--bert-path",
        default="WangZeJun/simbert-base-chinese",
        help="Local SimBERT directory or Hugging Face model ID",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not download SimBERT/tokenizer files",
    )
    speaker = parser.add_mutually_exclusive_group()
    speaker.add_argument("--speaker", default=None, help="Exact speaker label; default male speaker 9017")
    speaker.add_argument("--speaker-index", type=int)
    parser.add_argument("--prompt", default="Neutral", help="Style/emotion conditioning text")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Higher is faster; integer duration scaling before decoding"
    )
    parser.add_argument("--pitch-predictor-scale", type=float, default=1.0)
    parser.add_argument("--energy-predictor-scale", type=float, default=1.0)
    parser.add_argument(
        "--min-phoneme-frames",
        type=int,
        default=1,
        help="Minimum acoustic frames for lexical phones after speed scaling; 0 restores upstream behavior",
    )
    parser.add_argument(
        "--emphasis-json", type=Path, help="Phoneme-indexed duration/pitch/energy scale edits"
    )
    parser.add_argument("--max-chunk-chars", type=int, default=240)
    parser.add_argument(
        "--max-chunk-frames",
        type=int,
        default=2000,
        help="Reject oversized acoustic allocations before generation",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument(
        "--allow-unverified-cuda-budget", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output-dir", type=Path, default=Path("emotivoice_samples"))
    parser.add_argument("--sample-rate", type=int, default=0, help="0 uses native 16000 Hz")
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
    parser.add_argument("--max-vram-gb", type=float, default=5.0, help="Decimal GB, at most 5")
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
                except (TypeError, ValueError) as exc:
                    parser.error(f"Invalid JSON value for {key}: {exc}")
        if early.text is not None:
            defaults["text_file"] = None
        elif early.text_file is not None:
            defaults["text"] = None
        if early.speaker is not None:
            defaults["speaker_index"] = None
        elif early.speaker_index is not None:
            defaults["speaker"] = None
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if (args.text is None) == (args.text_file is None):
        parser.error("Specify exactly one of --text or --text-file, including JSON defaults")
    if args.speaker is not None and args.speaker_index is not None:
        parser.error("Specify a speaker label or a speaker index, not both")
    if any(getattr(args, field) is None for field in ("repo", "generator_checkpoint", "style_checkpoint")):
        parser.error("--repo, --generator-checkpoint and --style-checkpoint are required")
    for name in ("n", "threads", "max_chunk_chars", "max_chunk_frames"):
        if not isinstance(getattr(args, name), int) or getattr(args, name) < 1:
            parser.error(f"{name} must be a positive integer")
    if args.max_chunk_chars < 20:
        parser.error("--max-chunk-chars must be at least 20")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.speaker_index is not None and args.speaker_index < 0:
        parser.error("--speaker-index must be nonnegative")
    if not 0 <= args.min_phoneme_frames <= 10:
        parser.error("--min-phoneme-frames must be between 0 and 10")
    for name in ("speed", "pitch_predictor_scale", "energy_predictor_scale"):
        if not math.isfinite(getattr(args, name)) or not 0.25 <= getattr(args, name) <= 4:
            parser.error(f"{name} must be in [0.25, 4]; extreme controls may impair audio quality")
    if not math.isfinite(args.pause_ms) or args.pause_ms < 0:
        parser.error("--pause-ms must be finite and nonnegative")
    if not math.isfinite(args.edge_fade_ms) or not 0 <= args.edge_fade_ms <= 20:
        parser.error("--edge-fade-ms must be between 0 and 20")
    if not math.isfinite(args.gain_db) or abs(args.gain_db) > 120:
        parser.error("--gain-db must be finite and between -120 and 120")
    if not math.isfinite(args.max_vram_gb) or not 0 < args.max_vram_gb <= 5:
        parser.error("--max-vram-gb must be in (0, 5] decimal GB")
    if args.sample_rate and not 8000 <= args.sample_rate <= 192000:
        parser.error("--sample-rate must be 0 or 8000..192000 Hz")
    if args.device != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", args.device):
        parser.error("--device must be cpu or cuda[:index]")
    if args.device != "cpu" and not args.allow_unverified_cuda_budget:
        parser.error(
            "CUDA cannot guarantee total peak VRAM <=5 GB. Use cpu or explicitly --allow-unverified-cuda-budget"
        )
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


def is_lexical_phoneme(phoneme: str) -> bool:
    """Separate spoken English/Mandarin phones from boundaries and pause tags."""
    if phoneme == "_" or phoneme.startswith("<") or "sp" in phoneme:
        return False
    return any(character.isalpha() for character in phoneme)


class GuardedEnglishG2P:
    """Repair the pinned frontend's out-of-lexicon word boundary contract.

    get_eng_phoneme unconditionally pops its last item when punctuation follows.
    Its lexicon branch appends engsp1, but its G2P branch only appends that tag
    upon receiving a literal space. Supply that space after a spoken fallback
    word so punctuation replaces a pause rather than deleting its final phone.
    The pinned bilingual frontend continues to handle Chinese and normalization.
    """

    def __init__(self, backend: Any):
        self.backend = backend

    def __call__(self, text: str) -> list[str]:
        phones = list(self.backend(text))
        if any(character.isalnum() for character in text):
            if not phones or not any(re.fullmatch(r"[A-Z]+[0-2]?", phone) for phone in phones):
                raise ValueError(
                    f"G2P produced no spoken English phones for {text!r}; refusing to drop a word"
                )
        if phones and phones[0].isalnum() and phones[-1] != " ":
            phones.append(" ")
        return phones


def apply_duration_controls(
    base: Any, factors: Any, phonemes: list[str], minimum_frames: int
) -> tuple[Any, list[int], list[int]]:
    """Plan once, preserving short lexical phones instead of rounding them away."""
    import torch

    durations = torch.round(base.float() * factors).clamp_min(0).long()
    lexical = torch.tensor([[is_lexical_phoneme(phone) for phone in phonemes]], device=durations.device)
    zero_indices = torch.nonzero((durations == 0) & lexical, as_tuple=False)[:, 1].cpu().tolist()
    changed = torch.nonzero((durations < minimum_frames) & lexical, as_tuple=False)[:, 1].cpu().tolist()
    durations = torch.where(lexical, durations.clamp_min(minimum_frames), durations)
    return durations, zero_indices, changed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def repository_context(repo: Path) -> Iterator[None]:
    previous = Path.cwd()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    try:
        yield
    finally:
        os.chdir(previous)
        sys.path.remove(str(repo))


def verify_source(repo: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(
            f"Expected EmotiVoice source {SOURCE_REVISION}; got {revision}. Check out the pinned revision."
        )
    relevant = ["models", "frontend.py", "frontend_en.py", "frontend_cn.py", "config/joint/config.py"]
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", *relevant],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "Pinned inference/frontend source has local changes; refusing an unverified adapter"
        )


@dataclass(frozen=True)
class ChunkPlan:
    text: str
    phonemes: list[str]
    conditioned_tokens: Any
    mask: Any
    durations: Any
    pitch_factors: Any
    energy_factors: Any
    frames: int
    native_samples: int
    zero_duration_lexical_indices: list[int]
    minimum_duration_clamped_indices: list[int]


def conv_output_length(length: int, module: Any, transposed: bool = False) -> int:
    kernel, stride, padding, dilation = (
        int(getattr(module, name)[0]) for name in ("kernel_size", "stride", "padding", "dilation")
    )
    if transposed:
        return (
            (length - 1) * stride - 2 * padding + dilation * (kernel - 1) + int(module.output_padding[0]) + 1
        )
    return (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def vocoder_samples(frames: int, vocoder: Any) -> int:
    """Propagate convolution lengths through the verified upstream HiFiGAN path."""
    length = conv_output_length(frames, vocoder.conv_pre)
    for index, upsample in enumerate(vocoder.ups):
        length = conv_output_length(length, upsample, transposed=True)
        for block in vocoder.resblocks[index * vocoder.num_kernels : (index + 1) * vocoder.num_kernels]:
            if hasattr(block, "convs1"):
                for conv1, conv2 in zip(block.convs1, block.convs2):
                    if conv_output_length(conv_output_length(length, conv1), conv2) != length:
                        raise RuntimeError("Unsupported residual block length change")
            else:
                for convolution in block.convs:
                    if conv_output_length(length, convolution) != length:
                        raise RuntimeError("Unsupported residual block length change")
    return conv_output_length(length, vocoder.conv_post)


class Engine:
    def __init__(self, args: argparse.Namespace):
        if args.local_files_only:
            # huggingface_hub reads this at import time; StyleEncoder's internal
            # AutoModel constructor does not expose a local_files_only argument.
            os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        from transformers import AutoTokenizer
        from yacs import config as yacs_config

        self.torch = torch
        self.args = args
        self.repo = args.repo.resolve()
        verify_source(self.repo)
        torch.set_num_threads(args.threads)
        self.device = torch.device(args.device)
        self.budget_bytes = int(args.max_vram_gb * 1_000_000_000)
        self.allocator_limit_bytes = 0
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but this PyTorch installation has no available CUDA device"
                )
            torch.cuda.set_device(self.device)
            device_total = torch.cuda.get_device_properties(self.device).total_memory
            reserve = 768 * 1024 * 1024
            self.allocator_limit_bytes = min(self.budget_bytes, device_total) - reserve
            if self.allocator_limit_bytes <= 0:
                raise RuntimeError(
                    "The requested CUDA budget is smaller than the reserved context/workspace margin"
                )
            torch.cuda.set_per_process_memory_fraction(self.allocator_limit_bytes / device_total, self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.backends.cudnn.benchmark = False
        with repository_context(self.repo):
            from frontend import g2p_cn_en
            from frontend_en import G2p, read_lexicon
            from models.prompt_tts_modified.jets import JETSGenerator
            from models.prompt_tts_modified.simbert import StyleEncoder

            config_spec = importlib.util.spec_from_file_location(
                "emotivoice_config", self.repo / "config" / "joint" / "config.py"
            )
            if config_spec is None or config_spec.loader is None:
                raise RuntimeError("Cannot import the official EmotiVoice configuration")
            config_module = importlib.util.module_from_spec(config_spec)
            config_spec.loader.exec_module(config_module)
            cfg = config_module.Config()
            cfg.bert_path = args.bert_path
            # StyleEncoder constructs AutoModel internally. Keep ALL BERT inference on CPU.
            style_encoder = StyleEncoder(cfg).cpu().eval()
            style_state = torch.load(args.style_checkpoint.resolve(), map_location="cpu", weights_only=True)
            if not isinstance(style_state, dict) or "model" not in style_state:
                raise RuntimeError("Expected style checkpoint containing the 'model' state dictionary")
            cleaned = {key.removeprefix("module."): value for key, value in style_state["model"].items()}
            missing, unexpected = style_encoder.load_state_dict(cleaned, strict=False)
            if any(name.startswith("bert.") for name in missing):
                raise RuntimeError(f"Style checkpoint is missing BERT parameters: {missing}")
            self.style_load_report = {
                "missing_unused_head_keys": list(missing),
                "unexpected_keys": list(unexpected),
            }
            self.style_encoder = style_encoder
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.bert_path, local_files_only=args.local_files_only
            )
            yaml_path = (
                args.model_yaml.resolve()
                if args.model_yaml
                else self.repo / "config" / "joint" / "config.yaml"
            )
            with yaml_path.open(encoding="utf-8") as stream:
                conf = yacs_config.load_cfg(stream)
            conf.n_vocab = cfg.n_symbols
            conf.n_speaker = cfg.speaker_n_labels
            if int(conf.sr) != NATIVE_RATE:
                raise RuntimeError(
                    "This adapter supports the official 16000 Hz EmotiVoice checkpoint configuration"
                )
            model = JETSGenerator(conf).cpu().eval()
            checkpoint = torch.load(
                args.generator_checkpoint.resolve(), map_location="cpu", weights_only=True
            )
            if not isinstance(checkpoint, dict) or "generator" not in checkpoint:
                raise RuntimeError(
                    "Expected generator checkpoint containing the 'generator' state dictionary"
                )
            # Old and new torch weight_norm names have identical tensor semantics.
            expected_keys = set(model.state_dict())
            converted = {}
            for key, value in checkpoint["generator"].items():
                candidate = key.removeprefix("module.")
                if candidate not in expected_keys:
                    alternate = candidate.replace(".weight_g", ".parametrizations.weight.original0").replace(
                        ".weight_v", ".parametrizations.weight.original1"
                    )
                    if alternate in expected_keys:
                        candidate = alternate
                converted[candidate] = value
            model.load_state_dict(converted, strict=True)
            self.model = model.to(self.device)
            self.am = self.model.am
            self.g2p = GuardedEnglishG2P(G2p())
            self.lexicon = read_lexicon(str(self.repo / "lexicon" / "librispeech-lexicon.txt"))
            self.frontend = g2p_cn_en
            self.tokens = {
                token.strip(): index
                for index, token in enumerate(
                    Path(cfg.token_list_path).read_text(encoding="utf-8").splitlines()
                )
            }
            self.speakers = Path(cfg.speaker2id_path).read_text(encoding="utf-8").splitlines()
        if args.speaker_index is not None:
            self.speaker_index = args.speaker_index
        elif args.speaker is None:
            # The official speaker wiki identifies 9017 as male John Van Stan,
            # with a rich/resonant voice. A custom checkpoint must choose its
            # own label explicitly if this pretrained speaker is unavailable.
            try:
                self.speaker_index = self.speakers.index("9017")
            except ValueError as exc:
                raise ValueError(
                    "Default male speaker 9017 is absent; supply --speaker or --speaker-index"
                ) from exc
        else:
            try:
                self.speaker_index = self.speakers.index(args.speaker)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown speaker label {args.speaker!r}; see repo/data/youdao/text/speaker2"
                ) from exc
        if not 0 <= self.speaker_index < len(self.speakers):
            raise ValueError("--speaker-index exceeds the available embedding table")
        self.emphasis = self.read_emphasis(args.emphasis_json)
        self.weights = {
            "generator": sha256(args.generator_checkpoint),
            "style_encoder": sha256(args.style_checkpoint),
        }
        self.synchronize()

    @staticmethod
    def read_emphasis(path: Path | None) -> list[dict[str, Any]]:
        edits = json.loads(path.read_text(encoding="utf-8")) if path else []
        if not isinstance(edits, list):
            raise ValueError("--emphasis-json must contain a list")
        for edit in edits:
            if not isinstance(edit, dict) or not {"chunk", "phoneme"}.issubset(edit):
                raise ValueError("Each emphasis edit needs chunk and phoneme indices")
            if set(edit) - {"chunk", "phoneme", "duration_scale", "pitch_scale", "energy_scale"}:
                raise ValueError("Unknown emphasis edit key")
            for key in ("chunk", "phoneme"):
                if type(edit[key]) is not int or edit[key] < 0:
                    raise ValueError("Emphasis indices must be nonnegative integers")
            for key in ("duration_scale", "pitch_scale", "energy_scale"):
                value = edit.get(key, 1.0)
                if type(value) not in {float, int} or not math.isfinite(value) or not 0.25 <= value <= 4:
                    raise ValueError("Emphasis scales must be finite numbers in [0.25, 4]")
        return edits

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def memory_report(self) -> dict[str, Any]:
        if self.device.type == "cpu":
            return {"peak_gpu_vram_bytes": 0, "gpu_budget_guarantee": "CPU only: zero GPU allocations"}
        return {
            "peak_torch_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
            "peak_torch_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device),
            "torch_allocator_limit_bytes": self.allocator_limit_bytes,
            "gpu_budget_guarantee": "Unverified total VRAM; allocator cap excludes CUDA context/driver/external allocations",
        }

    def embedding(self, content: str) -> Any:
        inputs = self.tokenizer([content], return_tensors="pt", truncation=False)
        maximum = self.style_encoder.bert.config.max_position_embeddings
        if inputs["input_ids"].shape[-1] > maximum:
            raise ValueError(
                f"Style/content text exceeds SimBERT's {maximum}-token limit; reduce the chunk size"
            )
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = self.torch.zeros_like(inputs["input_ids"])
        return self.style_encoder(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
        )["pooled_output"].to(self.device)

    def plan(self, text: str, *, warmup: bool = False) -> list[ChunkPlan]:
        torch = self.torch
        chunks = chunk_text(text, self.args.max_chunk_chars)
        edits = [] if warmup else self.emphasis
        if any(edit["chunk"] >= len(chunks) for edit in edits):
            raise ValueError("An emphasis edit references a nonexistent chunk")
        plans: list[ChunkPlan] = []
        with torch.inference_mode(), repository_context(self.repo):
            style = self.embedding(self.args.prompt)
            for chunk_index, content in enumerate(chunks):
                phonemes = self.frontend(content, self.g2p, self.lexicon).split()
                unknown = [phoneme for phoneme in phonemes if phoneme not in self.tokens]
                if unknown:
                    raise ValueError(f"Unknown frontend phonemes: {unknown}")
                ids = torch.tensor([[self.tokens[p] for p in phonemes]], device=self.device, dtype=torch.long)
                lengths = torch.tensor([len(phonemes)], device=self.device)
                mask = self.am.get_mask_from_lengths(lengths)
                x, _ = self.am.encoder(self.am.src_word_emb(ids), ~mask.unsqueeze(-2))
                speaker = self.am.spk_tokenizer(torch.tensor([self.speaker_index], device=self.device))
                semantic = self.embedding(content)
                x = self.am.embed_projection1(
                    torch.cat(
                        (
                            x,
                            speaker[:, None].expand(-1, len(phonemes), -1),
                            style[:, None].expand(-1, len(phonemes), -1),
                            semantic[:, None].expand(-1, len(phonemes), -1),
                        ),
                        dim=-1,
                    )
                )
                base = self.am.duration_predictor.inference(x, mask.unsqueeze(-1))
                factors = torch.full_like(base, 1 / self.args.speed, dtype=torch.float32)
                pitch_factors = torch.full_like(base, self.args.pitch_predictor_scale, dtype=torch.float32)
                energy_factors = torch.full_like(base, self.args.energy_predictor_scale, dtype=torch.float32)
                for edit in edits:
                    if edit["chunk"] != chunk_index:
                        continue
                    token_index = edit["phoneme"]
                    if token_index >= len(phonemes):
                        raise ValueError(
                            f"Emphasis edit references phoneme {token_index} outside chunk {chunk_index}"
                        )
                    factors[0, token_index] *= edit.get("duration_scale", 1.0)
                    pitch_factors[0, token_index] *= edit.get("pitch_scale", 1.0)
                    energy_factors[0, token_index] *= edit.get("energy_scale", 1.0)
                durations, zero_indices, clamped_indices = apply_duration_controls(
                    base, factors, phonemes, self.args.min_phoneme_frames
                )
                if int(durations.sum().item()) == 0:
                    raise RuntimeError(
                        "The duration predictor/control combination assigned zero frames; refusing bad speech"
                    )
                frames = int(durations.sum().item())
                if frames > self.args.max_chunk_frames:
                    raise RuntimeError(
                        f"Chunk {chunk_index} allocates {frames} acoustic frames; reduce --max-chunk-chars or increase speed"
                    )
                samples = vocoder_samples(frames, self.model.generator)
                plans.append(
                    ChunkPlan(
                        content,
                        phonemes,
                        x.detach().cpu(),
                        mask.cpu(),
                        durations.cpu(),
                        pitch_factors.cpu(),
                        energy_factors.cpu(),
                        frames,
                        samples,
                        zero_indices,
                        clamped_indices,
                    )
                )
        self.synchronize()
        return plans

    def synthesize(self, plan: ChunkPlan, progress: Any) -> Any:
        torch = self.torch
        with torch.inference_mode():
            x = plan.conditioned_tokens.to(self.device)
            mask = plan.mask.to(self.device)
            durations = plan.durations.to(self.device)
            pitch = self.am.pitch_predictor(x, mask.unsqueeze(-1)) * plan.pitch_factors.to(self.device)
            energy = self.am.energy_predictor(x, mask.unsqueeze(-1)) * plan.energy_factors.to(self.device)
            x = (
                x
                + self.am.pitch_embed(pitch.unsqueeze(1)).transpose(1, 2)
                + self.am.energy_embed(energy.unsqueeze(1)).transpose(1, 2)
            )
            x = self.am.length_regulator(x, durations, None, ~mask)
            if x.shape[1] != plan.frames:
                raise RuntimeError("Gaussian upsampling violated the cached acoustic frame count")
            progress()
            x, _ = self.am.decoder(x, None)
            mel = self.am.to_mel(x).transpose(1, 2)
            progress()
            wav = self.model.generator(mel)
            self.synchronize()
            progress()
            if wav.shape != (1, 1, plan.native_samples):
                raise RuntimeError(
                    f"HiFiGAN length changed: {tuple(wav.shape)}, expected {(1, 1, plan.native_samples)}"
                )
            return edge_fade(wav[0, 0].cpu().numpy(), NATIVE_RATE, self.args.edge_fade_ms)


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
    # Resolve these paths before the source checkout temporarily becomes the cwd.
    for name in (
        "repo",
        "generator_checkpoint",
        "style_checkpoint",
        "model_yaml",
        "output_dir",
        "emphasis_json",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if Path(args.bert_path).exists():
        args.bert_path = str(Path(args.bert_path).resolve())
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8-sig")
    text = text.strip()
    if not text:
        raise ValueError("The input text is empty")
    words = len(re.findall(r"\S+", text))
    if not 10 <= words <= 500:
        raise ValueError(f"Expected the requested 10..500 whitespace-delimited words; got {words}")
    import numpy as np
    import soundfile as sf
    import torch

    if args.sample_rate and args.sample_rate != NATIVE_RATE:
        import scipy.signal  # noqa: F401
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "emotivoice_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Output already exists: {manifest_path}. Choose a fresh --output-dir")
    startup_start = time.perf_counter()
    engine = Engine(args)
    # Warm BERT/the phoneme encoder/duration predictor only; no acoustic decoder or vocoder.
    engine.plan("This short initialization sentence prepares the duration prediction network.", warmup=True)
    startup_s = time.perf_counter() - startup_start
    preview_start = time.perf_counter()
    plans = engine.plan(text)
    preview_s = time.perf_counter() - preview_start
    export_rate = args.sample_rate or NATIVE_RATE
    pause_samples = round(args.pause_ms * NATIVE_RATE / 1000)
    native_samples = sum(plan.native_samples for plan in plans) + pause_samples * (len(plans) - 1)
    expected_samples = predicted_export_samples(native_samples, NATIVE_RATE, export_rate)
    manifest: dict[str, Any] = {
        "model": "EmotiVoice",
        "source_revision": SOURCE_REVISION,
        "checkpoint_sha256": engine.weights,
        "style_checkpoint_load_report": engine.style_load_report,
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "speaker_index": engine.speaker_index,
        "speaker_label": engine.speakers[engine.speaker_index],
        "text": text,
        "word_count": words,
        "native_rate": NATIVE_RATE,
        "export_rate": export_rate,
        "startup_seconds": startup_s,
        "warm_preview_seconds": preview_s,
        "predicted_samples": expected_samples,
        "predicted_seconds": expected_samples / export_rate,
        "pause_native_samples": pause_samples,
        "duration_method": "Cached integer phoneme durations, analytic HiFiGAN convolution lengths, deterministic resampling",
        "frontend_guard": "Fallback English G2P words have explicit separators; punctuation never pops a spoken phone; empty fallback pronunciations rejected",
        "sampling_note": "Deterministic eval inference: seeds do not change speech at fixed controls",
        "emphasis_edits": engine.emphasis,
        "chunks": [
            {
                "text": p.text,
                "phonemes": p.phonemes,
                "phoneme_durations": p.durations[0].tolist(),
                "acoustic_frames": p.frames,
                "native_samples": p.native_samples,
                "zero_duration_lexical_indices_before_guard": p.zero_duration_lexical_indices,
                "minimum_duration_clamped_indices": p.minimum_duration_clamped_indices,
                "minimum_duration_clamped_phonemes": [
                    p.phonemes[index] for index in p.minimum_duration_clamped_indices
                ],
            }
            for p in plans
        ],
        **engine.memory_report(),
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
    if args.n > 1:
        print(
            "EmotiVoice inference is deterministic: repeated fixed-config samples will be identical.",
            flush=True,
        )
    write_json(manifest_path, manifest)
    if args.estimate_only:
        return 0
    units_done = 0
    units_total = args.n * len(plans) * 3

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
        torch.manual_seed(seed)
        path = args.output_dir / f"emotivoice_{index + 1:03d}_seed{seed}.wav"
        if path.exists():
            raise FileExistsError(path)
        started = time.perf_counter()
        pieces = []
        for chunk_index, plan in enumerate(plans):
            if chunk_index and pause_samples:
                pieces.append(np.zeros(pause_samples, dtype=np.float32))
            pieces.append(engine.synthesize(plan, progress))
        audio = resample_audio(np.concatenate(pieces), NATIVE_RATE, export_rate)
        audio, output_levels = protect_peak(audio, args.gain_db, args.normalize_peak)
        error = abs(len(audio) - expected_samples) / len(audio)
        if not np.isfinite(audio).all() or len(audio) != expected_samples or error >= 0.01:
            raise RuntimeError(
                f"Duration/finite-audio validation failed: predicted={expected_samples}, actual={len(audio)}, error={error:.8%}"
            )
        memory = engine.memory_report()
        if args.device != "cpu" and memory["peak_torch_reserved_bytes"] > engine.budget_bytes:
            raise RuntimeError("Observed PyTorch reserved memory exceeds the configured CUDA budget")
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
                **memory,
            }
        )
        manifest.update(memory)
        write_json(manifest_path, manifest)
    print(f"\nSaved {args.n} samples and {manifest_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

#!/usr/bin/env python3
"""
Safety-warning montage narrator

Pipeline:
  1. Read warning material + optional context.
  2. Get silent montage duration with ffprobe.
  3. Load a local Hugging Face causal LLM on the single GPU.
  4. Generate several narration candidates.
  5. Enforce intra-run and cross-run novelty with persistent SQLite history.
  6. Use the user's duration estimator to jointly choose script length,
     Chatterbox exaggeration, and cfg_weight in a conservative naturalness band.
  7. Unload the LLM from GPU.
  8. Load Chatterbox, synthesize sentence-aware chunks, concatenate them.
  9. Apply a small tempo correction to exactly fit the video duration.
 10. Mux the narration into the montage and store the accepted script in history.

The duration estimator is supplied as --estimator module:function. It must return
seconds and may accept either positional arguments:

    estimate_duration(text, exaggeration, cfg_weight) -> float

or keyword arguments with the same names.

If --estimator is omitted, a rough WPM fallback is used. For production, use
your own estimator.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence


# ----------------------------- generic utilities -----------------------------

WORD_RE = re.compile(r"[\w’'-]+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(\[]?[A-Z0-9])")


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_compare(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s'-]+", " ", text, flags=re.UNICODE)
    return normalize_space(text)


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    parts = SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / max(1, len(union))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_cmd(cmd: list[str], capture: bool = False) -> str:
    if capture:
        p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.stdout
    subprocess.run(cmd, check=True)
    return ""


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable '{name}' was not found on PATH.")


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ------------------------------- input loading -------------------------------


def flatten_json_strings(obj) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        if obj.strip():
            out.append(obj.strip())
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.strip():
                out.append(k.strip())
            out.extend(flatten_json_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(flatten_json_strings(v))
    elif obj is not None:
        out.append(str(obj))
    return out


def read_textish_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".rst", ".log", ".srt", ".vtt"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join(flatten_json_strings(obj))

    if suffix == ".jsonl":
        vals: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                vals.extend(flatten_json_strings(json.loads(line)))
        return "\n".join(vals)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("PyYAML is required for YAML warning files.") from e
        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
        return "\n".join(flatten_json_strings(obj))

    if suffix == ".csv":
        vals: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.reader(f):
                vals.extend(c.strip() for c in row if c.strip())
        return "\n".join(vals)

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("pypdf is required for PDF warning files.") from e
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "Image warnings require Pillow + pytesseract and the Tesseract executable, "
                "or pre-extract the warning text yourself."
            ) from e
        return pytesseract.image_to_string(Image.open(path))

    # Try as plain text for unknown extensions.
    return path.read_text(encoding="utf-8", errors="replace")


def read_argument_or_file(value: Optional[str]) -> str:
    if not value:
        return ""
    p = Path(value)
    if p.exists() and p.is_file():
        return read_textish_file(p)
    return value


def collect_warning_text(items: Sequence[str]) -> str:
    chunks: list[str] = []
    seen: set[str] = set()
    for item in items:
        p = Path(item)
        text = read_textish_file(p) if p.exists() else item
        # Keep source blocks distinct, but dedupe exact normalized blocks.
        key = normalize_for_compare(text)
        if key and key not in seen:
            seen.add(key)
            chunks.append(text.strip())
    return "\n\n--- WARNING SOURCE ---\n\n".join(chunks)


# ------------------------------- video timing --------------------------------


def media_duration_seconds(path: Path) -> float:
    require_binary("ffprobe")
    out = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    data = json.loads(out)
    return float(data["format"]["duration"])


# ---------------------------- persistent novelty DB ---------------------------


class HistoryDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                warning_hash TEXT NOT NULL,
                video_seconds REAL NOT NULL,
                script TEXT NOT NULL,
                exaggeration REAL NOT NULL,
                cfg_weight REAL NOT NULL,
                estimated_seconds REAL NOT NULL,
                actual_seconds REAL,
                metadata_json TEXT
            )
            """
        )
        self.conn.commit()

    def recent_scripts(self, limit: int = 50) -> list[str]:
        rows = self.conn.execute(
            "SELECT script FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [r[0] for r in rows]

    def add(
        self,
        warning_hash: str,
        video_seconds: float,
        script: str,
        exaggeration: float,
        cfg_weight: float,
        estimated_seconds: float,
        actual_seconds: Optional[float],
        metadata: dict,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO runs(
                created_at, warning_hash, video_seconds, script,
                exaggeration, cfg_weight, estimated_seconds,
                actual_seconds, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                warning_hash,
                video_seconds,
                script,
                exaggeration,
                cfg_weight,
                estimated_seconds,
                actual_seconds,
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


@dataclass
class NoveltyMetrics:
    duplicate_sentence_pairs: int
    worst_internal_sentence_similarity: float
    worst_history_sentence_similarity: float
    worst_history_ngram_jaccard: float
    repeated_history_phrases: list[str]

    @property
    def penalty(self) -> float:
        return (
            2.0 * self.duplicate_sentence_pairs
            + 1.5 * self.worst_internal_sentence_similarity
            + 2.0 * self.worst_history_sentence_similarity
            + 4.0 * self.worst_history_ngram_jaccard
        )


def sentence_similarity(a: str, b: str) -> float:
    wa, wb = set(words(a)), set(words(b))
    if not wa or not wb:
        return 0.0
    return jaccard(wa, wb)


def novelty_metrics(script: str, history: Sequence[str], ngram_n: int = 5) -> NoveltyMetrics:
    sents = [s for s in split_sentences(script) if len(words(s)) >= 4]
    duplicate_pairs = 0
    worst_internal = 0.0
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            sim = sentence_similarity(sents[i], sents[j])
            worst_internal = max(worst_internal, sim)
            if normalize_for_compare(sents[i]) == normalize_for_compare(sents[j]):
                duplicate_pairs += 1

    script_tokens = words(script)
    script_ng = ngrams(script_tokens, ngram_n)
    worst_hist_ng = 0.0
    worst_hist_sent = 0.0
    overlap_counter: Counter[tuple[str, ...]] = Counter()

    for old in history:
        old_tokens = words(old)
        old_ng = ngrams(old_tokens, ngram_n)
        worst_hist_ng = max(worst_hist_ng, jaccard(script_ng, old_ng))
        for ng in script_ng & old_ng:
            overlap_counter[ng] += 1

        old_sents = [s for s in split_sentences(old) if len(words(s)) >= 4]
        # Cap pairwise cost for very large histories.
        for s in sents[:80]:
            for osent in old_sents[:100]:
                worst_hist_sent = max(worst_hist_sent, sentence_similarity(s, osent))

    repeated_phrases = [" ".join(ng) for ng, _ in overlap_counter.most_common(40)]
    return NoveltyMetrics(
        duplicate_sentence_pairs=duplicate_pairs,
        worst_internal_sentence_similarity=worst_internal,
        worst_history_sentence_similarity=worst_hist_sent,
        worst_history_ngram_jaccard=worst_hist_ng,
        repeated_history_phrases=repeated_phrases,
    )


def novelty_passes(m: NoveltyMetrics, max_ngram_jaccard: float, max_sentence_similarity: float) -> bool:
    return (
        m.duplicate_sentence_pairs == 0
        and m.worst_internal_sentence_similarity < 0.78
        and m.worst_history_sentence_similarity < max_sentence_similarity
        and m.worst_history_ngram_jaccard < max_ngram_jaccard
    )


# ---------------------------- duration estimator ------------------------------


class DurationEstimator:
    def __init__(self, spec: Optional[str], fallback_wpm: float = 152.0):
        self.spec = spec
        self.fallback_wpm = fallback_wpm
        self.func: Optional[Callable] = None
        if spec:
            if ":" not in spec:
                raise ValueError("--estimator must be in module:function form")
            module_name, func_name = spec.split(":", 1)
            mod = importlib.import_module(module_name)
            self.func = getattr(mod, func_name)

    def __call__(self, text: str, exaggeration: float, cfg_weight: float) -> float:
        if self.func is not None:
            try:
                value = self.func(text, exaggeration, cfg_weight)
            except TypeError:
                value = self.func(
                    text=text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Duration estimator returned invalid value: {value}")
            return value

        # Rough fallback only. Official Chatterbox guidance indicates higher
        # exaggeration can speed speech and lower CFG can counter that.
        wc = max(1, len(words(text)))
        rate = self.fallback_wpm * (
            1.0 + 0.18 * (exaggeration - 0.5) + 0.12 * (cfg_weight - 0.5)
        )
        rate = max(90.0, min(220.0, rate))
        punctuation_pause = 0.08 * len(re.findall(r"[,;:]", text)) + 0.16 * len(
            re.findall(r"[.!?]", text)
        )
        return 60.0 * wc / rate + punctuation_pause


# ---------------------------- TTS-aware chunking ------------------------------


def hard_wrap_sentence(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    tokens = sentence.split()
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for tok in tokens:
        add = len(tok) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            chunks.append(" ".join(cur))
            cur = [tok]
            cur_len = len(tok)
        else:
            cur.append(tok)
            cur_len += add
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def split_tts_chunks(text: str, max_chars: int = 280) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    out: list[str] = []
    cur = ""
    for sentence in sentences:
        for piece in hard_wrap_sentence(sentence, max_chars):
            proposal = piece if not cur else cur + " " + piece
            if len(proposal) <= max_chars:
                cur = proposal
            else:
                if cur:
                    out.append(cur)
                cur = piece
    if cur:
        out.append(cur)
    return out


def estimated_script_seconds(
    text: str,
    estimator: DurationEstimator,
    exaggeration: float,
    cfg_weight: float,
    max_chunk_chars: int,
    inter_chunk_pause: float,
) -> float:
    chunks = split_tts_chunks(text, max_chunk_chars)
    if not chunks:
        return 0.0
    base = sum(estimator(c, exaggeration, cfg_weight) for c in chunks)
    return base + inter_chunk_pause * max(0, len(chunks) - 1)


# --------------------------- Chatterbox parameter search ----------------------


STYLE_PRIORS = {
    "neutral": (0.50, 0.50),
    "serious": (0.46, 0.56),
    "calm": (0.42, 0.58),
    "educational": (0.48, 0.54),
    "urgent": (0.62, 0.46),
}


@dataclass
class ParamPlan:
    exaggeration: float
    cfg_weight: float
    estimated_seconds: float
    duration_error_seconds: float
    duration_error_fraction: float
    objective: float


def frange(start: float, stop: float, step: float) -> list[float]:
    vals: list[float] = []
    x = start
    while x <= stop + 1e-9:
        vals.append(round(x, 4))
        x += step
    return vals


def choose_params(
    script: str,
    target_seconds: float,
    estimator: DurationEstimator,
    style: str,
    max_chunk_chars: int,
    inter_chunk_pause: float,
) -> ParamPlan:
    center_exag, center_cfg = STYLE_PRIORS.get(style, STYLE_PRIORS["neutral"])

    # Conservative band. Script length should do the heavy lifting.
    exags = frange(max(0.30, center_exag - 0.18), min(0.82, center_exag + 0.18), 0.04)
    cfgs = frange(max(0.25, center_cfg - 0.18), min(0.75, center_cfg + 0.18), 0.04)

    best: Optional[ParamPlan] = None
    for e in exags:
        for c in cfgs:
            est = estimated_script_seconds(
                script, estimator, e, c, max_chunk_chars, inter_chunk_pause
            )
            err = est - target_seconds
            frac = abs(err) / max(1e-6, target_seconds)

            # Naturalness prior: staying near the style center matters after
            # duration fit. This intentionally prevents control extremes.
            prior = ((e - center_exag) / 0.18) ** 2 + ((c - center_cfg) / 0.18) ** 2
            objective = 10.0 * frac + 0.20 * prior

            plan = ParamPlan(e, c, est, err, frac, objective)
            if best is None or plan.objective < best.objective:
                best = plan

    assert best is not None
    return best


# ------------------------------- local HF LLM --------------------------------


class HFLocalLLM:
    def __init__(
        self,
        model_name_or_path: str,
        load_in_4bit: bool,
        trust_remote_code: bool,
        seed: int,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)

        log(f"Loading LLM: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e:
                raise RuntimeError(
                    "4-bit loading requested but bitsandbytes/quantization setup failed."
                ) from e

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        self.model.eval()

    def _render_chat(self, system: str, user: str):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return rendered

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int,
        temperature: float = 0.82,
        top_p: float = 0.92,
        repetition_penalty: float = 1.08,
    ) -> str:
        rendered = self._render_chat(system, user)
        inputs = self.tokenizer(rendered, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                    if self.tokenizer.pad_token_id is None
                    else self.tokenizer.pad_token_id
                ),
            )
        text = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
        text = text.strip()
        text = re.sub(r"^(?:narration|script)\s*:\s*", "", text, flags=re.I)
        text = text.strip("` \n\t")
        return normalize_space(text)

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        free_gpu()


# --------------------------- narration prompt design --------------------------


SYSTEM_PROMPT = """You write spoken narration for safety-warning video montages.
Your output is read aloud by TTS, so it must sound natural, continuous, and easy to follow.
Preserve the meaning of the supplied warnings. Do not invent hazards, guarantees, incidents,
statistics, laws, or product facts that are not in the supplied material. The montage is a
collection: never imply that every warning appears exactly once, or that each sentence maps to
one specific shot. Avoid filler, repeated conclusions, repeated sentence openings, rhetorical
loops, lists that merely restate the same point, and generic phrases reused from earlier scripts.
If a warning is legally or technically quoted and its exact wording matters, retain that wording;
otherwise paraphrase accurately. Output only the finished narration, with no title, notes,
analysis, bullets, timestamps, or quotation fences."""


def build_generation_prompt(
    warnings: str,
    context: str,
    target_words: int,
    target_seconds: float,
    style: str,
    forbidden_phrases: Sequence[str],
    feedback: str,
) -> str:
    forbidden = "\n".join(f"- {p}" for p in forbidden_phrases[:50]) or "(none)"
    context = context.strip() or "(no extra context supplied)"
    return f"""Create one narration of about {target_words} words for approximately {target_seconds:.1f} seconds of speech.
Tone: {style}. Use varied syntax and transitions. Consolidate overlapping warnings instead of
repeating them. Give each sentence a distinct job: identify the risk, explain the safe behavior,
clarify a boundary, or connect the warning to the montage context. Do not pad with synonyms.

WARNING MATERIAL:
{warnings}

ADDED CONTEXT:
{context}

PHRASES OR FORMULATIONS TO AVOID BECAUSE THEY OVERLAP EARLIER OUTPUTS:
{forbidden}

REVISION FEEDBACK FROM THE TIMING/NOVELTY CHECKER:
{feedback or '(none)'}

Return narration only. Aim within ±6% of {target_words} words."""


def build_forbidden_history_phrases(history: Sequence[str], warning_text: str, limit: int = 45) -> list[str]:
    warning_ngrams = ngrams(words(warning_text), 6)
    counter: Counter[tuple[str, ...]] = Counter()
    # Favor recent material, but only ban longer exact formulations. Required
    # warning language that appears in source material is exempt.
    for old in history[:25]:
        for ng in ngrams(words(old), 6):
            if ng not in warning_ngrams:
                counter[ng] += 1
    return [" ".join(ng) for ng, _ in counter.most_common(limit)]


@dataclass
class Candidate:
    script: str
    novelty: NoveltyMetrics
    params: ParamPlan
    word_count: int
    score: float


def score_candidate(
    script: str,
    history: Sequence[str],
    target_seconds: float,
    estimator: DurationEstimator,
    style: str,
    max_chunk_chars: int,
    inter_chunk_pause: float,
) -> Candidate:
    novelty = novelty_metrics(script, history)
    params = choose_params(
        script,
        target_seconds,
        estimator,
        style,
        max_chunk_chars,
        inter_chunk_pause,
    )
    wc = len(words(script))
    score = params.objective + 0.65 * novelty.penalty
    return Candidate(script, novelty, params, wc, score)


def create_best_script(
    llm: HFLocalLLM,
    warnings: str,
    context: str,
    history: Sequence[str],
    target_seconds: float,
    estimator: DurationEstimator,
    style: str,
    initial_wpm: float,
    candidates_per_round: int,
    max_rounds: int,
    duration_tolerance_fraction: float,
    max_ngram_jaccard: float,
    max_sentence_similarity: float,
    max_chunk_chars: int,
    inter_chunk_pause: float,
    max_new_tokens_cap: int,
) -> Candidate:
    target_words = max(35, round(target_seconds * initial_wpm / 60.0))
    base_forbidden = build_forbidden_history_phrases(history, warnings)
    dynamic_forbidden = list(base_forbidden)
    feedback = ""
    best_overall: Optional[Candidate] = None

    for round_idx in range(1, max_rounds + 1):
        log(f"LLM round {round_idx}/{max_rounds}; target ≈ {target_words} words")
        round_candidates: list[Candidate] = []

        for i in range(candidates_per_round):
            prompt = build_generation_prompt(
                warnings,
                context,
                target_words,
                target_seconds,
                style,
                dynamic_forbidden,
                feedback,
            )
            max_new_tokens = min(
                max_new_tokens_cap,
                max(512, int(target_words * 2.2 + 220)),
            )
            script = llm.generate(
                SYSTEM_PROMPT,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.78 + 0.04 * min(i, 2),
                top_p=0.92,
                repetition_penalty=1.09,
            )
            if len(words(script)) < 20:
                continue

            cand = score_candidate(
                script,
                history,
                target_seconds,
                estimator,
                style,
                max_chunk_chars,
                inter_chunk_pause,
            )
            round_candidates.append(cand)
            log(
                f"  candidate {i+1}: {cand.word_count} words, "
                f"est={cand.params.estimated_seconds:.2f}s, "
                f"err={100*cand.params.duration_error_fraction:.2f}%, "
                f"exag={cand.params.exaggeration:.2f}, cfg={cand.params.cfg_weight:.2f}, "
                f"history_5gram={cand.novelty.worst_history_ngram_jaccard:.3f}"
            )

        if not round_candidates:
            raise RuntimeError("The LLM did not produce a usable narration candidate.")

        round_candidates.sort(key=lambda c: c.score)
        best = round_candidates[0]
        if best_overall is None or best.score < best_overall.score:
            best_overall = best

        nov_ok = novelty_passes(
            best.novelty, max_ngram_jaccard, max_sentence_similarity
        )
        dur_ok = best.params.duration_error_fraction <= duration_tolerance_fraction
        if nov_ok and dur_ok:
            return best

        repeated = best.novelty.repeated_history_phrases[:20]
        for p in repeated:
            if p not in dynamic_forbidden:
                dynamic_forbidden.append(p)

        # Adjust target length based on the estimator. Keep the correction
        # bounded so one noisy estimate cannot wildly swing the next prompt.
        ratio = target_seconds / max(1.0, best.params.estimated_seconds)
        ratio = min(1.28, max(0.78, ratio))
        target_words = max(35, round(best.word_count * ratio))

        issues: list[str] = []
        if not dur_ok:
            direction = "shorter" if best.params.estimated_seconds > target_seconds else "longer"
            issues.append(
                f"Make the next narration {direction}; current estimate is "
                f"{best.params.estimated_seconds:.1f}s for a {target_seconds:.1f}s target."
            )
        if not nov_ok:
            issues.append(
                "Use substantially different sentence structures and transitions from prior outputs; "
                "do not merely swap synonyms."
            )
        feedback = " ".join(issues)

    assert best_overall is not None
    log("Warning: strict timing/novelty thresholds were not both met; using the best candidate found.")
    return best_overall


# ------------------------------- Chatterbox TTS -------------------------------


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def synthesize_chatterbox(
    script: str,
    out_wav: Path,
    voice_ref: Optional[Path],
    language: str,
    multilingual_v3: bool,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    tts_repetition_penalty: float,
    max_chunk_chars: int,
    inter_chunk_pause: float,
    seed: int,
) -> tuple[float, int, int]:
    import torch
    import torchaudio as ta

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        log("Warning: CUDA is not available; Chatterbox will run on CPU.")

    if language == "en":
        from chatterbox.tts import ChatterboxTTS

        log("Loading Chatterbox English TTS...")
        model = ChatterboxTTS.from_pretrained(device=device)
        multilingual = False
    else:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        log(f"Loading Chatterbox Multilingual ({'v3' if multilingual_v3 else 'v2'})...")
        kwargs = {"device": device}
        if multilingual_v3:
            kwargs["t3_model"] = "v3"
        model = ChatterboxMultilingualTTS.from_pretrained(**kwargs)
        multilingual = True

    chunks = split_tts_chunks(script, max_chunk_chars)
    if not chunks:
        raise RuntimeError("No TTS chunks were produced.")

    waves = []
    sr = int(model.sr)
    silence = torch.zeros((1, max(1, round(sr * inter_chunk_pause))), dtype=torch.float32)

    for i, chunk in enumerate(chunks):
        set_all_seeds(seed + i)
        kwargs = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "repetition_penalty": tts_repetition_penalty,
        }
        if voice_ref is not None:
            kwargs["audio_prompt_path"] = str(voice_ref)
        if multilingual:
            kwargs["language_id"] = language

        log(f"  TTS chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        with torch.inference_mode():
            wav = model.generate(chunk, **kwargs)
        wav = wav.detach().float().cpu()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim > 2:
            wav = wav.reshape(1, -1)
        waves.append(wav)
        if i != len(chunks) - 1:
            waves.append(silence.clone())

    full = torch.cat(waves, dim=-1)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(out_wav), full, sr)
    seconds = full.shape[-1] / sr

    del model
    del full
    del waves
    free_gpu()
    return seconds, sr, len(chunks)


# -------------------------- exact duration + muxing ---------------------------


def atempo_filter_chain(factor: float) -> str:
    # ffmpeg atempo historically had a practical 0.5..2 range on many builds.
    # Chain factors so the function remains portable.
    factors: list[float] = []
    x = factor
    while x > 2.0:
        factors.append(2.0)
        x /= 2.0
    while x < 0.5:
        factors.append(0.5)
        x /= 0.5
    factors.append(x)
    return ",".join(f"atempo={f:.8f}" for f in factors)


def fit_audio_to_video(
    raw_wav: Path,
    fitted_wav: Path,
    raw_seconds: float,
    video_seconds: float,
    head_pad: float,
    tail_pad: float,
    max_tempo_change: float,
) -> tuple[float, float]:
    """Tempo-fit the spoken portion, then make the final WAV exact by samples.

    `atempo` has a small algorithmic latency, so its measured output duration is
    not always exactly raw_duration/factor. We therefore measure and refine the
    factor up to three times, then add head/tail silence and trim/pad at the
    sample level for a deterministic final duration.
    """
    require_binary("ffmpeg")
    narration_target = video_seconds - head_pad - tail_pad
    if narration_target <= 0.25:
        raise ValueError("head_pad + tail_pad leaves no usable narration duration.")

    factor = raw_seconds / narration_target
    if abs(factor - 1.0) > max_tempo_change:
        raise RuntimeError(
            f"Actual TTS duration is {raw_seconds:.2f}s but narration target is "
            f"{narration_target:.2f}s; required tempo factor {factor:.4f} exceeds "
            f"--max-tempo-change={max_tempo_change:.3f}. Improve/calibrate the duration "
            "estimator or increase that limit knowingly."
        )

    tempo_tmp = fitted_wav.with_name(fitted_wav.stem + ".tempo_tmp.wav")
    measured = None
    try:
        for _ in range(3):
            tempo = atempo_filter_chain(factor)
            run_cmd(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw_wav),
                    "-af",
                    tempo,
                    "-c:a",
                    "pcm_s16le",
                    str(tempo_tmp),
                ]
            )
            measured = media_duration_seconds(tempo_tmp)
            if abs(measured - narration_target) <= 0.015:
                break

            corrected = factor * (measured / narration_target)
            # Keep refinement within the user's naturalness guardrail.
            if abs(corrected - 1.0) > max_tempo_change:
                break
            factor = corrected

        import torch
        import torchaudio as ta

        wav, sr = ta.load(str(tempo_tmp))
        wav = wav.float()
        target_narr_samples = round(narration_target * sr)
        if wav.shape[-1] > target_narr_samples:
            wav = wav[..., :target_narr_samples]
        elif wav.shape[-1] < target_narr_samples:
            pad = target_narr_samples - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad))

        head_samples = round(head_pad * sr)
        total_samples = round(video_seconds * sr)
        head = torch.zeros((wav.shape[0], head_samples), dtype=wav.dtype)
        combined = torch.cat([head, wav], dim=-1)
        if combined.shape[-1] < total_samples:
            combined = torch.nn.functional.pad(combined, (0, total_samples - combined.shape[-1]))
        else:
            combined = combined[..., :total_samples]

        fitted_wav.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(fitted_wav), combined, sr)
        final_seconds = combined.shape[-1] / sr
        return factor, final_seconds
    finally:
        try:
            tempo_tmp.unlink(missing_ok=True)
        except Exception:
            pass

def mux_video_audio(video: Path, audio: Path, output: Path, video_seconds: float) -> None:
    require_binary("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{video_seconds:.6f}",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


# ---------------------------------- CLI/main ----------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--video", required=True, type=Path, help="Audioless montage video")
    p.add_argument(
        "--warning",
        action="append",
        required=True,
        help="Warning text or path; repeat this flag for multiple sources",
    )
    p.add_argument("--context", default="", help="Extra context text or a file path")
    p.add_argument("--voice-ref", type=Path, default=None, help="Reference voice WAV/MP3")
    p.add_argument("--output-dir", type=Path, default=Path("montage_narration_out"))
    p.add_argument("--output-name", default="montage_with_narration.mp4")

    p.add_argument("--llm", required=True, help="Local HF model name or path")
    p.add_argument("--llm-4bit", action="store_true", help="Load the LLM with bitsandbytes 4-bit")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--llm-max-new-tokens", type=int, default=8192)

    p.add_argument(
        "--estimator",
        default=None,
        help="Your duration function as module:function",
    )
    p.add_argument("--fallback-wpm", type=float, default=152.0)
    p.add_argument("--initial-wpm", type=float, default=150.0)

    p.add_argument(
        "--style",
        choices=sorted(STYLE_PRIORS),
        default="serious",
    )
    p.add_argument("--language", default="en", help="Chatterbox language ID")
    p.add_argument("--multilingual-v3", action="store_true")
    p.add_argument("--tts-temperature", type=float, default=0.76)
    p.add_argument("--tts-repetition-penalty", type=float, default=1.22)
    p.add_argument("--max-chunk-chars", type=int, default=270)
    p.add_argument("--inter-chunk-pause", type=float, default=0.12)

    p.add_argument("--head-pad", type=float, default=0.18)
    p.add_argument("--tail-pad", type=float, default=0.18)
    p.add_argument("--duration-tolerance", type=float, default=0.025, help="Estimator tolerance fraction")
    p.add_argument("--max-tempo-change", type=float, default=0.06, help="Max |atempo_factor - 1|")

    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--history-limit", type=int, default=50)
    p.add_argument("--max-history-ngram-jaccard", type=float, default=0.16)
    p.add_argument("--max-history-sentence-similarity", type=float, default=0.82)
    p.add_argument("--history-db", type=Path, default=Path("narration_history.sqlite3"))
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    require_binary("ffprobe")
    require_binary("ffmpeg")

    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if args.voice_ref is not None and not args.voice_ref.exists():
        raise FileNotFoundError(args.voice_ref)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_wav = args.output_dir / "narration_raw.wav"
    fitted_wav = args.output_dir / "narration_fitted.wav"
    script_path = args.output_dir / "narration.txt"
    plan_path = args.output_dir / "plan.json"
    output_video = args.output_dir / args.output_name

    warnings = collect_warning_text(args.warning)
    if not normalize_space(warnings):
        raise RuntimeError("No warning text could be extracted.")
    context = read_argument_or_file(args.context)
    warning_hash = sha256_text(warnings)

    video_seconds = media_duration_seconds(args.video)
    narration_target = video_seconds - args.head_pad - args.tail_pad
    if narration_target <= 1.0:
        raise RuntimeError("Video is too short after head/tail padding.")
    log(f"Video duration: {video_seconds:.3f}s; narration target: {narration_target:.3f}s")

    estimator = DurationEstimator(args.estimator, fallback_wpm=args.fallback_wpm)
    if args.estimator is None:
        log("Warning: using the rough fallback duration model. Supply --estimator module:function for accurate fitting.")

    history_db = HistoryDB(args.history_db)
    history = history_db.recent_scripts(args.history_limit)
    log(f"Loaded {len(history)} prior narration(s) for cross-run novelty checking.")

    llm = HFLocalLLM(
        args.llm,
        load_in_4bit=args.llm_4bit,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    try:
        best = create_best_script(
            llm=llm,
            warnings=warnings,
            context=context,
            history=history,
            target_seconds=narration_target,
            estimator=estimator,
            style=args.style,
            initial_wpm=args.initial_wpm,
            candidates_per_round=args.candidates,
            max_rounds=args.rounds,
            duration_tolerance_fraction=args.duration_tolerance,
            max_ngram_jaccard=args.max_history_ngram_jaccard,
            max_sentence_similarity=args.max_history_sentence_similarity,
            max_chunk_chars=args.max_chunk_chars,
            inter_chunk_pause=args.inter_chunk_pause,
            max_new_tokens_cap=args.llm_max_new_tokens,
        )
    finally:
        llm.unload()
        del llm
        free_gpu()

    script_path.write_text(best.script + "\n", encoding="utf-8")
    log(
        f"Selected script: {best.word_count} words, "
        f"exaggeration={best.params.exaggeration:.2f}, "
        f"cfg_weight={best.params.cfg_weight:.2f}, "
        f"estimated={best.params.estimated_seconds:.2f}s"
    )

    raw_seconds, sr, chunk_count = synthesize_chatterbox(
        script=best.script,
        out_wav=raw_wav,
        voice_ref=args.voice_ref,
        language=args.language,
        multilingual_v3=args.multilingual_v3,
        exaggeration=best.params.exaggeration,
        cfg_weight=best.params.cfg_weight,
        temperature=args.tts_temperature,
        tts_repetition_penalty=args.tts_repetition_penalty,
        max_chunk_chars=args.max_chunk_chars,
        inter_chunk_pause=args.inter_chunk_pause,
        seed=args.seed + 1000,
    )
    log(f"Actual raw TTS duration: {raw_seconds:.3f}s across {chunk_count} chunks")

    tempo_factor, fitted_seconds = fit_audio_to_video(
        raw_wav=raw_wav,
        fitted_wav=fitted_wav,
        raw_seconds=raw_seconds,
        video_seconds=video_seconds,
        head_pad=args.head_pad,
        tail_pad=args.tail_pad,
        max_tempo_change=args.max_tempo_change,
    )
    log(f"Applied atempo factor {tempo_factor:.5f}; fitted audio={fitted_seconds:.3f}s")

    mux_video_audio(args.video, fitted_wav, output_video, video_seconds)

    metadata = {
        "video": str(args.video),
        "output_video": str(output_video),
        "word_count": best.word_count,
        "style": args.style,
        "language": args.language,
        "tts_chunks": chunk_count,
        "sample_rate": sr,
        "raw_tts_seconds": raw_seconds,
        "fitted_audio_seconds": fitted_seconds,
        "tempo_factor": tempo_factor,
        "novelty": asdict(best.novelty),
        "param_plan": asdict(best.params),
        "args": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in {"warning"}
        },
    }
    plan_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    history_db.add(
        warning_hash=warning_hash,
        video_seconds=video_seconds,
        script=best.script,
        exaggeration=best.params.exaggeration,
        cfg_weight=best.params.cfg_weight,
        estimated_seconds=best.params.estimated_seconds,
        actual_seconds=raw_seconds,
        metadata=metadata,
    )
    history_db.close()

    log(f"Done. Video: {output_video}")
    log(f"Narration: {script_path}")
    log(f"Plan: {plan_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise

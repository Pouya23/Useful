"""Pinned F5/StyleTTS2 voice and control documentation for notebook authoring.

This module contains documentation and JSON-safe catalog data only. It does not
load models, download assets, or alter the standalone generation scripts.
"""

from __future__ import annotations


def _option(key: str, default: object, accepted: str, effect: str) -> dict:
    return {"key": key, "default": default, "accepted": accepted, "effect": effect}


def _common(output_dir: str) -> list[dict]:
    return [
        _option("config", None, "Optional JSON file; standalone CLI only", "Loads argparse destination keys; explicit CLI arguments take precedence. Notebook writes this file for you. Unknown keys and incorrect JSON types fail validation."),
        _option("text", None, "Exactly one of text/text_file; 10–500 whitespace-separated words", "Literal text to speak; use TEXT_CASES in the notebook. Text affects pronunciation, native punctuation pauses, chunking, and the plan."),
        _option("text_file", None, "UTF-8/UTF-8-BOM file; mutually exclusive with text", "Reads the complete passage from a file. The supervisor creates this from TEXT_CASES."),
        _option("repo", None, "Required path to the exact pinned source checkout", "Inference implementation. The notebook installs and supplies it; a different revision or modified checked model source is rejected."),
        _option("n", 1, "Integer ≥1", "Number of takes; set global CONFIG['n'] or profile-level n. These models generate takes/chunks sequentially within their worker."),
        _option("seed", 42, "Integer; notebook accepts 0 through 2**63−n", "Base random seed; take i uses seed+i and each chunk has a reproducible derived seed. Reproducibility is conditional on the same software/hardware/settings."),
        _option("device", "cpu", "cpu or available cuda:N", "Standalone compute device. The notebook selects a physical GPU, exposes it to the worker, and supplies cuda:0; do not put device in a profile."),
        _option("output_dir", output_dir, "Fresh output directory", "Writes WAVs, manifest, and previews/plans. Existing run artifacts are rejected. The notebook owns naming and directories."),
        _option("sample_rate", 24000, "Integer 8000–192000 Hz", "Export rate after deterministic polyphase resampling from 24 kHz. Increasing it does not create new acoustic detail. Final duration uses the exact exported sample count, including resampling rounding."),
        _option("pause_ms", 0, "Finite ≥0; cannot be positive with crossfade_ms", "Inserts this many milliseconds of zero-valued samples between generated chunks. It adds to exported duration; it does not change pauses already generated inside a chunk."),
        _option("crossfade_ms", 0, "Finite ≥0; cannot be positive with pause_ms", "Linearly overlaps neighboring chunks, subtracting the actual overlap from exported length. Overlap is capped by available samples; large overlaps can superimpose words."),
        _option("fade_ms", 5, "Finite 0–20 ms", "Linear amplitude fade at both ends of every native chunk, capped at half its length. It changes no sample counts. A large fade can weaken an initial/final phoneme; zero disables it."),
        _option("max_vram_gb", 5, "Finite 0<value≤5 decimal GB; CUDA also requires >0.75 GB", "GPU allocator budgeting with a 0.75 GB reserve, plus observed peak checks. CUDA library allocations exist outside the allocator; sampled monitoring is not an instantaneous hard-peak certificate. Notebook also supervises NVML externally."),
        _option("estimate_only", False, "Boolean; --estimate-only / --no-estimate-only", "Writes the exact preview (and StyleTTS2 tensor plan) without waveform generation. The regular notebook benchmark generates audio, so the supervisor controls this key."),
        _option("wav_subtype", "PCM_24", "PCM_16, PCM_24, FLOAT", "WAV sample encoding: signed 16-/24-bit PCM or 32-bit float. This changes precision/file size, not rate, sample count, or generation style."),
        _option("peak_limit", 0.98, "Finite 0<value≤1", "If the final resampled waveform exceeds this absolute peak, applies one uniform gain ≤1. It neither boosts quiet audio nor changes duration; the gain and original peak are logged."),
    ]


F5_OPTIONS = _common("f5_samples") + [
    _option("reference_audio", None, "Required finite nonsilent 1–8 s audio; shared notebook reference must be mono", "Speaker/style prompt. Standalone reading averages multiple channels and resamples to 24 kHz; no automatic silence trimming. Clean speech from one speaker gives a more useful prompt than music, overlapping speakers, clipping, or a cut word."),
    _option("reference_text", None, "Required exact nonempty transcript", "Conditions the spoken reference together with the waveform. Adapter appends a terminal period if needed and a separating space. Its UTF-8 byte length, with the reference duration, also defines the allocation speaking-rate estimate."),
    _option("checkpoint", None, "Optional local official-layout F5TTS_v1_Base EMA .safetensors", "Replaces the pinned generator file. Architecture is fixed in this adapter and load is strict; Small, E2TTS, non-EMA layouts, and arbitrary unrelated checkpoints are not interchangeable."),
    _option("vocab", None, "Optional local matching vocabulary file", "Text token-to-ID mapping; must match the generator. Otherwise uses the pinned F5TTS_v1_Base vocab. A different vocabulary is not a language-quality guarantee."),
    _option("vocos_dir", None, "Optional directory containing compatible config.yaml and pytorch_model.bin", "Offline Vocos resources. The validated decoder uses vocos==0.1.0 and centered 24 kHz/256-hop output; incompatible temporal topology is rejected."),
    _option("speed", 1, "Finite >0", "Divides the reference-derived generated-frame allocation: >1 requests shorter/faster speech, <1 longer/slower. It changes planning, not post-generation playback. Token/reference clamps and chunk boundaries prevent a universal 1/speed duration ratio."),
    _option("duration_seconds", None, "Optional finite >0 seconds", "Requests a TOTAL generated-content allocation divided among chunks by UTF-8 byte length, excluding reference audio and joins. Frame quantization and token/reference minimums may increase/change it: the reported preview, not this raw target, is the final duration. With this set, speed still affects chunk splitting but no longer determines each chunk's duration allocation."),
    _option("steps", 32, "Integer ≥2; notebook quality profile 64", "Flow-integration intervals. More intervals cost generation time and may improve quality; they do not alter the frozen allocated frame count. Improvement is not guaranteed to be monotonic."),
    _option("cfg_strength", 2, "Finite ≥0", "Classifier-free guidance strength. The pinned flow combines conditional/unconditional predictions as conditional + strength*(conditional−unconditional); near zero uses only the conditional prediction. Stronger guidance can change adherence and sound quality, but leaves the allocation fixed."),
    _option("sway", -1, "Finite −1 through 1", "Warps integration times after EPSS/uniform schedule: t'=t+sway*(cos(pi*t/2)−1+t). Negative values concentrate evaluation toward early flow time; zero removes this warp. Changes generation behavior, not output length."),
    _option("ode_method", "euler", "euler, midpoint, rk4", "ODE solver: approximately 1, 2, or 4 transformer evaluations per interval respectively. Higher-order solvers are more expensive; their quality is an empirical choice. The progress bar counts evaluations; length stays fixed."),
    _option("no_epss", False, "Boolean; CLI flag --no-epss sets true", "True selects uniformly spaced times BEFORE the optional sway warp. False enables pinned Empirically Pruned Step Sampling at 5/6/7/10/12/16 steps; other step counts already fall back to uniform times. At the notebook's 64 steps, toggling this does not change the pre-sway schedule."),
    _option("target_rms", 0.1, "Finite >0", "If the reference RMS is below this value, boosts conditioning to this RMS, then reverses that gain on generated audio. It is reference-level handling, not final loudness mastering or an emotion/intensity control. Final peak protection still applies."),
    _option("max_chunk_seconds", 6, "Finite 0.2–12 s", "Bounds generated allocation per chunk; also determines the reference-derived byte budget. A clamped allocation beyond this limit (with a one-hop allowance) fails rather than truncating speech. Smaller chunks can reduce VRAM but create more joins and reset context."),
    _option("max_chunk_bytes", 160, "Integer ≥8 UTF-8 bytes", "Hard text byte budget per chunk, combined with the time-derived budget. Sentence/clause boundaries are preferred; tiny final tails are merged/rebalanced when possible. A single over-budget word fails. UTF-8 bytes are not words, especially for Chinese."),
]

STYLE_OPTIONS = _common("styletts2_samples") + [
    _option("reference_audio", None, "Required finite nonsilent 0.5–15 s audio; shared notebook reference: mono 1–8 s", "Speaker/style prompt; no transcript needed. Standalone reading averages channels and resamples to 24 kHz without silence trimming. Acoustic and prosodic encoders extract a 256-dimensional reference style."),
    _option("checkpoint", None, "Optional compatible official-layout LibriTTS .pth checkpoint", "Replaces the generator weights. Inference modules load strictly; single-speaker LJSpeech and unrelated architectures are not drop-in alternatives for this adapter."),
    _option("model_config", None, "Optional matching official LibriTTS config.yml", "Architecture configuration paired with the checkpoint. It is not a per-generation style configuration. The adapter requires the validated HifiGAN temporal topology."),
    _option("reuse_plan", None, "Optional duration_plan.pt from an identical run setup", "Reuses cached sampled styles, integer durations, and duration-conditioned embeddings. Fingerprint includes text, reference/checkpoint/config contents, seed/takes, style, pitch, chunk, join, and export settings. Use a fresh output_dir; changed settings fail validation."),
    _option("speed", 1, "Finite >0", "Divides each predicted phoneme duration BEFORE rounding, then clamps every duration to at least one unit. >1 usually shortens/faster; <1 lengthens/slower. Rounding/minimums mean the total is not exactly proportional to 1/speed."),
    _option("alpha", 0.1, "Finite 0–1", "Acoustic/timbre blend: alpha*sampled_acoustic + (1−alpha)*reference_acoustic. Zero maximizes direct reference timbre; one uses sampled acoustic style. Higher alpha can reduce speaker similarity; it is not an emotion label."),
    _option("beta", 0.3, "Finite 0–1", "Prosody blend: beta*sampled_prosody + (1−beta)*reference_prosody. Zero favors the reference delivery; one favors text-conditioned sampled prosody. It can change phoneme durations, expressivity, and similarity, so exact preview must be recomputed."),
    _option("diffusion_steps", 10, "Integer ≥2", "ADPM2 latent-style sampling steps. Affects preflight cost, sampled style, and therefore potentially the duration plan. This is not waveform diffusion; the cached style is reused during decoding."),
    _option("embedding_scale", 1, "Finite >0", "Text-embedding classifier-free guidance in style sampling: fixed_embedding_prediction + scale*(text_prediction−fixed_embedding_prediction). At one, uses the normal text-conditioned path; larger values emphasize its difference and add conditional/unconditional computation. Can affect style and planned duration."),
    _option("style_continuity", 0.7, "Finite 0–1", "On chunks after the first, blends previous FINAL blended style with the new sampled style: continuity*previous + (1−continuity)*new. Zero removes that carryover; one carries the previous style into this intermediate blend. Alpha/beta reference blending is still applied afterward, so one does not freeze every final chunk style."),
    _option("pitch_scale", 1, "Finite >0", "Multiplies predicted F0 immediately before HifiGAN decoding, without shifting durations. >1 usually higher pitch; <1 lower. This is a model-F0 multiplier, not pitch-preserving playback speed; extreme values can hurt naturalness and speaker identity."),
    _option("max_chunk_words", 35, "Integer 1–80; notebook quality profile 45", "Initial word budget, with preferred sentence/clause endings. Smaller chunks lower context/VRAM requirements but add joins and may weaken sentence prosody."),
    _option("max_chunk_tokens", 400, "Integer 16–500 phoneme tokens", "Further recursively splits chunks that exceed the phonemized token budget, preferring sentence/clause boundaries near the split. Includes the initial special token. One over-budget word fails."),
    _option("max_chunk_seconds", 20, "Finite 1–30 s", "When building a NEW plan, rejects a chunk whose predicted native audio exceeds this limit before decoding; it does not automatically resplit. Use smaller word/token chunks or an appropriate faster speed. Existing reuse_plan data bypasses this check, and this setting is currently absent from the cache fingerprint: retain the original limit when reusing a plan."),
    _option("tail_trim_samples", 50, "Integer ≥0 native 24 kHz samples", "Fixed trim at every chunk's decoder tail, matching the official demo's end-pulse workaround. Default is about 2.083 ms per chunk. Zero retains full decoder output. It is included in the preview and is not content-dependent silence removal; a large trim can remove real speech or make a plan invalid."),
    _option("language", "en-us", "en-us, en-gb", "Selects the eSpeak English phonemizer's pronunciation variety. The same English LibriTTS checkpoint is used; this does not turn it into a multilingual model or guarantee a British voice identity."),
]


STYLE_DEMO_REFERENCES = [
    {"group": "seen-speaker examples", "file": "696_92939_000016_000006.wav"},
    {"group": "seen-speaker examples", "file": "1789_142896_000022_000005.wav"},
    {"group": "unseen-speaker examples", "file": "1221-135767-0014.wav"},
    {"group": "unseen-speaker examples", "file": "5639-40744-0020.wav"},
    {"group": "unseen-speaker examples", "file": "908-157963-0027.wav"},
    {"group": "unseen-speaker examples", "file": "4077-13754-0000.wav"},
    {"group": "zero-shot examples", "file": "3.wav"},
    {"group": "zero-shot examples", "file": "4.wav"},
    {"group": "zero-shot examples", "file": "5.wav"},
    {"group": "delivery examples", "file": "anger.wav"},
    {"group": "delivery examples", "file": "sleepy.wav"},
    {"group": "delivery examples", "file": "amused.wav"},
    {"group": "delivery examples", "file": "disgusted.wav"},
    {"group": "named examples", "file": "Yinghao.wav"},
    {"group": "named examples", "file": "Gavin.wav"},
    {"group": "named examples", "file": "Vinay.wav"},
    {"group": "named examples", "file": "Nima.wav"},
]

VOICE_CATALOG = {
    "f5": {
        "checkpoint": "F5TTS_v1_Base/model_1250000.safetensors",
        "source_revision": "283252563dbf91be625e0c27926acfaac449186c",
        "weight_revision": "84e5a410d9cead4de2f847e7c9369a6440bdfaca",
        "voice_mode": "open-ended reference audio; no finite speaker-ID bank",
        "arbitrary_reference_audio": True,
        "reference_transcript_required": True,
        "reference_seconds": [1, 8],
        "native_sample_rate": 24000,
        "checkpoint_languages": ["English", "Mandarin Chinese"],
        "notebook_fallback": "Kokoro am_fenrir synthetic reference; not an F5 built-in voice",
        "source_example_references": [
            "src/f5_tts/infer/examples/basic/basic_ref_en.wav",
            "src/f5_tts/infer/examples/basic/basic_ref_zh.wav",
            "src/f5_tts/infer/examples/multi/main.flac",
            "src/f5_tts/infer/examples/multi/town.flac",
            "src/f5_tts/infer/examples/multi/country.flac",
        ],
        "controls": F5_OPTIONS,
    },
    "styletts2": {
        "checkpoint": "Models/LibriTTS/epochs_2nd_00020.pth",
        "source_revision": "5cedc71c333f8d8b8551ca59378bdcc7af4c9529",
        "weight_revision": "3aa7ba7f8f275ec13dce21682a61494c35089e2a",
        "voice_mode": "open-ended reference audio; no finite speaker-ID bank",
        "arbitrary_reference_audio": True,
        "reference_transcript_required": False,
        "standalone_reference_seconds": [0.5, 15],
        "notebook_global_reference_seconds": [1, 8],
        "native_sample_rate": 24000,
        "checkpoint_languages": ["English"],
        "phonemizer_variants": ["en-us", "en-gb"],
        "notebook_fallback": "Kokoro am_fenrir synthetic reference; not a StyleTTS2 built-in voice",
        "official_demo_reference_files": STYLE_DEMO_REFERENCES,
        "reference_bundle_url": "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/3aa7ba7f8f275ec13dce21682a61494c35089e2a/reference_audio.zip",
        "controls": STYLE_OPTIONS,
    },
}


def _table(options: list[dict]) -> str:
    rows = ["| Config key | Standalone default | Accepted values | Exact effect |", "|---|---|---|---|"]
    for option in options:
        value = option["default"]
        default = "unset" if value is None else repr(value)
        cells = [f"`{option['key']}`", f"`{default}`", option["accepted"], option["effect"]]
        rows.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    return "\n".join(rows)


GUIDE_MARKDOWN = r"""
## F5-TTS v1 Base: voices and complete adapter configuration

**Voice choices are open-ended:** supply speech from the speaker you want using `reference_audio` and its exact `reference_text`. There is no finite list of F5 speaker IDs and no F5-native serious-male preset. The notebook's male fallback is a disclosed Kokoro `am_fenrir` reference. It is a convenience prompt, and copying a synthetic prompt does not establish the same quality as a clean human reference.

The adapter accepts a finite, nonsilent **1–8 second** reference. The notebook's shared reference validator additionally requires **mono** audio; the standalone script averages channels and resamples to 24 kHz. Preserve complete words and a natural ending; the adapter does not trim silence or automatically transcribe the prompt. The checkpoint covers **English and Mandarin Chinese**. Community language checkpoints are separate models, not additional built-in voices, and compatibility with this fixed v1 Base architecture is not automatic. [Official language/checkpoint list](https://github.com/SWivid/F5-TTS/blob/283252563dbf91be625e0c27926acfaac449186c/src/f5_tts/infer/SHARED.md)

Optional upstream example recordings are `basic_ref_en.wav`, `basic_ref_zh.wav`, and the multi-voice examples `main.flac`, `town.flac`, `country.flac` under `src/f5_tts/infer/examples/`. They are WAV/FLAC prompts, not a speaker-embedding bank. Inspect their duration and obtain the exact transcript before using them in this adapter. The English basic example's supplied transcript is “Some call me nature, others call me mother nature.” [Official examples and reference guidance](https://github.com/SWivid/F5-TTS/blob/283252563dbf91be625e0c27926acfaac449186c/src/f5_tts/infer/README.md)

The following table covers **every CLI/config destination**, including operational/resource fields; `config` itself is a CLI-only file selector and cannot occur inside that JSON. Use underscore keys inside `MODEL_PROFILES['f5'][...]['parameters']`; standalone CLI flags use hyphens. The notebook owns `config`, `text`, `text_file`, `device`, `n`, `seed`, `output_dir`, `max_vram_gb`, and `estimate_only`; profile-level `n` and `seed` are separate supported fields. `repo`, `checkpoint`, `vocab`, and `vocos_dir` are installed automatically but can be replaced by compatible local resource paths. All floating-point settings must be finite. The displayed numeric bounds are validation bounds, not a claim that extreme settings produce intelligible speech.

__F5_OPTIONS__

**Notebook defaults versus script defaults:** the enabled quality profile raises `steps` from **32 to 64**; global defaults use **2 takes**, seed **1234**, **48 kHz export**, **0 added pause**, and assigned CUDA workers. Other F5 settings inherit the script defaults shown above unless a profile overrides them.

**What the exact duration means:** the planner freezes a frame allocation derived from reference speaking rate or `duration_seconds`, then applies the model's required text/reference minimum. Native chunk length is `(allocated_frames − floor(reference_samples/256) − 1) * 256`. The preview accounts for joins and export-rate rounding. Every take shares that allocation; noise, solver, steps, guidance, and sway can change the sound without changing its length. This is an exact allocation contract, not an independent forecast of the voice's unconstrained natural speaking time. A target that is too short may make pronunciation poor even when the sample count is exact. [Pinned flow implementation](https://github.com/SWivid/F5-TTS/blob/283252563dbf91be625e0c27926acfaac449186c/src/f5_tts/model/cfm.py)

**Upstream features outside this adapter:** automatic reference ASR, the upstream multi-voice `[speaker]` text markup, speech editing, no-reference experiments, alternate E2/Small architectures, BigVGAN, and content-based silence removal are not parameters here. Multiple reference voices can be tested through separate notebook profiles, one prompt per profile. There is no independent pitch, emotion-label, temperature/top-p, or exact word-emphasis control. Delivery can be influenced by the reference and punctuation, with empirical quality limits. Source is MIT; the official pretrained F5 weights are CC-BY-NC-4.0. [Pinned model card](https://huggingface.co/SWivid/F5-TTS/blob/84e5a410d9cead4de2f847e7c9369a6440bdfaca/README.md)

## StyleTTS2 LibriTTS: voices and complete adapter configuration

**Voice choices are open-ended:** give `reference_audio` from the desired speaker; **no transcript is required**. There is no speaker-ID selector in this adapter. The notebook's fallback is the same disclosed Kokoro `am_fenrir` recording, not a StyleTTS2 male preset. Standalone input accepts finite nonsilent **0.5–15 second** references and averages channels; the shared notebook path is deliberately stricter at **mono, 1–8 seconds**. References are resampled to 24 kHz and not silence-trimmed.

The installed model is the **English multi-speaker LibriTTS** checkpoint. `language='en-us'/'en-gb'` switches English pronunciation rules, not model weights. Other StyleTTS2 checkpoints, including single-speaker LJSpeech, are not interchangeable with this adapter's fixed inference topology. [Official model/inference documentation](https://github.com/yl4579/StyleTTS2/blob/5cedc71c333f8d8b8551ca59378bdcc7af4c9529/README.md)

The official demonstration references are listed below. These are optional **recording filenames**, not the complete universe of clonable voices or guaranteed serious/deep male choices. The emotional filenames refer to recordings with that delivery; they do not introduce an emotion-label API. Download the [pinned reference bundle](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/3aa7ba7f8f275ec13dce21682a61494c35089e2a/reference_audio.zip) manually if wanted, inspect each clip against the input constraints, and point a profile to its extracted audio file. The benchmark does not automatically test all examples. [Pinned official demo](https://github.com/yl4579/StyleTTS2/blob/5cedc71c333f8d8b8551ca59378bdcc7af4c9529/Demo/Inference_LibriTTS.ipynb)

| Demonstration group | Reference recording filenames |
|---|---|
| Seen speakers | `696_92939_000016_000006.wav`, `1789_142896_000022_000005.wav` |
| Unseen speakers | `1221-135767-0014.wav`, `5639-40744-0020.wav`, `908-157963-0027.wav`, `4077-13754-0000.wav` |
| Zero-shot examples | `3.wav`, `4.wav`, `5.wav` |
| Delivery examples | `anger.wav`, `sleepy.wav`, `amused.wav`, `disgusted.wav` |
| Named examples | `Yinghao.wav`, `Gavin.wav`, `Vinay.wav`, `Nima.wav` |

The table below covers **every CLI/config destination**; `config` is CLI-only. Config keys use underscores; standalone flags use hyphens. Supervisor-controlled keys and per-profile take/seed fields work as described for F5 above. All floating-point settings must be finite; acceptance is not an audio-quality guarantee.

__STYLE_OPTIONS__

**Notebook defaults versus script defaults:** the enabled quality profile uses `max_chunk_words=45` rather than the script's **35**. Both use `alpha=0.1`, `beta=0.3`, `diffusion_steps=10`, `embedding_scale=1`, and `pitch_scale=1`. Global notebook settings supply 2 takes, seed 1234, 48 kHz export, zero added gap, and CUDA workers. An optional expressive profile is disabled until you enable it.

**Duration and style interaction:** exact preflight actually runs BERT, latent style diffusion, and the duration predictor for **every requested take**. It caches sampled styles and integer phoneme durations for reuse; it is heavier than F5's arithmetic allocation and is not universally near-instant. Each native chunk has `sum(integer_durations)*600 − tail_trim_samples` samples; joins/resampling are counted afterward. Changing seed, style weights, guidance, continuity, steps, reference, speed, or pronunciation can change the plan. `pitch_scale` changes F0 during decoding without changing that duration contract, although the saved plan fingerprint conservatively includes it.

For a serious/deeper rendition, start with a clean reference that already has that timbre and delivery. `alpha=0` maximizes direct reference acoustic conditioning; low `beta` favors its prosody. Moderate `pitch_scale<1` can lower predicted F0 but cannot guarantee a convincing deeper speaker. Higher `beta` gives more sampled text-conditioned delivery; it does not select a named emotion. The text-embedding guidance formula and its compute cost are in the [pinned diffusion module](https://github.com/yl4579/StyleTTS2/blob/5cedc71c333f8d8b8551ca59378bdcc7af4c9529/Modules/diffusion/modules.py).

**Upstream features outside this adapter:** direct reference-text style-transfer methods, arbitrary emotion labels, per-word SSML emphasis, other languages/checkpoints, sample-wise speaker changes, and custom diffusion samplers/sigma schedules are not exposed. This adapter fixes ADPM2 with Karras sigma min 0.0001, max 3, rho 9, and uses the official LibriTTS HifiGAN conditioning shift. It caches duration decisions rather than modifying generated speech to fit a target. There is no `duration_seconds` override for StyleTTS2.
""".strip()

GUIDE_MARKDOWN = GUIDE_MARKDOWN.replace("__F5_OPTIONS__", _table(F5_OPTIONS)).replace(
    "__STYLE_OPTIONS__", _table(STYLE_OPTIONS)
)

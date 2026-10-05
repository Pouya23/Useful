"""Documentation embedded in the extended notebook; no TTS runtime imports."""

GUIDE_MARKDOWN = r'''
## Voice and configuration reference

This guide is exhaustive for the **six embedded adapters and their pinned
checkpoints**. Numeric settings admit infinitely many combinations; the tables
give every exposed key, its domain, and its effect. Options from unrelated
checkpoints, training pipelines, or other frontends are outside this notebook.
The option inventory is also extracted directly from each adapter's argument
parser, so it includes file/cache/runtime controls as well as synthesis knobs.

| Model | Voice choices in this notebook | Reference WAV? | Explicit emotion / emphasis |
|---|---|---|---|
| Kokoro | 54 shipped presets, compatible `.pt` packs, weighted mixtures | No WAV-to-speaker encoder | No named emotion or word-emphasis control; speed and punctuation affect delivery |
| F5-TTS | Any suitable speaker recording; no finite preset bank | WAV + exact transcript | Reference delivery conditions style; no independent named emotion, pitch, or word-emphasis key |
| EmotiVoice | 2,014 ordered speaker labels in the pinned vocabulary | No | Text style prompt; pitch/energy predictor scales; phoneme-index emphasis |
| StyleTTS2 | Any suitable speaker recording; optional official demo WAVs | WAV only | Reference + sampled style/prosody; continuous style weights and pitch; no named emotion key |
| OmniVoice | Any suitable speaker recording, saved prompt, or supported attribute combinations | WAV + exact transcript, or cached prompt | Voice-design attributes including whisper; no validated named emotion or word-emphasis vocabulary in this pinned release |
| Supertonic 3 | M1–M5 / F1–F5; compatible custom voice-style JSON | No local WAV encoder | Supported expression tags in text; no separate emotion or per-word emphasis flag |

**What “all reference voices” means:** F5, StyleTTS2, and OmniVoice do not restrict
you to a named list of speakers. A valid recording can represent any speaker;
recording suitability and cloning fidelity must be checked by listening. Kokoro
voice packs and Supertonic style JSON are model embeddings, not audio files.
EmotiVoice speaker labels are learned identities, not speaker WAV filenames.

The shared notebook reference field accepts a readable, finite, nonsilent **mono
1–8 second** WAV. Prefer one speaker, steady recording level, little background
noise, and 3–8 seconds of continuous speech. For F5/Omni, transcribe the actual
spoken recording exactly. A transcript is unnecessary for StyleTTS2. Individual
adapter bounds are documented below; the shared notebook uses the narrower
1–8 second contract. Reference preprocessing and model loading are separately
timed and are not part of a warmed duration-only preview.

### How to change settings

Edit `CONFIG`, `TEXT_CASES`, and `MODEL_PROFILES` in the experiment cell. Adapter
keys inside `parameters` use underscores: CLI `--cfg-strength` becomes
`cfg_strength`. JSON/Python booleans must be actual booleans, not strings such as
`"False"`. Omit optional keys to retain their adapter defaults; profile entries
whose value is `None` are omitted before invoking the script.

```python
CONFIG["speaker_mode"] = "auto"
CONFIG["reference_audio"] = "/kaggle/input/my-voice/reference.wav"
CONFIG["reference_text"] = "The exact words spoken in the reference recording."

# Add another take configuration without changing the original profile:
MODEL_PROFILES["kokoro"].append({
    "name": "george_comparison", "enabled": True, "n": 2, "seed": 900,
    "parameters": {"voice": "bm_george", "language": "b", "speed": 0.95},
})
```

The example is a configuration recipe; replace its WAV path and transcript
before using it. It does not upload or download a voice automatically.

Settings merge in this order: global sample rate/pause/take settings → installed
model asset paths → profile `parameters` → reference routing → Omni
`clone_parameters`. Thus a profile can override export rate, pause, voice,
compatible asset paths, and native synthesis controls. The saved per-job
`config.json` is the authoritative resolved configuration. Standalone CLI flags
override a standalone JSON configuration file.

### Every experiment-level setting

These are supervisor settings, distinct from the model parameters below.

| `CONFIG` key | Notebook default / permitted values | Exact purpose |
|---|---|---|
| `install_models` | All six; list of `kokoro`, `f5`, `emotivoice`, `styletts2`, `omnivoice`, `supertonic` | Install and schedule only these models; enabled profiles for omitted models do not run |
| `force_setup` | `False` / boolean | Request reinstall for Torch and the pip-installed Kokoro/Omni source packages; it does not delete all cached assets or force every dependency to reinstall |
| `torch_cuda_variant` | `"auto"`, wheel name, or per-model dictionary | Auto uses cu121 for Torch 2.5.1 models and cu128 for Torch 2.8.0 Omni/Super; 2.5.1 supports cu118/cu121/cu124, 2.8 supports cu126/cu128/cu129; Super must retain cu128 in this setup |
| `strict_cuda_driver` | `False` / boolean | If true, Super setup requires NVIDIA Linux driver >=570.26; runtime CUDA/provider preflight remains necessary either way |
| `n` | `2`, positive integer | Takes per model/profile/text; profile-level `n` overrides it; the notebook rejects invalid signed-64-bit seed ranges |
| `seed` | `1234`, nonnegative integer | Starting seed; takes use consecutive seeds. EmotiVoice is deterministic, so changing seed alone does not create different delivery |
| `sample_rate` | `48000`, integer | Export Hz for all models unless overridden in a profile; synthesis remains at the model's native rate. Adapter-specific bounds apply |
| `pause_ms` | `0.0`, nonnegative number | Extra digital silence between planned chunks; native punctuation/silence remains. Profile override allowed; F5/Style cannot combine a positive pause with a positive crossfade |
| `max_vram_gb` | `5.0`, `(0,5]` | Decimal GB target per GPU; passed to adapters and external monitor. Allocator limits plus NVML polling are not an instantaneous process-wide certificate |
| `nvml_interval_ms` | `10.0`, `[1,1000]` | Requested memory polling interval; actual polling can be slower under load |
| `enforce_observed_budget` | `True` / boolean | Stop workers after a sampled overrun and retain partial files/logs; false records overruns without budget termination |
| `budget_scope` | `"device_total"` or `"process_tree"` | Device total includes baseline/other processes. Process tree counts tracked worker/descendants; if per-process accounting is unavailable it falls back to device total |
| `job_timeout_seconds` | `3600`, positive seconds | Maximum worker lifetime including interpreter startup, model/reference loading, planning, and all takes |
| `gpu_release_timeout_seconds` | `20`, positive seconds | Wait after worker exit for tracked GPU processes to disappear before reusing a slot; release failure disables that slot |
| `cpu_threads_per_job` | `4`, positive integer | Sets OMP/MKL environment limits and CPU quality-helper threads; an adapter's explicit `threads` key can separately configure its own pools |
| `require_two_t4` | `True` / boolean | Require two visible T4s. False allows other two-GPU hardware; it does not change the fixed two-slot scheduler into a single-GPU runner |
| `speaker_mode` | `"auto"`, `"reference"`, `"default"` | Auto uses supplied global WAV for cloning-capable models, else male defaults. Reference requires a global user WAV and blocks preset-only models. Default ignores the global WAV and uses male fallback; explicit per-profile references still work |
| `reference_audio` | `None` or file path | Global reference for F5/Style and requested Omni cloning; if absent in auto/default mode, CPU Kokoro creates a disclosed male reference |
| `reference_text` | `None` or string | Exact global WAV transcript for F5/Omni; Style needs only the WAV |
| `include_reference_in_zip` | `True` / boolean | Include the shared reference WAV and transcript in the final ZIP; per-profile external WAVs are not automatically collected by this switch |
| `preview_players` | `6`, nonnegative integer | Maximum distinct models shown as listening players in the notebook; all created WAVs still enter the ZIP |
| `quality_checks` | `True` / boolean | Run CPU acoustic diagnostics before packaging; false skips them and ASR |
| `asr_quality_check` | `True` / boolean | Add independent CPU/int8 ASR comparison; adds elapsed time without TTS GPU allocation |
| `asr_model` | `"base.en"`, supported faster-whisper name or CTranslate2 path | Base English model is revision-pinned. `small.en` is a larger English alternative; use multilingual models for other languages. Other remote names are not revision-pinned by this notebook |
| `asr_language` | `"en"` or recognizer-supported language code | Force the independent recognizer's language; this does not set TTS language or translate the text |
| `quality_timeout_seconds` | `7200`, positive seconds | Timeout for the quality-helper subprocess; dependency/model downloads occur before that subprocess and are not included in this timeout |

### Texts and profile fields

| Field | Meaning |
|---|---|
| `TEXT_CASES` | List of `{name, text}` objects; each stripped text must have 10–500 whitespace-separated words. Names identify cases; complete paragraphs and punctuation are preserved |
| `WORD_COUNTS` / `SAMPLE_PASSAGES` | Convenience defaults for the three supplied 10/100/500-word texts. Other counts require your own full text; the notebook does not repeat or cut the supplied passages |
| Profile `name` | Human-readable experiment name, sanitized into filenames; ordinal/text/config hashes prevent collisions even if names repeat |
| Profile `enabled` | Boolean, default true when omitted; disabled profiles create no jobs |
| Profile `parameters` | Adapter destination keys documented below; omitted keys use script defaults. A different profile is a separate model-loading job |
| Profile `n`, `seed` | Override global take count/seed; put them at profile level rather than inside `parameters` |
| Omni profile `reference` | `"auto"` / `True` / `False`; auto follows the global reference choice, true requests cloning (fallback reference is possible), false selects design unless an explicit reference/cache or global reference-only mode requires cloning |
| Omni profile `clone_parameters` | Optional adapter overrides applied only when cloning; use this for clone-specific controls. Default design `instruct` is removed unless `clone_parameters` supplies it |

The notebook owns `config`, `text`, `text_file`, `device`, `n`, `seed`,
`output_dir`, `max_vram_gb`, `estimate_only`, and `self_test`. Do not place them
inside `parameters` or `clone_parameters`; use their supervisor/profile controls
or standalone CLI. Every GPU worker sees only its assigned physical GPU and uses
logical `cuda:0`. Up to two jobs run together; different model names are preferred
when the queue permits it.

### Duration and sound quality: effects shared by the models

- `speed > 1` generally shortens duration; `< 1` lengthens it. This changes native
  duration conditioning/allocation, not the WAV playback rate. Frame rounding,
  minimum durations, and text/reference clamps can prevent exact inverse scaling.
- F5/Omni explicit duration selects a synthesis allocation before waveform
  generation. It is quantized and excludes configured joins; it is not an
  independent prediction of natural unconstrained reading time. Allocations
  that are too short can omit words even when the total sample count is exact.
- Voices, reference preprocessing, emotion/style, seed, text normalization,
  chunking, speed, and joins can affect duration where they enter that model's
  planner. Changing any relevant configuration requires a fresh matching plan.
- Sampling steps usually change quality/runtime rather than the fixed allocated
  output length. StyleTTS2's style diffusion occurs before duration prediction,
  so its sampled style can also change the plan.
- `sample_rate` changes export sampling density, not native acoustic bandwidth;
  peak attenuation/gain and short edge fades change amplitude without changing
  length. Crossfades overlap audio and reduce length; pauses add length. Both
  effects are accounted for by the applicable adapter's plan.
- Native rates: Kokoro/F5/Style/Omni 24 kHz, EmotiVoice 16 kHz, Supertonic 44.1 kHz.
  PCM_16/PCM_24 set integer storage precision; FLOAT preserves floating samples
  and may be less convenient for some players. These are not neural quality knobs.
- ASR omissions/substitutions are fallible screening results, not a perceptual
  quality guarantee. Silence diagnostics distinguish interior quiet spans from
  leading/trailing silence; intended punctuation pauses can still be flagged.

The new catalog section can be inspected and exported **without loading model
weights**. It lists all shipped speaker rows and every parser option. Catalog
viewing does not enable all speakers as experiments: default runs remain two
takes for each enabled profile and each of the three passages.
'''

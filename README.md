# Standalone speech sampling with duration previews

Six independent programs; none imports another sampling program or a shared library.
Each accepts UTF-8 text, configuration flags or a JSON defaults file, `--n`,
`--seed`, `--estimate-only`, an export sample rate, explicit pauses, and an output
directory. They display generation progress and compare planned samples with
the samples actually written to WAV. Use a fresh output directory per run.

For Kaggle **GPU T4 x2**, use the self-contained
`kaggle_dual_t4_tts_benchmark.ipynb`. It embeds all six programs, installs separate
model environments, runs paired GPU jobs, monitors NVML usage, and produces a
ZIP with named WAVs, CSV/JSON/HTML reports, charts, memory traces, and logs.
Select GPU T4 x2 and enable Internet before running. GPU execution is tested by
the notebook run; local validation did not have NVIDIA hardware.

For the complete voice and configuration reference, use the new
`kaggle_dual_t4_tts_voice_configuration_guide.ipynb`. It preserves the benchmark
workflow and embeds all 54 Kokoro, 2,014 EmotiVoice, and 10 Supertonic preset
records, reference-WAV guidance for F5/Style/Omni, all 185 explicit adapter
options, supervisor controls, and detailed duration algorithms with code
locations. Its catalog explorer filters speaker metadata, exports complete CSVs,
and includes the guide/catalogs in the final WAV/report ZIP. Catalog inspection
requires no model downloads or inference. The previous notebook is preserved.

The planned durations are computed before waveform synthesis. Learned phoneme
durations (Kokoro/Emoti/Style), chosen frame/token allocations (F5/Omni), or a
learned seconds/latent allocation (Supertonic) are frozen and reused. Decoder
geometry plus deterministic joins/resampling makes successful exports
sample-exact; the later independent WAV-header check verifies that contract.
For F5/Omni, zero duration error is not proof that a heuristic predicts
unconstrained natural reading time. Word completeness is checked separately.

The quality revision uses sentence/clause boundaries, no added chunk pauses,
small length-preserving seam fades, and global attenuation to prevent export
clipping. It fixes EmotiVoice's missing final phoneme after an unknown word and
F5's unexpected slowdown of short chunks. Higher sampling steps and male voice
profiles are configurable. These repairs do not guarantee that every model will
speak every text correctly; independent acoustic and CPU ASR checks now report
gaps, transcripts, and word edits separately from duration accuracy.

Set `speaker_mode="auto"`, `reference_audio`, and `reference_text` in the notebook
to clone your speaker with F5, StyleTTS2 and OmniVoice. Style needs only the WAV;
the other two also need its exact transcript. `speaker_mode="reference"` reports
the preset-only models as unsupported. Without a user reference, the defaults
are Kokoro `am_fenrir`, EmotiVoice `9017/Neutral`, Supertonic `M2`, and OmniVoice
male/low-pitch design; F5/Style use a disclosed male synthetic reference.
Supertonic's published M2 description specifically matches a deep, serious voice.

The nonrepetitive example files are `sample_text_10.txt`, `sample_text_100.txt`,
and `sample_text_500.txt`. Their whitespace word counts are exactly 10, 100,
and 500. The notebook embeds these same passages, including paragraph breaks.

No new TTS backend was added in this revision: the identified pipeline defects
are repairable, and reference cloning already exists in three of the six models.
Other cloning models examined lack a verified fast exact-duration preflight in
their released path: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py)
generates until codec EOS; [Chatterbox](https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/tts.py)
first samples speech tokens; [IndexTTS 2.5](https://github.com/index-tts/index-tts/blob/main/indextts/infer_v2_5.py)
applies its duration factor after semantic token generation. This is an inference
from the released code, not a claim about all future versions or modified research models.

The workspace already contains verified Kokoro/Supertonic assets and an isolated
Python 3.12 CPU environment. From this directory, these two examples can run now:

```powershell
.\.verify_venv\Scripts\python.exe kokoro_samples.py --config kokoro.example.json
.\.verify_venv\Scripts\python.exe supertonic_samples.py --config supertonic.example.json
```

Add `--estimate-only` for preview mode. If a Supertonic output already exists,
provide a new `--output-dir`. The example JSON files reference local validation
assets; use the installation instructions below when copying the programs to
another machine.

## What the duration preview means

The requested quantity is **total exported WAV duration**, including generated
silence, configured pauses, fixed processing, and resampling. The computation is
`planned_export_samples / export_sample_rate`. It does not rely on a words-per-minute
regression. Cached model durations or explicitly allocated frames/tokens are
reused during generation. An unexpected length causes an error instead of a
stretched, padded, or content-trimmed output that conceals the mismatch.

| Program | Planning method | Native rate | Voice/style controls |
|---|---|---:|---|
| `kokoro_samples.py` | Actual voice-conditioned duration network; 600 samples per duration unit | 24,000 | Presets, compatible local voice packs, weighted preset mixtures, speed |
| `f5_samples.py` | Allocate frames using the supplied reference or a requested duration; include the model's text/reference length clamp | 24,000 | Reference voice/style, speed or duration, flow sampling controls |
| `emotivoice_samples.py` | Actual speaker/style-conditioned phoneme durations; decoder geometry propagated exactly | 16,000 | Speaker IDs, emotion/style prompt, speed, predictor pitch/energy scales, phoneme emphasis |
| `styletts2_samples.py` | Sample and cache style, embeddings, and actual phoneme durations **for each take** | 24,000 | Reference voice, acoustic/prosodic style weights, style continuity, speed, pitch scale |
| `omnivoice_samples.py` | Allocate audio tokens using upstream rules or a requested duration, then supply those exact lengths to synthesis | 24,000 | Voice design instructions or reference voice, speed/duration, iterative sampling controls |
| `supertonic_samples.py` | Actual voice-conditioned duration predictor; reuse the official latent allocation and predetermined padding removal | 44,100 | Voice styles, language, speed, flow steps, supported text expression tags |

F5 and OmniVoice preview a **chosen synthesis allocation**. They do not independently
predict how long an unconstrained reading would naturally take. An excessively
short allocation can impair intelligibility even though the WAV length is exact.

Near-instant preview means the planning step after dependencies, model loading,
and reference preprocessing. The programs report these costs separately. OmniVoice
auto/design preview avoids loading Torch and model weights. StyleTTS2 needs style
diffusion before its exact duration is known, so its preview is heavier; no
universal near-instant latency claim is made for it.

The user-supplied Kaggle run reported 30 exact-length exports across five models;
F5 failed an eager training-module import because installation lacked `wandb`.
The revised environment includes all eager dependencies, including Accelerate,
EMA and Datasets. Revised CPU Kokoro/Supertonic runs at 10 and 500 words retain
zero duration error. Kokoro's 500-word CPU preflight remains several seconds,
while Supertonic and Omni's allocation paths are faster. Updated GPU inference
and perceptual quality still require the new Kaggle run.

Output resampling changes the file's sampling frequency without increasing the
model's native acoustic bandwidth. Emotion labels and arbitrary cloned voices
are supported only where the underlying model provides them; unsupported knobs
are not simulated. EmotiVoice is deterministic: identical settings produce
identical repeated takes. Kokoro's seed changes decoder noise, while other models
also have stochastic generation paths.

## GPU memory

Every program defaults to CPU and therefore allocates **zero GPU VRAM**. All
programs offer optional CUDA with conservative
allocator budgets, bounded sequential chunks, and reported allocator peaks.
EmotiVoice and Supertonic require `--allow-unverified-cuda-budget` to make the limitation explicit.

`--max-vram-gb 5` means **5,000,000,000 bytes**, not 5 GiB. PyTorch's allocator cap
does not constrain CUDA context, driver, or external library allocations. Optional
sampled process measurements also cannot prove an instantaneous universal peak.
**CUDA operation below a hard total 5 GB peak has not been certified on this
machine.** Use CPU if that ceiling must hold without any GPU-specific validation.

## Installation

Use Python **3.11 or 3.12**, preferably a separate virtual environment for each
model: their upstream dependency constraints differ. Install a matching Torch
and Torchaudio pair for your platform when required. For CPU wheels, use PyTorch's
CPU index. Do not install all upstream requirements into the existing narration
environment.

Run the examples below from this directory. Replace `passage.txt`, reference
audio, transcripts, and model paths with your files. `--help` lists every supported
setting. All reference transcripts must accurately match their reference audio.

### Kokoro

```powershell
python -m pip install "git+https://github.com/hexgrad/kokoro.git@dfb907a02bba8152ca444717ca5d78747ccb4bec" "misaki[en]" transformers==5.3.0 scipy soundfile
python kokoro_samples.py --text-file passage.txt --voice am_fenrir --speed 1.0 --sample-rate 44100 --pause-ms 0 --n 3 --output-dir outputs/kokoro
python kokoro_samples.py --text-file passage.txt --voice am_fenrir --speed 1.0 --sample-rate 44100 --pause-ms 0 --estimate-only --output-dir previews/kokoro
```

Weights and voice packs download from the pinned Hugging Face revision. Local
`--checkpoint` and `--model-config` overrides must preserve the supported decoder
architecture. `--voice "af_heart,af_bella" --voice-weights "0.7,0.3"` mixes compatible
presets. Non-English use needs the corresponding Misaki language extras and G2P
dependencies. Complete published inference weights are accessible; Kokoro's
unpublished original training style encoder is not included in this release.

### F5-TTS v1 Base

```powershell
git clone https://github.com/SWivid/F5-TTS.git models/F5-TTS
git -C models/F5-TTS checkout 283252563dbf91be625e0c27926acfaac449186c
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install scipy soundfile tqdm safetensors huggingface_hub vocos==0.1.0 pypinyin rjieba librosa einops x_transformers==2.16.2 torchdiffeq wandb==0.19.8 accelerate==1.6.0 ema_pytorch==0.7.7 datasets==3.5.0
python f5_samples.py --repo models/F5-TTS --text-file passage.txt --reference-audio reference.wav --reference-text "Exact reference transcript." --speed 1.1 --steps 32 --cfg-strength 2 --n 3 --output-dir outputs/f5
```

The official EMA checkpoint, vocabulary, and centered Vocos weights download
from pinned revisions. Provide a clean **1–8 second** reference. `--duration-seconds`
allocates speech time before pauses/crossfades; frame quantization and minimum
text length can alter the final allocation, which the preview reports. Choose
either `--pause-ms` or `--crossfade-ms`. Add `--estimate-only` for a preview;
the generator and vocoder weights are not loaded on that path.

### EmotiVoice

```powershell
git clone https://github.com/netease-youdao/EmotiVoice.git models/EmotiVoice
git -C models/EmotiVoice checkout 59f0f36de4db12825f4705dd4e0780d79dd6bb01
python -m pip install -r models/EmotiVoice/requirements.txt
python -m pip install scipy huggingface_hub
git lfs install
git clone https://www.modelscope.cn/syq163/outputs.git models/EmotiVoice/outputs
git -C models/EmotiVoice/outputs checkout d1332ad9fee7020df5f358fdbe8d257dbf0d1a7c
git -C models/EmotiVoice/outputs lfs pull
hf download WangZeJun/simbert-base-chinese --revision 8096d1f9a7f7c2d9dbd7974511c4bc4a1755932b --local-dir models/EmotiVoice/simbert
python -m nltk.downloader averaged_perceptron_tagger_eng averaged_perceptron_tagger cmudict
python emotivoice_samples.py --repo models/EmotiVoice --generator-checkpoint models/EmotiVoice/outputs/prompt_tts_open_source_joint/ckpt/g_00140000 --style-checkpoint models/EmotiVoice/outputs/style_encoder/ckpt/checkpoint_163431 --bert-path models/EmotiVoice/simbert --local-files-only --speaker 9017 --prompt Neutral --speed 1.0 --text-file passage.txt --n 3 --output-dir outputs/emotivoice
```

Speaker labels come from the checkout's `data/youdao/text/speaker2`.
`--speaker-index` instead selects the embedding index explicitly. Pitch and
energy scales act in predictor space; they are not semitone or decibel values.
`--emphasis-json` edits duration, pitch, and energy for specified **phoneme indices**:

```json
[{"chunk": 0, "phoneme": 3, "duration_scale": 1.2, "pitch_scale": 1.1, "energy_scale": 1.15}]
```

Preview manifests show phonemes so indices can be chosen deliberately. Changes
to durations are rounded before synthesis and included in the preview. Add
`--estimate-only` to skip the acoustic decoder and vocoder.

### StyleTTS2 LibriTTS

```powershell
git clone https://github.com/yl4579/StyleTTS2.git models/StyleTTS2
git -C models/StyleTTS2 checkout 5cedc71c333f8d8b8551ca59378bdcc7af4c9529
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install scipy numpy soundfile tqdm huggingface_hub pyyaml munch transformers==4.44.2 einops einops-exts librosa phonemizer nltk
python styletts2_samples.py --repo models/StyleTTS2 --text-file passage.txt --reference-audio reference.wav --alpha 0.3 --beta 0.7 --diffusion-steps 5 --speed 1.1 --n 3 --output-dir outputs/styletts2
```

Install **espeak-ng** and make its library discoverable by phonemizer; on Windows,
`PHONEMIZER_ESPEAK_LIBRARY` can point to its DLL. The adapter downloads the
pinned official LibriTTS checkpoint/config. Its complete inference BERT weights
are in the final checkpoint; separate ASR/JDC/PLBERT pretrained checkpoints and
the training-only monotonic alignment extension are unnecessary in this adapter.
The official LibriTTS decoder is **HiFiGAN**.
The fixed per-chunk 50-sample tail correction is counted; set `--tail-trim-samples 0`
to retain the full decoder waveform. No energy-based silence removal is applied.

`--estimate-only --output-dir previews/styletts2` writes a tensor plan for every
take. Resume it into a **fresh** directory with identical synthesis settings:

```powershell
python styletts2_samples.py --repo models/StyleTTS2 --text-file passage.txt --reference-audio reference.wav --alpha 0.3 --beta 0.7 --diffusion-steps 5 --speed 1.1 --n 3 --reuse-plan previews/styletts2/duration_plan.pt --output-dir outputs/styletts2
```

### OmniVoice

```powershell
python -m pip install torch==2.8.0 torchaudio==2.8.0
python -m pip install "git+https://github.com/k2-fsa/OmniVoice.git@08be0b4ccbac3e13e374e86fbfead4b4cac343e2" transformers==5.3.0 scipy tqdm
python omnivoice_samples.py --text-file passage.txt --instruct "male, middle-aged, low pitch, british accent" --speed 1.0 --num-step 64 --n 3 --output-dir outputs/omni
python omnivoice_samples.py --text-file passage.txt --instruct "male, middle-aged, low pitch, british accent" --speed 1.0 --estimate-only --output-dir previews/omni
```

Voice cloning uses `--ref-audio reference.wav --ref-text "Exact transcript."`.
`--prompt-cache voice.pt` stores the prepared clone prompt; later runs can use
that file alone. A new reference requires preprocessing. `--duration` sets speech
token time before explicit pauses, rounded to the model's 40 ms token grid.
The default disables silence removal, built-in long-form chunking, and optional
text normalization. `--normalize-text` needs upstream normalization dependencies.

### Supertonic 3

```powershell
python -m pip install numpy onnxruntime==1.23.1 soundfile scipy huggingface_hub
git clone https://github.com/supertone-oss-archive/supertonic.git models/supertonic
git -C models/supertonic checkout 1e9799e964ea4c0dad7cde993b65c3c813a7b373
hf download supertone-oss-archive/supertonic-3 --revision aafc6e32416a594460b32413efc49d7fe4ce6d46 --local-dir models/supertonic_assets
python supertonic_samples.py --repo models/supertonic --assets-dir models/supertonic_assets --voice M2 --lang en --speed 1.0 --steps 16 --text-file passage.txt --n 3 --sample-rate 48000 --pause-ms 0 --output-dir outputs/supertonic
```

Add `--estimate-only` for duration-only inference. `--voice-style` accepts a
compatible local style JSON. ONNX weights are exposed and hash-checked. Decoder
shape probes run during cold initialization; the text preview itself calls only
the duration predictor. The official removal of unused final latent padding is
predetermined by that duration, independent of waveform content.
For CUDA, use a separate `onnxruntime-gpu[cuda,cudnn]==1.23.2` environment and
`--device cuda:0 --allow-unverified-cuda-budget`. The four ONNX sessions divide
the arena budget; external GPU memory monitoring is still required. The Kaggle
notebook sets this up automatically.

## JSON configuration and output

JSON keys are the flag destination names, with underscores. Explicit CLI values
override those defaults. Unknown keys and invalid types/ranges are rejected.
Use model-specific keys from `--help`; not every control applies to every model.

```json
{
  "text_file": "passage.txt",
  "n": 3,
  "seed": 42,
  "device": "cpu",
  "speed": 1.1,
  "sample_rate": 44100,
  "pause_ms": 100,
  "output_dir": "outputs/my_run"
}
```

Run `python <model>_samples.py --config settings.json` with any required
model-specific repository/reference/checkpoint fields also supplied.
Preview/manifest JSON files contain settings, text/chunks, predicted length,
actual length/error for each exported take, planner/startup/generation timing,
source and weight provenance, and memory information. StyleTTS2 also saves the
cached tensor plan. Kokoro creates a unique run subdirectory automatically.

## Verification and primary sources

Focused tests check all 10–500 word counts for word preservation, exact integer
resampling math, configuration validation, frame allocation and joins. Model
runtime validation and its limitations are recorded in `validation_report.json`.
These checks establish output timing; they do not establish perceptual quality
or transcription accuracy for every text/configuration.

```powershell
python -m pip install pytest
python -m pytest -q test_kokoro_planning.py test_kokoro_frontend.py test_f5_style_planning.py test_emoti_super_planning.py test_omni_planning.py
```

- [Kokoro source](https://github.com/hexgrad/kokoro/tree/dfb907a02bba8152ca444717ca5d78747ccb4bec), [published inference weights](https://huggingface.co/hexgrad/Kokoro-82M)
- [F5 source](https://github.com/SWivid/F5-TTS/tree/283252563dbf91be625e0c27926acfaac449186c), [weights](https://huggingface.co/SWivid/F5-TTS), [Vocos](https://huggingface.co/charactr/vocos-mel-24khz)
- [EmotiVoice source](https://github.com/netease-youdao/EmotiVoice/tree/59f0f36de4db12825f4705dd4e0780d79dd6bb01), [checkpoint instructions](https://github.com/netease-youdao/EmotiVoice/wiki/Pretrained-models)
- [StyleTTS2 source](https://github.com/yl4579/StyleTTS2/tree/5cedc71c333f8d8b8551ca59378bdcc7af4c9529), [LibriTTS weights](https://huggingface.co/yl4579/StyleTTS2-LibriTTS)
- [OmniVoice source](https://github.com/k2-fsa/OmniVoice/tree/08be0b4ccbac3e13e374e86fbfead4b4cac343e2), [weights](https://huggingface.co/k2-fsa/OmniVoice)
- [Supertonic source](https://github.com/supertone-oss-archive/supertonic/tree/1e9799e964ea4c0dad7cde993b65c3c813a7b373), [complete ONNX weights](https://huggingface.co/supertone-oss-archive/supertonic-3)

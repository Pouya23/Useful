"""Build the self-contained Kaggle notebook from reviewed authoring cells."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

import kaggle_benchmark_cells as cells
import kaggle_setup_cells as setup


def markdown(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.strip() + "\n"}


def code(source: str, cell_id: str) -> dict:
    compile(source, f"<notebook:{cell_id}>", "exec")
    return {
        "cell_type": "code", "id": cell_id, "metadata": {}, "execution_count": None,
        "outputs": [], "source": source.strip() + "\n",
    }


def build(output_name: str = "kaggle_dual_t4_tts_benchmark.ipynb") -> Path:
    root = Path(__file__).resolve().parent
    model_filenames = ["kokoro_samples.py", "f5_samples.py", "emotivoice_samples.py", "styletts2_samples.py",
                       "omnivoice_samples.py", "supertonic_samples.py"]
    filenames = model_filenames + ["audio_quality_check.py"]
    embedded = {}
    for filename in filenames:
        contents = (root / filename).read_bytes()
        embedded[filename] = {"sha256": hashlib.sha256(contents).hexdigest(),
                              "gzip_base64": base64.b64encode(gzip.compress(contents, mtime=0)).decode()}
    signature = hashlib.sha256(json.dumps(embedded, sort_keys=True).encode()).hexdigest()[:12]
    extraction = (
        "# Six complete standalone adapters and the CPU quality checker. No Python file uploads are needed.\n"
        "import base64\nimport gzip\n\n"
        f"EMBEDDED_SOURCES = {json.dumps(embedded, indent=2)}\n"
        f"SCRIPTS_DIR = ROOT / 'scripts' / '{signature}'\n"
        "SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "for filename, record in EMBEDDED_SOURCES.items():\n"
        "    data = gzip.decompress(base64.b64decode(record['gzip_base64']))\n"
        "    if hashlib.sha256(data).hexdigest() != record['sha256']:\n"
        "        raise RuntimeError('Embedded source checksum mismatch: ' + filename)\n"
        "    compile(data.decode('utf-8'), filename, 'exec')\n"
        "    destination = SCRIPTS_DIR / filename\n"
        "    destination.write_bytes(data)\n"
        "atomic_json(SCRIPTS_DIR / 'source_manifest.json', {name: {'sha256': record['sha256']} for name, record in EMBEDDED_SOURCES.items()})\n"
        "print('Extracted verified adapters:', SCRIPTS_DIR)\n"
    )
    notebook_cells = [
        markdown("""# TTS duration and GPU memory benchmark — Kaggle T4 ×2

This single notebook contains all six adapters: **Kokoro, F5-TTS, EmotiVoice,
StyleTTS2, OmniVoice, and Supertonic 3**. It installs each model in a separate
environment, tests configurable profiles and 10–500-word passages, and creates a
downloadable ZIP with named WAVs, timing/error reports, memory traces, source
provenance, and full generation logs.

This revision fixes the F5 eager-import dependency failure observed in the
uploaded Kaggle run and replaces arbitrary chunk breaks with punctuation-aware
splitting. It removes added per-chunk gaps, protects PCM exports from clipping,
and uses male defaults. **Exact duration alone does not establish word completeness.**
Independent CPU ASR and silence/clipping diagnostics are included after synthesis.

**Before running:** import this notebook into Kaggle, enable **Internet**, select
**GPU T4 x2**, and edit the experiment cell below. Use a fresh kernel. First setup
downloads several GB; runtime environments/weights stay under `/tmp`, and reports
and audio stay under `/kaggle/working`. Nothing is published externally.

Up to two workers run concurrently, each restricted to one physical GPU UUID.
Different models are preferred whenever possible. Generation progress and current
GPU memory appear live. The default target is **5,000,000,000 bytes per GPU**, with
allocator limits and a 10 ms external NVML monitor that terminates an observed
overrun. **Sampling cannot prove an instantaneous peak ceiling.** Both sampled
process/device usage and native allocator peaks are retained separately.

Duration means the **complete exported WAV**, including silence, configured
pauses, and resampling. Frozen duration/frame/token plans are reused in synthesis;
WAV headers are independently checked afterward. F5 and OmniVoice preview a
chosen allocation, rather than predicting an unconstrained natural reading.
StyleTTS2's exact preview needs style diffusion and may be slower. Quality and
intelligibility must be judged by listening, even when duration matches exactly.

This notebook's GPU execution must be validated by your Kaggle run: it has not
been executed on two T4s locally. Failures are reported rather than silently
removing a model from the comparison.
""", "overview"),
        markdown("""## 1. Experiment configuration

Defaults: all six models, **10/100/500 words**, **two takes per job**, one enabled
profile per model. Enable alternate profiles or add any native option shown in
the saved help logs. Profile-level `n` and `seed` override the global values;
put other controls in `parameters`. GPU selection, text, output paths and the
VRAM target are controlled by the supervisor.

The built-in cases are three distinct, complete English passages: a ten-word
sentence, a one-hundred-word historical vignette, and a five-hundred-word story
with five paragraphs. No sentence is looped to reach a target word count. Edit
`TEXT_CASES` for your own passages or other lengths within 10–500 words.

For reference-based models, supply a clean **mono 1–8 second** reference and
its exact transcript (3–8 seconds is recommended), or keep both fields `None`
to create a disclosed male Kokoro reference on CPU. Synthetic reference quality
can affect cloning quality.

`speaker_mode="auto"` uses your supplied WAV for F5, StyleTTS2 and OmniVoice.
Otherwise defaults are Kokoro **am_fenrir**, EmotiVoice **9017/Neutral**,
Supertonic **M2** (documented deep/serious), and OmniVoice male/low-pitch design;
F5/Style use the male reference. `speaker_mode="reference"` tests only the three
cloning-capable models and reports the others as unsupported. `speaker_mode="default"`
uses male defaults. Style needs only the WAV; F5/Omni also require its transcript.
For Omni clone-specific instructions, use a profile's `clone_parameters` dictionary;
the default male design instruction is removed when your reference is selected.
""", "configuration-notes"),
        code(cells.CONFIG_CODE, "configuration"),
        markdown("## 2. Notebook helpers and hardware verification", "hardware-notes"),
        code(cells.KERNEL_CODE, "kernel-helpers"),
        code(cells.HARDWARE_CODE, "hardware"),
        markdown("""## 3. Extract the embedded standalone adapters

The compressed payload contains the six complete Python adapters and the CPU
quality checker. They are
verified by SHA-256, written locally for inspection, and included in the ZIP.
No external source archive upload is required.
""", "source-notes"),
        code(extraction, "embedded-sources"),
        markdown("""## 4. Isolated environments, pinned sources and pretrained weights

Setup is sequential and uses **no GPU allocations**. Each model has its own
Python 3.11 environment and package versions. The notebook kernel's Torch is not
replaced. Shared UV caches/hardlinks reduce duplicate runtime storage.
All source commits, weight revisions and package versions are logged. Model
failures appear in the final report; other models can continue. Runtime CUDA
compatibility is confirmed only by the later monitored inference jobs.

Progress is printed per model; detailed install output is retained in setup logs.
""", "setup-notes"),
        code(setup.SETUP_CODE, "model-setup"),
        code(setup.PREFLIGHT_CODE, "api-preflight"),
        markdown("""## 5. Reference voice

The CPU bootstrap is a prerequisite for reference-based models; paired GPU
experiments begin after it finishes. No content-dependent trimming is used.
The reference and transcript are included in the ZIP when configured.
""", "reference-notes"),
        code(setup.BOOTSTRAP_REFERENCE_CODE, "reference"),
        markdown("""## 6. Paired GPU experiments

Each job loads its model once and produces all requested takes. Both slots
display the worker's actual progress line, plus sampled VRAM. Jobs are named by
model, profile, word count, text/config hashes and seed. Full carriage-return
progress logs are saved. Failed/over-budget jobs and any partial recordings are
retained. If you interrupt, run the report cell afterward to package the results.
""", "benchmark-notes"),
        code(cells.SUPERVISOR_CODE, "paired-benchmark"),
        markdown("""## 7. Reports, listening players and ZIP download

`jobs.csv` includes every scheduled job, including setup/runtime failures and
duration plans available before a failure. `samples.csv` independently compares
predicted and actual duration/rate/sample count for each created WAV. Planner
timing scopes differ by model and are labeled explicitly. VRAM CSV traces and
native manifests remain available per job.

`quality.csv`/`quality.json` include acoustic gaps/clipping statistics, independent
Whisper transcripts, word edits, and ASR word error rate. Checks run on **CPU**
after generation and do not enter model VRAM or estimator latency measurements.
ASR errors are screening evidence: pronunciation, punctuation and recognition
errors can contribute. Listen to flagged samples before deciding a word is missing.
Disable ASR with `asr_quality_check=False` to retain acoustic checks without its
additional download/runtime, or disable both with `quality_checks=False`.

Download the ZIP from the link or Kaggle's **Output** panel. After extracting,
open `index.html` to compare the report and listen. Dependencies and pretrained
weights are excluded from the ZIP. A full kernel/VM shutdown can interrupt
packaging; completed files remain under `/kaggle/working/tts_benchmark/runs`
for as long as Kaggle retains that filesystem.
""", "report-notes"),
        code(cells.REPORT_CODE, "reports-zip-audio"),
        markdown("""## Primary references

- [Kaggle notebook settings](https://www.kaggle.com/docs/notebooks)
- [Kaggle accelerator identifiers](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [ONNX Runtime CUDA requirements and arena limits](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [PyTorch CUDA wheel versions](https://pytorch.org/get-started/previous-versions/)
- [Kokoro](https://github.com/hexgrad/kokoro/tree/dfb907a02bba8152ca444717ca5d78747ccb4bec)
- [F5-TTS](https://github.com/SWivid/F5-TTS/tree/283252563dbf91be625e0c27926acfaac449186c)
- [EmotiVoice](https://github.com/netease-youdao/EmotiVoice/tree/59f0f36de4db12825f4705dd4e0780d79dd6bb01)
- [StyleTTS2](https://github.com/yl4579/StyleTTS2/tree/5cedc71c333f8d8b8551ca59378bdcc7af4c9529)
- [OmniVoice](https://github.com/k2-fsa/OmniVoice/tree/08be0b4ccbac3e13e374e86fbfead4b4cac343e2)
- [Supertonic 3](https://github.com/supertone-oss-archive/supertonic/tree/1e9799e964ea4c0dad7cde993b65c3c813a7b373)
- [Supertonic speaker descriptions](https://github.com/supertone-oss-archive/supertonic-py/blob/main/docs/voices.md)
- [EmotiVoice speaker descriptions](https://github.com/netease-youdao/EmotiVoice/tree/main/data/youdao/text)
- [OmniVoice design attributes](https://github.com/k2-fsa/OmniVoice/blob/08be0b4ccbac3e13e374e86fbfead4b4cac343e2/docs/voice-design.md)
- [CPU ASR implementation](https://github.com/SYSTRAN/faster-whisper)

Use pretrained model licenses appropriate to your intended use; for example,
F5's published checkpoint is CC-BY-NC-4.0. This notebook is a benchmarking tool.
""", "references"),
    ]
    notebook = {"cells": notebook_cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "language": "python"},
        "tts_benchmark": {"embedded_source_signature": signature, "models": model_filenames,
                          "helpers": ["audio_quality_check.py"], "revision": "quality-and-speaker-controls",
                          "local_gpu_validation": False, "requires_manual_accelerator_selection": "GPU T4 x2"},
    }, "nbformat": 4, "nbformat_minor": 5}
    output = root / output_name
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Built {output}: {len(notebook_cells)} cells, {output.stat().st_size:,} bytes")
    return output


if __name__ == "__main__":
    build()

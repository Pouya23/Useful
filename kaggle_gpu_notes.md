# CUDA adapter audit for Kaggle T4 workers

Each worker process should be assigned one physical GPU UUID through
`CUDA_VISIBLE_DEVICES`, and receive `--device cuda:0` (the resulting visible
ordinal). Two workers can then run different models concurrently without either
model allocating across both GPUs.

Supertonic now accepts `cpu`, `cuda`, and `cuda:N`. CUDA requires
`--allow-unverified-cuda-budget`, as EmotiVoice already does. Its four independent
ONNX Runtime CUDA sessions split the aggregate arena budget. The sum of their
`gpu_mem_limit` values is at most the requested decimal GB budget minus
`--cuda-reserve-mb` (default 768 decimal MB). Each session uses
`kSameAsRequested`, `HEURISTIC`, and a limited convolution workspace. CUDA provider
initialization failures and automatic provider retries are rejected. The explicit
CPU provider remains available for unsupported operators and is disclosed.

These allocator settings do not guarantee a whole-process 5 GB peak. ONNX Runtime
documents that `gpu_mem_limit` covers only the execution provider arena. The
Supertonic manifest therefore leaves the CUDA total peak null, records arena
options, and requires the notebook's external NVML measurement. CPU still records
zero GPU allocation. NVML sampling measures an observed peak and can miss a
shorter spike; it cannot prove a strict continuous-time ceiling.

Use the isolated `onnxruntime-gpu[cuda,cudnn]==1.23.2` environment, and do not
install both CPU and GPU ONNX Runtime wheels. The current official compatibility
table lists this family as CUDA 12.8 / cuDNN 9. The adapter preloads NVIDIA
site-package runtime libraries with `onnxruntime.preload_dlls(directory="")`
before constructing sessions. The notebook should run an actual CUDA session
smoke check and preserve its diagnostics.

OmniVoice's `auto` dtype chooses float16 on CUDA, which matches its official GPU
example. A T4 has no native BF16 support; use `auto` or `float16`, rather than
making bfloat16 a default. F5's model uses float16 on CUDA, while its mel frontend
and Vocos decoder stay float32. The sampler explicitly casts generated mel to
float32 before Vocos. StyleTTS2, Kokoro, and EmotiVoice use float32. All models
continue to reuse the precise duration plan; device-dependent predictor rounding
does not invalidate the export length because synthesis consumes that cached
plan.

Validation on this machine:

- Ruff: both modified/new Python files passed.
- Nine new Supertonic GPU configuration tests and four existing EmotiVoice /
  Supertonic planning tests passed.
- Actual Supertonic CPU export after the changes: 10 words, three chunks, two
  flow steps, 24 kHz export, 75 ms chunk pauses. Predicted and exported lengths
  were both 130,368 samples (5.432 seconds), with 0% duration error. Warm preview
  took 7.47 ms and cold startup took 2.520 seconds. The manifest is in
  `.validation/supertonic_kaggle_cpu_regression/supertonic_manifest.json`.
- CUDA execution remains untested locally because this machine exposes no NVIDIA
  GPU. The Kaggle notebook performs the hardware-dependent tests.

Primary references:

- [ONNX Runtime CUDA provider requirements and options](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [Pinned F5 CFM sampler](https://github.com/SWivid/F5-TTS/blob/283252563dbf91be625e0c27926acfaac449186c/src/f5_tts/model/cfm.py)
- [Pinned F5 mel frontend](https://github.com/SWivid/F5-TTS/blob/283252563dbf91be625e0c27926acfaac449186c/src/f5_tts/model/modules.py)
- [Pinned OmniVoice GPU usage example](https://huggingface.co/k2-fsa/OmniVoice/blob/c5fdb5ccb189668d56333f77ba2629f4cd7535f4/README.md)

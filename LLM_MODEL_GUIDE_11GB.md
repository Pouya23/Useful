# Best Local LLM for the Safety-Montage Pipeline — ~11 GB VRAM

## Recommended model

For a GPU with roughly **11 GB of VRAM**, my quality-first recommendation for this narration pipeline is:

```text
mistralai/Ministral-3-14B-Instruct-2512-GGUF
Quantization: Q4_K_M
Runtime: llama.cpp
```

The official Q4_K_M GGUF is about **8.24 GB**.

That matters on an 11 GB GPU: it leaves roughly 2.5–3 GB of nominal VRAM for the KV cache, CUDA/runtime buffers, and temporary allocations.

This model is especially attractive here because the job is primarily:

- controlled prose generation,
- instruction following,
- rewriting to a target length,
- varying wording and structure,
- avoiding repetitive formulations,
- preserving factual warning content.

The model is an instruction-tuned 14B-class model designed for edge/local deployment.

## Why I prefer it over Qwen3-14B on 11 GB

Qwen3-14B remains a very good model, but its official Q4_K_M GGUF is about:

```text
9 GB
```

Ministral 3 14B Instruct Q4_K_M is about:

```text
8.24 GB
```

On a 24 GB GPU, that difference is not important.

On an 11 GB GPU, it is important.

Every extra ~0.5–1 GB can determine whether a prompt fits comfortably or triggers an out-of-memory error.

For that reason, I would use:

```text
Ministral 3 14B Instruct Q4_K_M
```

as the quality-first model.

---

# Important: use llama.cpp, not the current BitsAndBytes loader

For an 11 GB GPU, I would change the architecture slightly.

Instead of loading the LLM directly in the narration script with:

```python
AutoModelForCausalLM.from_pretrained(...)
```

run the LLM through a local `llama.cpp` server.

This gives you much more direct control over:

- GPU layer placement,
- context size,
- KV cache,
- quantized GGUF weights,
- CPU fallback,
- VRAM consumption.

It is also easy to unload the LLM server before starting Chatterbox.

---

# Install llama.cpp

One supported method is:

```bash
curl -LsSf https://llama.app/install.sh | sh
```

Then start the model with:

```bash
llama-server \
  -hf mistralai/Ministral-3-14B-Instruct-2512-GGUF:Q4_K_M \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 999
```

If full GPU offload does not fit, reduce GPU layers or let some layers remain in system RAM.

For this narration task I would begin with:

```text
context = 4096
```

and increase to:

```text
6144
```

or:

```text
8192
```

only if your inputs actually require it.

Do not allocate a huge context window simply because the model supports one.

---

# Why 4K context is usually enough

The LLM does not need the entire history database in its prompt.

The Python program should provide only:

- current warnings,
- current context,
- target duration / word count,
- a small number of previous phrases to avoid,
- structural instructions.

The full cross-generation history belongs in SQLite and should be searched/scored externally.

That keeps the LLM prompt small and saves VRAM.

---

# Suggested sampling for narration

For this type of prose generation, start around:

```python
temperature = 0.70
top_p = 0.90
top_k = 40
repeat_penalty = 1.06
```

For three candidates, vary temperature only modestly:

```python
candidate_temperatures = [
    0.66,
    0.72,
    0.78,
]
```

The Python novelty scorer should be responsible for rejecting repetitive candidates.

Do not use very high temperature as the main mechanism for obtaining novelty.

---

# Local server API

`llama-server` exposes an OpenAI-compatible API.

A simple Python client can look like:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

def generate(system_prompt, user_prompt, temperature=0.72, max_tokens=1200):
    response = client.chat.completions.create(
        model="local",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=temperature,
        top_p=0.90,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content
```

I would modify `safety_montage_narrator.py` so its LLM backend can use this API instead of loading Transformers directly.

---

# Memory strategy

The pipeline should operate sequentially:

```text
Start llama.cpp
      ↓
Generate narration candidates
      ↓
Select / revise final text
      ↓
Shut down llama.cpp
      ↓
VRAM is released
      ↓
Load Chatterbox
      ↓
Generate speech
      ↓
Unload Chatterbox
```

Do **not** try to keep Ministral and Chatterbox resident on an 11 GB card at the same time.

---

# If 14B still runs out of memory

There are three useful fallbacks.

## Option 1 — reduce context first

Try:

```text
--ctx-size 3072
```

or:

```text
--ctx-size 2048
```

before abandoning the 14B model.

For this application, 2K–4K tokens are often enough if the prompt is designed efficiently.

## Option 2 — partially offload to CPU

Keep most layers on the GPU and let llama.cpp place a small number in system RAM.

This will be slower, but narration generation is offline and you only need a few generations per montage.

For a quality-oriented pipeline this is often preferable to dropping immediately from 14B to 8B.

## Option 3 — use Ministral 3 8B Instruct

Fallback model:

```text
mistralai/Ministral-3-8B-Instruct-2512-GGUF
```

This leaves substantially more VRAM and should run comfortably.

Use it when:

- generation speed matters,
- you have very little system RAM,
- the 14B model cannot remain stable,
- you need a larger context.

---

# Alternative: Qwen3-8B

If you want to keep the existing Hugging Face / Transformers architecture with minimal changes, use:

```text
Qwen/Qwen3-8B
```

with 4-bit NF4.

This is the easiest option with the existing code because it works directly with:

```python
AutoTokenizer
AutoModelForCausalLM
BitsAndBytesConfig
```

and Qwen3 supports:

```python
enable_thinking=False
```

For non-thinking mode, Qwen recommends approximately:

```text
temperature = 0.7
top_p = 0.8
top_k = 20
```

Qwen3-8B is therefore my **compatibility-first** recommendation.

Ministral 3 14B Q4_K_M is my **quality-first** recommendation.

---

# Alternative: Qwen3-14B with partial CPU offload

You can also run:

```text
Qwen/Qwen3-14B-GGUF
Q4_K_M
```

The official Q4_K_M file is approximately:

```text
9 GB
```

It may fit on an 11 GB GPU with a modest context, but it leaves less headroom than Ministral 3 14B.

If using Qwen3-14B on this hardware, I would prefer llama.cpp rather than BitsAndBytes.

---

# What I would actually use

Given:

```text
GPU VRAM: ~11 GB
Workload: offline narration generation
TTS: Chatterbox
Models run sequentially
Need: maximum text quality within hardware
```

I would use:

```text
LLM
  Ministral 3 14B Instruct 2512
  Q4_K_M GGUF
  llama.cpp

Context
  4096 tokens initially

Candidate generation
  3 candidates

Temperature
  0.66
  0.72
  0.78

Top-p
  0.90

Repeat penalty
  ~1.06

Novelty
  enforced by Python + SQLite history

Duration
  enforced by your external duration estimator

TTS
  Chatterbox, loaded only after the LLM is unloaded
```

---

# Priority order for an 11 GB GPU

For this particular project:

```text
1. Ministral 3 14B Instruct Q4_K_M + llama.cpp
   Best quality-oriented fit.

2. Qwen3-14B Q4_K_M + llama.cpp / partial CPU offload
   Very strong alternative, but slightly tighter VRAM fit.

3. Qwen3-8B 4-bit NF4
   Best option if you want to preserve the current Transformers code.

4. Ministral 3 8B Instruct Q4_K_M
   Very comfortable memory footprint and good fallback.
```

I would not attempt a 30B model on an 11 GB GPU for this pipeline. Heavy CPU offloading would make it much slower and is unlikely to be a good trade compared with a modern 14B model.

---

# Sources

Official Mistral model:

https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512

Official Mistral GGUF:

https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-GGUF

Official Qwen3-14B:

https://huggingface.co/Qwen/Qwen3-14B

Official Qwen3-14B GGUF:

https://huggingface.co/Qwen/Qwen3-14B-GGUF

Official Qwen3-8B:

https://huggingface.co/Qwen/Qwen3-8B

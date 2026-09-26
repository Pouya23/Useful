# Chatterbox TTS Duration Estimator

This package estimates the duration of the **original English Chatterbox TTS**
voice from text, `exaggeration`, and `cfg_weight`.

It intentionally has two prediction paths:

- **Fast ML estimator** — appropriate for hundreds of duration evaluations while
  searching narration text and TTS controls.
- **T3 token probe** — appropriate for the final candidate when timing must be
  extremely accurate.

## Why the target is speech-token count

Chatterbox's S3 speech representation runs at **25 tokens per second**, i.e.
one token represents about **40 ms**. The estimator therefore learns the number
of T3 speech tokens rather than learning seconds directly.

This is usually easier to model and avoids noise introduced by measuring
waveform files.

A pure ML model still cannot guarantee <1% on *every* input. For example, on a
1-second clip, 1% is 10 ms, smaller than a single 40 ms speech-token frame.

For the final result, the `probe` command runs T3 and obtains the generated
speech-token count directly. If you reuse those exact speech tokens for S3Gen
decoding, duration is determined before waveform synthesis.

## Recommended training data

Accuracy depends far more on data matching than on model complexity.

Best:

1. Collect thousands of 280-character-or-shorter chunks from the **same LLM**
   that writes your safety-warning narration.
2. Include the same punctuation conventions and warning vocabulary used in
   production.
3. Sample the exact `exaggeration` and `cfg_weight` range your pipeline searches.
4. Keep Chatterbox temperature, top-p, min-p and repetition penalty the same
   between calibration and production.

A good starting experiment is:

- 1,500–2,500 unique chunks
- 5 parameter variants per chunk
- 7,500–12,500 T3 labels

If held-out error is still above your requirement, add the real production
chunks where error was highest and retrain.

## Install

Install Chatterbox in the environment required by its project, then:

```bash
pip install scipy scikit-learn joblib cmudict lightgbm
```

## 1. Collect labels

```bash
python chatterbox_duration_estimator.py collect \
  --voice-ref voice.wav \
  --corpus narration_corpus.txt \
  --out duration_data.csv \
  --texts 2000 \
  --variants-per-text 5 \
  --device cuda
```

If there are not enough texts in the corpus, the collector supplements them
with synthetic safety-oriented text. Real narration chunks are preferable.

`--waveform-check-rate 0.02` (the default) decodes about 2% of examples through
S3Gen as a sanity check of the speech-token-duration mapping.

## 2. Train

```bash
python chatterbox_duration_estimator.py train \
  --data duration_data.csv \
  --model duration_model.joblib
```

The split is grouped by normalized text. Parameter variants of one sentence
cannot leak between train and test.

The report includes:

- MAE seconds
- RMSE seconds
- MAPE
- median APE
- p90 / p95 / p99 APE
- maximum APE
- fraction below 1%, 2%, and 5%

You can make training fail if the measured requirement is not reached:

```bash
python chatterbox_duration_estimator.py train \
  --data duration_data.csv \
  --model duration_model.joblib \
  --require-mape 1.0
```

For the much stronger condition "every held-out sample is below 1%":

```bash
--require-max-ape 1.0
```

Expect that condition to be difficult or impossible for very short clips.

## 3. Fast prediction

```bash
python chatterbox_duration_estimator.py predict \
  --model duration_model.joblib \
  --text "Keep hands clear of moving parts while the equipment is operating." \
  --exaggeration 0.50 \
  --cfg-weight 0.50
```

Long narration is split into chunks and the configured inter-chunk pause is
added exactly like the montage pipeline.

## 4. Final T3 probe

```bash
python chatterbox_duration_estimator.py probe \
  --voice-ref voice.wav \
  --text "Keep hands clear of moving parts while the equipment is operating." \
  --exaggeration 0.50 \
  --cfg-weight 0.50 \
  --seed 1234 \
  --verify-waveform
```

The `--verify-waveform` option actually decodes the probed token sequence and
shows the measured waveform duration.

To retain the final generated T3 tokens:

```bash
--save-tokens selected_tokens.pt
```

For the strongest determinism, synthesize the final waveform from these saved
tokens rather than asking T3 to sample them again.



## 5. Decode the exact probed tokens

After probing with:

```bash
python chatterbox_duration_estimator.py probe \
  --voice-ref voice.wav \
  --text-file narration.txt \
  --exaggeration 0.50 \
  --cfg-weight 0.50 \
  --seed 1234 \
  --save-tokens selected_tokens.pt
```

decode **those exact tokens** rather than running T3 a second time:

```bash
python chatterbox_duration_estimator.py synthesize-tokens \
  --voice-ref voice.wav \
  --tokens selected_tokens.pt \
  --out narration.wav
```

The command prints both the duration predicted from speech-token count and the
actual waveform duration. This removes stochastic T3 resampling as a source of
timing error.


## 6. Search exaggeration / CFG quickly

```bash
python chatterbox_duration_estimator.py search \
  --model duration_model.joblib \
  --text-file narration.txt \
  --target 58.4 \
  --step 0.02
```

Use the ML search to find the top few candidates, then use `probe` on those few
candidates.

## Montage-pipeline adapter

The Python file exports:

```python
estimate_duration(text, exaggeration, cfg_weight) -> float
```

Set the trained model path:

```bash
export CHATTERBOX_DURATION_MODEL=/path/to/duration_model.joblib
```

Then use:

```bash
--estimator chatterbox_duration_estimator:estimate_duration
```

If that environment variable is absent, the function falls back to a rough
rule-based syllable/punctuation estimate.

## Accuracy strategy

The most accurate practical architecture is:

```text
LLM narration candidates
        |
        v
fast learned duration model
        |
        +---- search many exaggeration / CFG combinations
        |
        v
top few candidates
        |
        v
T3 speech-token probe using production seed
        |
        v
exact token count / 25 Hz
        |
        v
reuse selected T3 tokens in S3Gen
        |
        v
waveform
```

This is materially stronger than trying to train a regression model and
pretending it can guarantee an arbitrary relative-error bound on unseen text.

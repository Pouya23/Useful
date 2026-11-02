#!/usr/bin/env python3
"""Optimize one rigging question with DSPy GEPA; no few-shot examples.

Install: pip install 'dspy==3.4.1' 'openai==2.54.0'
Required env: VLM_BASE_URL, VLM_API_KEY, VLM_MODEL.
Optional env: EDITOR_BASE_URL, EDITOR_API_KEY, EDITOR_MODEL (default to VLM).
API assumption: OpenAI-compatible Chat Completions accepting video_url blocks.
If your provider differs, change only ask_vlm() below.

Uses the previously supplied rigging_vqa_9_call_config.json.
Each train/val/test JSONL line:
{"video":"clips/001.mp4","group":"lift_001","answers":{"4.4":"yes"}}
Video paths are relative to the JSONL file; HTTP video URLs also work.
All clips from one lift/recording must share a group and stay in one split.
Use labels from the selected question; not_applicable rows are skipped.

Run: python optimize_rigging.py --question 4.4 --train train.jsonl --val val.jsonl
Optional: --test test.jsonl --budget 60 --out optimized_prompt.json
Use --costs error_costs.json for asymmetric errors; otherwise this is an accuracy baseline.
Costs are indexed as costs[question_id][true_label][predicted_label].
Every row must cover all allowed labels and __invalid__; correct-label costs are zero.
Choose costs for your actual review/action policy, including what unclear means operationally.
Budget counts GEPA evaluations, not dollars; reporting/test calls are additional.
Repeated identical VLM requests are cached in memory. VLM retries are disabled.
"""

import argparse
import base64
import json
import math
import mimetypes
import os
from functools import lru_cache
from pathlib import Path

import dspy
from dspy.teleprompt.gepa import InstructionProposer
from openai import OpenAI


@lru_cache(maxsize=8)
def video_url(video):
    if video.startswith(("http://", "https://", "data:")):
        return video
    mime = mimetypes.guess_type(video)[0] or "video/mp4"
    encoded = base64.b64encode(Path(video).read_bytes()).decode()
    return f"data:{mime};base64,{encoded}"


def load_examples(filename, question, labels):
    path = Path(filename)
    examples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        answer = row["answers"][question]
        if answer == "not_applicable":
            continue
        if answer not in labels:
            raise ValueError(f"Invalid gold label in {filename}: {answer!r}")
        video = row["video"]
        if not video.startswith(("http://", "https://", "data:")):
            video = str((path.parent / video).resolve(strict=True))
        examples.append(dspy.Example(
            video=video, answer=answer, group=row["group"]
        ).with_inputs("video"))
    if not examples:
        raise ValueError(f"No applicable examples in {filename}")
    return examples


def load_costs(filename, question, labels):
    columns = [*labels, "__invalid__"]
    costs = (json.loads(Path(filename).read_text())[question] if filename else
             {y: {p: float(y != p) for p in columns} for y in labels})
    if set(costs) != set(labels) or any(set(row) != set(columns) for row in costs.values()):
        raise ValueError("Costs must cover every true label and every prediction, including __invalid__.")
    values = [v for row in costs.values() for v in row.values()]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Costs must be finite nonnegative numbers.")
    if any(costs[y][y] != 0 for y in labels) or max(values) == 0:
        raise ValueError("Correct answers must cost zero; at least one error must have positive cost.")
    return costs


def make_metric(costs):
    maximum = max(v for row in costs.values() for v in row.values())

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        row = costs[gold.answer]
        cost = row.get(pred.answer, row["__invalid__"])
        return dspy.Prediction(
            score=1 - cost / maximum,
            feedback=(f"Expected {gold.answer!r}; received {pred.answer!r}. "
                      f"Error cost: {cost:g}; 0 is best and {maximum:g} is worst."),
        )

    return metric


def evaluate(program, examples, costs):
    labels, columns = list(costs), [*costs, "__invalid__"]
    confusion = {y: dict.fromkeys(columns, 0) for y in labels}
    for example in examples:
        answer = program(video=example.video).answer
        confusion[example.answer][answer if answer in costs else "__invalid__"] += 1
    n = len(examples)
    per_class = {}
    for label in labels:
        support = sum(confusion[label].values())
        tp = confusion[label][label]
        fp = sum(confusion[y][label] for y in labels if y != label)
        per_class[label] = {
            "support": support, "false_negatives": support - tp, "false_positives": fp,
            "recall": tp / support if support else None,
            "precision": tp / (tp + fp) if tp + fp else None,
        }
    recalls = [v["recall"] for v in per_class.values() if v["support"]]
    mean_cost = sum(confusion[y][p] * costs[y][p] for y in labels for p in columns) / n
    maximum = max(v for row in costs.values() for v in row.values())
    return {
        "n": n, "accuracy": sum(confusion[y][y] for y in labels) / n,
        "mean_error_cost": mean_cost, "mean_optimization_score": 1 - mean_cost / maximum,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "unclear_rate": sum(confusion[y].get("unclear", 0) for y in labels) / n,
        "invalid_rate": sum(confusion[y]["__invalid__"] for y in labels) / n,
        "per_class": per_class, "confusion_true_by_predicted": confusion,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="4.4")
    parser.add_argument("--config", default="rigging_vqa_9_call_config.json")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test")
    parser.add_argument("--costs", help="Question-specific error-cost matrices; omitted = uniform errors.")
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--out", default="optimized_prompt.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    question = config["questions"][args.question]
    labels = question["model_selectable_answers"]
    costs = load_costs(args.costs, args.question, labels)
    metric = make_metric(costs)
    objective = "expected_error_cost" if args.costs else "accuracy_baseline_uniform_costs"
    seed = "\n\n".join([
        config["shared_instruction"], question["question"], question["selection_rule"],
        json.dumps(question.get("definitions", {})),
    ])
    suffix = "\n\nReturn exactly one of: " + " / ".join(labels) + ". No explanation."
    train = load_examples(args.train, args.question, labels)
    val = load_examples(args.val, args.question, labels)
    test = load_examples(args.test, args.question, labels) if args.test else []
    seen_groups, seen_videos = set(), set()
    for split in (train, val, test):
        groups, videos = {e.group for e in split}, {e.video for e in split}
        if seen_groups & groups or seen_videos & videos:
            raise ValueError("A lift/recording or video appears in multiple splits.")
        seen_groups.update(groups)
        seen_videos.update(videos)
    if args.budget <= len(val):
        raise ValueError("--budget must exceed the number of validation examples.")

    model, base, key = (os.environ[k] for k in ("VLM_MODEL", "VLM_BASE_URL", "VLM_API_KEY"))
    client = OpenAI(base_url=base, api_key=key, max_retries=0, timeout=120)
    dspy.configure(lm=dspy.LM(f"openai/{model}", api_base=base, api_key=key))
    editor = dspy.LM(
        "openai/" + os.getenv("EDITOR_MODEL", model),
        api_base=os.getenv("EDITOR_BASE_URL", base),
        api_key=os.getenv("EDITOR_API_KEY", key),
        temperature=1.0, max_tokens=2048, num_retries=0,
    )

    @lru_cache(maxsize=None)
    def ask_vlm(video, prompt):
        response = client.chat.completions.create(
            model=model, temperature=0, max_tokens=96,
            messages=[{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": video_url(video)}},
                {"type": "text", "text": prompt + suffix},
            ]}],
        )
        return (response.choices[0].message.content or "").strip()

    class VideoAdapter(dspy.Adapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            return [{"answer": ask_vlm(inputs["video"], signature.instructions)}]

    class VideoQuestion(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(dspy.Signature("video -> answer", instructions=seed))

        def forward(self, video):
            with dspy.context(adapter=VideoAdapter()):
                return self.predict(video=video)

    program = VideoQuestion()
    print(f"Objective: {objective}")
    before = evaluate(program, val, costs)  # Checks API/video compatibility before search.
    optimizer = dspy.GEPA(
        metric=metric, reflection_lm=editor, max_metric_calls=args.budget,
        num_threads=1, seed=0,
        instruction_proposer=InstructionProposer(
            additional_instructions=(
                "Optimize wording only. Preserve the question's meaning, temporal scope, "
                "thresholds, selection priorities, and every answer label and meaning. "
                "Do not add questions, examples, demonstrations, or new labels. "
                "Minimize error costs without changing the observed facts being assessed.\n"
                "Costs, indexed by true label then predicted label:\n" + json.dumps(costs) +
                "\nOriginal task specification:\n" + seed + suffix
            ),
            max_chars=3000,
        ),
    )
    optimized = optimizer.compile(program, trainset=train, valset=val)
    after = evaluate(optimized, val, costs)
    result = {
        "question_id": args.question, "model": model, "answers": labels,
        "prompt": optimized.predict.signature.instructions + suffix,
        "objective": objective, "evaluation_scope": "single_question",
        "error_costs": costs, "validation_before": before, "validation_after": after,
    }
    if test:
        result["test"] = evaluate(optimized, test, costs)
    result["vlm_api_calls"] = ask_vlm.cache_info().misses
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validation accuracy: {before['accuracy']:.3f} -> {after['accuracy']:.3f}; "
          f"mean error cost: {before['mean_error_cost']:.3g} -> {after['mean_error_cost']:.3g}; "
          f"VLM calls: {result['vlm_api_calls']}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

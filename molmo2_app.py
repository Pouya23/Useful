import os
import json
import tempfile
import threading
from bisect import bisect_left

import av
import torch
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
from flasgger import Swagger
from transformers import AutoProcessor, AutoModelForImageTextToText

# -------------------------
# Configuration
# -------------------------
MODEL_ID = f'allenai/Molmo2-{os.environ.get("MODEL_SIZE", "8B").upper()}'
UPLOAD_FOLDER = "uploads"
MAX_ICL_IMAGES = 32  # Preflight budget: each video reserves num_frames images.

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

swagger = Swagger(app)

# Prevent multiple requests from consuming GPU memory simultaneously.
inference_lock = threading.Lock()

# -------------------------
# Load Model Once at Startup
# -------------------------
print("Loading model...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype="auto",
    device_map="cuda",
    use_fast=False
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=(torch.float16 if os.environ["MODEL_DTYPE"].endswith("16") else torch.float32) if os.environ.get("MODEL_DTYPE", "").lower() in ("float32", "fp32", "float16", "fp16") else torch.bfloat16,
    device_map="cuda"
)

model.eval()
print("Model loaded successfully!")
print("CUDA available:", torch.cuda.is_available())
print("Model first param device:", next(model.parameters()).device)
print("Model first param dtype:", next(model.parameters()).dtype)
print("Device map:", getattr(model, "hf_device_map", "No device map"))


def load_media(upload, kind, video_path, num_frames):
    """Return [(RGB image, timestamp or None)]. Used for images and ICL videos.

    Decode videos twice to sample by actual presentation time without retaining
    every frame's pixels. num_frames is a maximum; 1 selects the last frame.
    Video pixels retain their encoded orientation. Audio is not included.
    """
    if kind == "image":
        with Image.open(upload.stream) as image:
            return [(ImageOps.exif_transpose(image).convert("RGB"), None)]

    upload.save(video_path)
    with av.open(video_path) as video:
        if not video.streams.video:
            raise ValueError("Uploaded video has no video stream")
        times = [frame.time for frame in video.decode(video=0)]
    if not times or any(t is None for t in times):
        raise ValueError("Video must contain frames with presentation timestamps")
    times = [float(t - times[0]) for t in times]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("Video timestamps must be chronological")

    count = min(num_frames, len(times))
    indices = {len(times) - 1}
    if count > 1:
        indices.add(0)
        for k in range(1, count - 1):
            target = times[-1] * k / (count - 1)
            index = min(bisect_left(times, target), len(times) - 1)
            if index > 0 and abs(times[index - 1] - target) < abs(times[index] - target):
                index -= 1
            indices.add(index)

    frames = []
    with av.open(video_path) as video:
        for index, frame in enumerate(video.decode(video=0)):
            if index in indices:
                frames.append((frame.to_image().convert("RGB"), times[index]))
            if index == max(indices):
                break
    if len(frames) != len(indices):
        raise ValueError("Could not decode the selected video frames consistently")
    return frames


# -------------------------
# API Endpoint
# -------------------------
@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate from a main image/video, with optional labeled examples
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: prompt
        in: formData
        type: string
        required: true
        description: Question for the main input; default question for examples
      - name: video
        in: formData
        type: file
        required: false
        description: Main video; provide exactly one of image or video
      - name: image
        in: formData
        type: file
        required: false
        description: Main image; provide exactly one of image or video
      - name: examples
        in: formData
        type: string
        required: false
        description: 'JSON array, e.g. [{"field":"example_0","type":"image","answer":"SUPPORTED"}]. Each field names another uploaded file. Optional prompt per example.'
      - name: example_0
        in: formData
        type: file
        required: false
        description: Optional reference file; describe it in examples
      - name: example_1
        in: formData
        type: file
        required: false
        description: Optional reference file; further named fields work via API clients
      - name: num_frames
        in: formData
        type: integer
        default: 6
        required: false
        description: Maximum frames per video in ICL requests (1 to 32); ordinary video requests use the existing native processor
    responses:
      200:
        description: Generated text in the response field
      400:
        description: Invalid request or context too long
    """
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    main_types = [kind for kind in ("image", "video") if kind in request.files]
    if len(main_types) != 1:
        return jsonify({"error": "Upload exactly one main image or video"}), 400
    main_type = main_types[0]
    main_file = request.files[main_type]
    if not main_file.filename or len(request.files.getlist(main_type)) != 1:
        return jsonify({"error": "Provide one nonempty main file field"}), 400

    try:
        examples = json.loads(request.form.get("examples", "[]"))
        num_frames = int(request.form.get("num_frames", "6"))
    except (ValueError, TypeError):
        return jsonify({"error": "examples must be JSON; num_frames must be an integer"}), 400
    if not isinstance(examples, list) or not 1 <= num_frames <= 32:
        return jsonify({"error": "examples must be an array; num_frames must be 1 to 32"}), 400

    used_fields = {"image", "video"}
    for i, example in enumerate(examples):
        if not isinstance(example, dict):
            return jsonify({"error": f"Example {i} must be an object"}), 400
        field = example.get("field")
        answer = example.get("answer")
        question = example.get("prompt", prompt)
        if (
            not isinstance(field, str) or field in used_fields
            or example.get("type") not in ("image", "video")
            or not isinstance(answer, str) or not answer.strip()
            or not isinstance(question, str) or not question.strip()
            or len(request.files.getlist(field)) != 1
            or not request.files[field].filename
        ):
            return jsonify({"error": f"Example {i} needs a unique uploaded field, image/video type, nonempty answer, and optional nonempty prompt"}), 400
        used_fields.add(field)

    if examples:
        media_types = [main_type] + [example["type"] for example in examples]
        image_budget = sum(num_frames if kind == "video" else 1 for kind in media_types)
        if image_budget > MAX_ICL_IMAGES:
            return jsonify({"error": f"ICL is limited to {MAX_ICL_IMAGES} image/frame slots per request. Reduce examples or num_frames."}), 400

    # Unique paths per request; automatically clean up every uploaded video.
    with tempfile.TemporaryDirectory(dir=app.config["UPLOAD_FOLDER"]) as request_dir:
        images = []
        try:
            if not examples:
                # Preserve the existing native video path. Images use the same API.
                if main_type == "video":
                    media = os.path.join(request_dir, "main.mp4")
                    main_file.save(media)
                else:
                    media = load_media(main_file, "image", None, num_frames)[0][0]
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": main_type, main_type: media},
                ]}]
            else:
                # Molmo's template numbers images globally across the request.
                def add_group(upload, kind, name):
                    path = os.path.join(request_dir, f"{name}.mp4")
                    mapping = []
                    for image, timestamp in load_media(upload, kind, path, num_frames):
                        images.append(image)
                        stamp = "still image" if timestamp is None else f"t={timestamp:.3f}s"
                        mapping.append(f"Image {len(images)} ({stamp})")
                    return f"{name}: " + "; ".join(mapping)

                sections = [
                    "Each REFERENCE group and MAIN are separate scenes. "
                    "Images within each video group are chronological sampled frames, "
                    "with timestamps in seconds from that video's first frame. "
                    "Reference answers are demonstrations. Answer MAIN only, "
                    "using MAIN's own visible evidence."
                ]
                for i, example in enumerate(examples, 1):
                    sections.append(
                        add_group(request.files[example["field"]], example["type"], f"REFERENCE_{i}")
                        + f"\nQuestion: {example.get('prompt', prompt)}"
                        + f"\nCorrect answer: {example['answer']}"
                    )
                sections.append(
                    add_group(main_file, main_type, "MAIN")
                    + f"\nQuestion: {prompt}\nAnswer MAIN only."
                )
                content = [{"type": "image", "image": image} for image in images]
                content.append({"type": "text", "text": "\n\n".join(sections)})
                messages = [{"role": "user", "content": content}]
        except (ValueError, OSError, av.FFmpegError) as exc:
            return jsonify({"error": f"Could not read uploaded media: {exc}"}), 400

        # Keep preprocessing, GPU transfer, inference AND cleanup under the lock.
        with inference_lock:
            inputs = generated_ids = generated_tokens = None
            try:
                try:
                    if examples:
                        text = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = processor(
                            text=text,
                            images=images,
                            images_kwargs={"max_crops": 2},
                            return_tensors="pt"
                        )
                    else:
                        inputs = processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                            return_dict=True
                        )
                except (ValueError, OSError, av.FFmpegError) as exc:
                    return jsonify({"error": f"Could not process uploaded media: {exc}"}), 400

                # Reject oversized requests; truncation could break visual mappings.
                text_config = getattr(model.config, "text_config", model.config)
                context_limit = getattr(text_config, "max_position_embeddings", None)
                if context_limit and inputs["input_ids"].size(1) + 256 > context_limit:
                    return jsonify({"error": "Input exceeds the model context. Reduce examples, num_frames, or prompt length."}), 400

                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, max_new_tokens=256)

                generated_tokens = generated_ids[0, inputs["input_ids"].size(1):].cpu()
                generated_text = processor.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                return jsonify({"response": generated_text})
            finally:
                del generated_tokens, generated_ids, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

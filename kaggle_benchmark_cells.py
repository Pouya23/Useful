"""Authoring sources embedded into the single delivered Kaggle notebook.

These strings become notebook cells; the notebook does not import this file.
"""

import json
from pathlib import Path

CONFIG_CODE = r'''
import datetime as dt
import uuid
from pathlib import Path

ROOT = Path("/kaggle/working/tts_benchmark")
RUN_ID = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
RUN_DIR = ROOT / "runs" / RUN_ID
PACKAGE_DIR = RUN_DIR / "download"
RUN_DIR.mkdir(parents=True, exist_ok=False)
PACKAGE_DIR.mkdir(parents=True, exist_ok=False)

# Change these before running the installation/benchmark cells.
CONFIG = {
    "install_models": ["kokoro", "f5", "emotivoice", "styletts2", "omnivoice", "supertonic"],
    "force_setup": False,
    "torch_cuda_variant": "auto",
    "strict_cuda_driver": False,
    "n": 2,
    "seed": 1234,
    "sample_rate": 48000,
    "pause_ms": 0.0,           # Native punctuation already contributes pauses; no extra gap per chunk.
    "max_vram_gb": 5.0,          # Decimal GB: 5,000,000,000 bytes.
    "nvml_interval_ms": 10.0,
    "enforce_observed_budget": True,
    "budget_scope": "device_total", # Conservative; includes pre-existing allocations.
    "job_timeout_seconds": 3600,
    "gpu_release_timeout_seconds": 20,
    "cpu_threads_per_job": 4,
    "require_two_t4": True,
    "speaker_mode": "auto",  # auto: user WAV if supplied, otherwise male defaults. reference: cloning models only. default: male defaults.
    "reference_audio": None,    # e.g. /kaggle/input/my-voice/reference.wav
    "reference_text": None,     # Exact transcript; use a clean 1-8 second reference.
    "include_reference_in_zip": True,
    "preview_players": 6,
    "quality_checks": True,
    "asr_quality_check": True, # CPU-only word comparison after synthesis; it adds runtime, not GPU memory.
    "asr_model": "base.en",    # Use small.en for a stronger English check, or a local CTranslate2 model path.
    "asr_language": "en",
    "quality_timeout_seconds": 7200,
}

# Three distinct passages, with no repeated sentences. For other lengths,
# supply your own TEXT_CASES rather than cutting or cycling a passage.
WORD_COUNTS = [10, 100, 500]
SAMPLE_PASSAGES = __SAMPLE_PASSAGES_JSON__
TEXT_CASES = [
    {"name": f"w{count:03d}", "text": SAMPLE_PASSAGES[str(count)]}
    for count in WORD_COUNTS
]
# Example replacement:
# TEXT_CASES = [{"name": "my_passage", "text": Path("/kaggle/input/my-text/passage.txt").read_text()}]

# Each profile is a separate experiment, repeated n times. Toggle enabled,
# change parameters, or add profiles. Names appear in filenames and reports.
# Parameters use each script's own CLI destination names (saved --help files explain them).
MODEL_PROFILES = {
    "kokoro": [
        {"name": "male_fenrir", "enabled": True, "parameters": {"voice": "am_fenrir", "speed": 1.0, "chunk_words": 45, "normalize_peak": True}},
        {"name": "male_george", "enabled": False, "parameters": {"voice": "bm_george", "language": "b", "speed": 1.0, "chunk_words": 45, "normalize_peak": True}},
    ],
    "f5": [
        {"name": "reference_quality", "enabled": True, "parameters": {"speed": 1.0, "steps": 64, "cfg_strength": 2.0, "sway": -1.0, "ode_method": "euler", "max_chunk_seconds": 6.0}},
        {"name": "reference_fast", "enabled": False, "parameters": {"speed": 1.2, "steps": 32, "cfg_strength": 2.0, "sway": -1.0, "max_chunk_seconds": 6.0}},
    ],
    "emotivoice": [
        {"name": "male9017_neutral", "enabled": True, "parameters": {"speaker": "9017", "prompt": "Neutral", "speed": 1.0, "pitch_predictor_scale": 1.0, "energy_predictor_scale": 1.0}},
        {"name": "male9017_sad", "enabled": False, "parameters": {"speaker": "9017", "prompt": "Sad", "speed": 1.0, "pitch_predictor_scale": 1.0, "energy_predictor_scale": 1.0}},
    ],
    "styletts2": [
        {"name": "reference_quality", "enabled": True, "parameters": {"speed": 1.0, "alpha": 0.1, "beta": 0.3, "diffusion_steps": 10, "embedding_scale": 1.0, "pitch_scale": 1.0, "max_chunk_words": 45}},
        {"name": "reference_expressive", "enabled": False, "parameters": {"speed": 1.1, "alpha": 0.5, "beta": 0.9, "diffusion_steps": 7, "embedding_scale": 1.2, "pitch_scale": 1.05, "max_chunk_words": 35}},
    ],
    "omnivoice": [
        {"name": "male_or_user_reference", "enabled": True, "reference": "auto", "parameters": {"instruct": "male, middle-aged, low pitch, british accent", "speed": 1.0, "num_step": 64, "guidance_scale": 2.0, "dtype": "float16", "max_chunk_seconds": 12.0}},
        {"name": "clone_reference", "enabled": False, "reference": True, "parameters": {"speed": 1.0, "num_step": 64, "guidance_scale": 2.0, "dtype": "float16", "max_chunk_seconds": 12.0}},
    ],
    "supertonic": [
        {"name": "m2_deep_serious", "enabled": True, "parameters": {"voice": "M2", "lang": "en", "speed": 1.0, "steps": 16, "max_chunk_chars": 360}},
        {"name": "f1_fast", "enabled": False, "parameters": {"voice": "F1", "lang": "en", "speed": 1.2, "steps": 12, "max_chunk_chars": 300}},
    ],
}
print("Run:", RUN_ID)
print("Cases:", [(c["name"], len(c["text"].split())) for c in TEXT_CASES])
print("Enabled profiles:", {m: [p["name"] for p in profiles if p.get("enabled", True)] for m, profiles in MODEL_PROFILES.items()})
'''

_passages = {
    str(count): Path(__file__).with_name(f"sample_text_{count}.txt").read_text(encoding="utf-8").strip()
    for count in (10, 100, 500)
}
for _count, _passage in _passages.items():
    if len(_passage.split()) != int(_count):
        raise ValueError(f"sample_text_{_count}.txt must contain exactly {_count} words")
CONFIG_CODE = CONFIG_CODE.replace("__SAMPLE_PASSAGES_JSON__", json.dumps(_passages, ensure_ascii=False, indent=4))

KERNEL_CODE = r'''
import csv
import hashlib
import html
import importlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import zipfile
from collections import deque


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)

# Lightweight notebook/report packages only; model packages stay in their own venvs.
kernel_packages = {
    "pynvml": "nvidia-ml-py>=12,<14", "psutil": "psutil>=5.9", "tqdm": "tqdm>=4.66",
    "soundfile": "soundfile>=0.12", "pandas": "pandas>=2", "matplotlib": "matplotlib>=3.7", "uv": "uv>=0.6",
}
KERNEL_READY = True
for module, package in kernel_packages.items():
    try:
        importlib.import_module(module)
    except ImportError:
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", package], env=env, check=True)
            importlib.invalidate_caches()
            importlib.import_module(module)
        except Exception as exc:
            KERNEL_READY = False
            atomic_json(RUN_DIR / "kernel_failure.json", {"module": module, "error": str(exc)})
            print("Kernel dependency failed:", module, exc)
            break

if KERNEL_READY:
    import pandas as pd
    import psutil
    import pynvml
    import soundfile as sf
    from IPython.display import HTML, Audio, FileLink, display
    from tqdm.auto import tqdm
print("Kernel helpers ready:", KERNEL_READY)
'''

HARDWARE_CODE = r'''
GPU_INVENTORY = []
GPU_HANDLES = {}
HARDWARE_READY = False
hardware_error = None
try:
    if not KERNEL_READY:
        raise RuntimeError("Notebook dependencies are unavailable; enable Internet and rerun.")
    if not (0 < CONFIG["max_vram_gb"] <= 5):
        raise ValueError("max_vram_gb must be in (0, 5] decimal GB")
    if not 1 <= CONFIG["nvml_interval_ms"] <= 1000:
        raise ValueError("nvml_interval_ms must be between 1 and 1000")
    if CONFIG["budget_scope"] not in {"device_total", "process_tree"}:
        raise ValueError("budget_scope must be device_total or process_tree")
    pynvml.nvmlInit()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = list(range(pynvml.nvmlDeviceGetCount()))
    if visible is not None:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        handles = [pynvml.nvmlDeviceGetHandleByUUID(token) if token.startswith("GPU-") else pynvml.nvmlDeviceGetHandleByIndex(int(token)) for token in tokens]
    else:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical]
    driver = pynvml.nvmlSystemGetDriverVersion()
    driver = driver.decode() if isinstance(driver, bytes) else driver
    for slot, handle in enumerate(handles):
        name = pynvml.nvmlDeviceGetName(handle)
        name = name.decode() if isinstance(name, bytes) else name
        gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
        gpu_uuid = gpu_uuid.decode() if isinstance(gpu_uuid, bytes) else gpu_uuid
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        GPU_INVENTORY.append({"slot": slot, "uuid": gpu_uuid, "name": name,
                              "total_bytes": info.total, "baseline_used_bytes": info.used, "driver": driver})
        GPU_HANDLES[slot] = handle
    if len(GPU_INVENTORY) != 2:
        raise RuntimeError(f"Expected two GPUs; found {len(GPU_INVENTORY)}. Select Kaggle GPU T4 x2.")
    if CONFIG["require_two_t4"] and not all("T4" in g["name"] for g in GPU_INVENTORY):
        raise RuntimeError("Select GPU T4 x2, or explicitly disable require_two_t4 to test other hardware.")
    HARDWARE_READY = True
except Exception as exc:
    hardware_error = f"{type(exc).__name__}: {exc}"
atomic_json(RUN_DIR / "hardware.json", {"ready": HARDWARE_READY, "error": hardware_error,
    "gpus": GPU_INVENTORY, "kernel_python": sys.version, "platform": platform.platform(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "peak_measurement": "NVML sampled total/device and process tree; allocator peaks separately. No instantaneous hard-peak certificate."})
print(json.dumps({"hardware_ready": HARDWARE_READY, "error": hardware_error, "gpus": GPU_INVENTORY}, indent=2))
'''

SUPERVISOR_CODE = r'''
SCRIPT_NAMES = {"kokoro": "kokoro_samples.py", "f5": "f5_samples.py", "emotivoice": "emotivoice_samples.py",
                "styletts2": "styletts2_samples.py", "omnivoice": "omnivoice_samples.py", "supertonic": "supertonic_samples.py"}
JOB_RESULTS = []
SAMPLE_RESULTS = []
TIMELINE = []

def slug(value, maximum=48):
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value)).strip("-")
    return (value or "unnamed")[:maximum]

def profile_jobs():
    jobs = []
    speaker_mode = CONFIG.get("speaker_mode", "auto")
    if speaker_mode not in {"auto", "reference", "default"}:
        raise ValueError("speaker_mode must be auto, reference, or default")
    for case_index, case in enumerate(TEXT_CASES):
        text = case["text"].strip()
        count = len(text.split())
        if not 10 <= count <= 500:
            raise ValueError(f"Case {case['name']!r} has {count} words, outside 10-500")
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
        for model in CONFIG["install_models"]:
            if model not in SCRIPT_NAMES:
                raise ValueError(f"Unknown model {model}")
            for profile_index, profile in enumerate(MODEL_PROFILES.get(model, [])):
                if not profile.get("enabled", True):
                    continue
                parameters = dict(profile.get("parameters", {}))
                forbidden = set(parameters) & {"config", "text", "text_file", "device", "n", "seed", "output_dir", "max_vram_gb", "estimate_only", "self_test"}
                if forbidden:
                    raise ValueError(f"Profile {model}/{profile['name']} has supervisor-controlled keys: {sorted(forbidden)}")
                count_takes = profile.get("n", CONFIG["n"])
                seed = profile.get("seed", CONFIG["seed"])
                if type(count_takes) is not int or count_takes < 1 or type(seed) is not int or not 0 <= seed <= 2**63 - count_takes:
                    raise ValueError("Profile n and seed must be positive/nonnegative integers within signed 64-bit range")
                # Incorporate case ordinal/config identity: duplicate names cannot overwrite runs.
                config_hash = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:8]
                job_id = f"{model}__{slug(profile['name'])}__w{count:03d}__c{case_index:02d}p{profile_index:02d}__{text_hash}_{config_hash}"
                job_dir = RUN_DIR / "jobs" / job_id
                job_dir.mkdir(parents=True, exist_ok=False)
                text_path = job_dir / "text.txt"
                text_path.write_text(text, encoding="utf-8")
                settings = {"text_file": str(text_path), "n": count_takes, "seed": seed,
                            "device": "cuda:0", "sample_rate": CONFIG["sample_rate"], "pause_ms": CONFIG["pause_ms"],
                            "max_vram_gb": CONFIG["max_vram_gb"], "output_dir": str(job_dir / "raw")}
                setup = SETUP_RESULTS.get(model, {"status": "failed", "error": "Model setup not requested/completed"})
                resources = dict(setup.get("resources", {}))
                settings.update({k: str(v) if isinstance(v, Path) else v for k, v in resources.items() if v is not None})
                settings.update({k: v for k, v in parameters.items() if v is not None})
                if model == "kokoro":
                    voice_dir = Path(resources.get("checkpoint", ".")).parent / "voices"
                    voices = [voice.strip() for voice in settings.get("voice", "am_fenrir").split(",")]
                    settings["voice"] = ",".join(str(voice_dir / f"{voice}.pt") if (voice_dir / f"{voice}.pt").is_file() else voice for voice in voices)
                error = None
                blocked_status = "setup_failed"
                speaker_source = "preset"
                if setup.get("status") != "ready":
                    error = setup.get("error", "Model environment is unavailable")
                if model in {"f5", "styletts2"}:
                    speaker_source = "user_reference" if parameters.get("reference_audio") or (CONFIG.get("reference_audio") and speaker_mode != "default") else "synthetic_male_reference"
                    settings.setdefault("reference_audio", str(REFERENCE_AUDIO) if REFERENCE_AUDIO else None)
                    if model == "f5":
                        settings.setdefault("reference_text", REFERENCE_TEXT)
                    if not settings.get("reference_audio") or (model == "f5" and not settings.get("reference_text")):
                        error = "Reference WAV/transcript preparation failed; see reference_report.json. F5 requires an exact transcript."
                if model == "omnivoice":
                    reference_choice = profile.get("reference", "auto")
                    if type(reference_choice) is not bool and reference_choice != "auto":
                        raise ValueError("OmniVoice profile reference must be True, False, or 'auto'")
                    cached_prompt = bool(settings.get("prompt_cache") and Path(settings["prompt_cache"]).is_file())
                    use_reference = bool(settings.get("ref_audio")) or cached_prompt or reference_choice is True or speaker_mode == "reference" or (reference_choice == "auto" and speaker_mode == "auto" and bool(CONFIG.get("reference_audio")))
                    if use_reference:
                        if not cached_prompt:
                            settings.setdefault("ref_audio", str(REFERENCE_AUDIO) if REFERENCE_AUDIO else None)
                            settings.setdefault("ref_text", REFERENCE_TEXT)
                        # Preset voice attributes must not conflict with the supplied person's voice.
                        clone_parameters = profile.get("clone_parameters", {})
                        forbidden_clone = set(clone_parameters) & {"config", "text", "text_file", "device", "n", "seed", "output_dir", "max_vram_gb", "estimate_only", "self_test"}
                        if forbidden_clone:
                            raise ValueError(f"clone_parameters contains supervisor-controlled keys: {sorted(forbidden_clone)}")
                        if "instruct" not in clone_parameters:
                            settings.pop("instruct", None)
                        settings.update(clone_parameters)
                        speaker_source = "cached_reference_prompt" if cached_prompt else "user_reference" if parameters.get("ref_audio") or (CONFIG.get("reference_audio") and speaker_mode != "default") else "synthetic_male_reference"
                        if not cached_prompt and (not settings.get("ref_audio") or not settings.get("ref_text")):
                            error = "Reference WAV/transcript preparation failed; see reference_report.json. OmniVoice requires an exact transcript in this adapter."
                    else:
                        speaker_source = "voice_design"
                if speaker_mode == "reference" and model not in {"f5", "styletts2", "omnivoice"}:
                    error = "This model supports preset speakers rather than arbitrary reference-WAV cloning."
                    blocked_status = "reference_unsupported"
                if model in {"emotivoice", "supertonic"}:
                    settings["allow_unverified_cuda_budget"] = True
                settings = {k: v for k, v in settings.items() if v is not None}
                config_path = job_dir / "config.json"
                atomic_json(config_path, settings)
                job = {"job_id": job_id, "model": model, "profile": profile["name"], "case": case["name"],
                       "word_count": count, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "n": count_takes, "seed": seed, "job_dir": str(job_dir), "settings": settings,
                       "speaker_source": speaker_source, "blocked_status": blocked_status,
                       "config_path": str(config_path), "python": setup.get("python"), "blocked_reason": error}
                atomic_json(job_dir / "job.json", job)
                jobs.append(job)
    return jobs

def process_tree(pid):
    ids = {pid}
    if pid:
        try:
            ids.update(child.pid for child in psutil.Process(pid).children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return ids - {None}

def nvml_process_usage(handle, tracked):
    allocations = {}
    available = False
    for getter_name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
        try:
            for proc in getattr(pynvml, getter_name)(handle):
                if proc.pid in tracked:
                    value = proc.usedGpuMemory
                    if isinstance(value, int) and 0 <= value < 2**60:
                        allocations[proc.pid] = max(value, allocations.get(proc.pid, 0))
                        available = True
        except pynvml.NVMLError:
            continue
    return (sum(allocations.values()) if available else None), sorted(allocations)

class JobMonitor:
    """Monitor external allocations from before model loading through process exit."""
    def __init__(self, job, slot):
        self.job, self.slot, self.pid = job, slot, None
        self.handle = GPU_HANDLES[slot]
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.process_ids = set()
        self.process_births = {}
        self.device_peak = 0
        self.process_peak = None
        self.last_device = 0
        self.last_process = None
        self.violation = None
        self.error = None
        self.poll_count = 0
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.observe, daemon=True)

    def start(self):
        self.thread.start()

    def observe(self):
        path = Path(self.job["job_dir"]) / "vram_trace.csv"
        budget = int(CONFIG["max_vram_gb"] * 1e9)
        interval = CONFIG["nvml_interval_ms"] / 1000
        fields = ["elapsed_seconds", "device_used_bytes", "process_tree_used_bytes", "gpu_utilization_percent", "memory_utilization_percent", "worker_pid", "tracked_gpu_pids"]
        try:
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                while not self.stop_event.is_set():
                    tick = time.perf_counter()
                    if self.pid:
                        self.process_ids.update(process_tree(self.pid))
                        for pid in self.process_ids:
                            if pid not in self.process_births:
                                try:
                                    self.process_births[pid] = psutil.Process(pid).create_time()
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                    info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    usage, gpu_pids = nvml_process_usage(self.handle, self.process_ids)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    self.last_device, self.last_process = info.used, usage
                    self.device_peak = max(self.device_peak, info.used)
                    if usage is not None:
                        self.process_peak = max(self.process_peak or 0, usage)
                    self.poll_count += 1
                    writer.writerow({"elapsed_seconds": f"{tick-self.started:.6f}", "device_used_bytes": info.used,
                        "process_tree_used_bytes": usage, "gpu_utilization_percent": utilization.gpu,
                        "memory_utilization_percent": utilization.memory, "worker_pid": self.pid,
                        "tracked_gpu_pids": ";".join(map(str, gpu_pids))})
                    observed = usage if CONFIG["budget_scope"] == "process_tree" and usage is not None else info.used
                    scope = "process_tree" if CONFIG["budget_scope"] == "process_tree" and usage is not None else "device_total"
                    if observed > budget and self.violation is None:
                        self.violation = {"observed_bytes": observed, "scope": scope,
                                          "budget_bytes": budget, "elapsed_seconds": tick-self.started}
                    if self.poll_count % 100 == 0:
                        stream.flush()
                    self.stop_event.wait(max(0.001, interval-(time.perf_counter()-tick)))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.error = "Monitor thread did not finish"
        result = {"nvml_sampled_device_peak_bytes": self.device_peak,
                  "nvml_sampled_process_tree_peak_bytes": self.process_peak,
                  "requested_interval_ms": CONFIG["nvml_interval_ms"], "samples": self.poll_count,
                  "budget_violation": self.violation, "monitor_error": self.error,
                  "hard_instantaneous_peak_certified": False}
        atomic_json(Path(self.job["job_dir"]) / "memory_monitor.json", result)
        return result

class Worker:
    def __init__(self, job, slot):
        import codecs
        self.job, self.slot = job, slot
        self.started = time.perf_counter()
        self.latest_line = "Starting interpreter and loading model"
        self.tail = deque(maxlen=40)
        self.abort_reason = None
        self.monitor = JobMonitor(job, slot)
        self.monitor.start()
        env = globals().get("BASE_ENV", os.environ).copy()
        env.update({"CUDA_VISIBLE_DEVICES": GPU_INVENTORY[slot]["uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(CONFIG["cpu_threads_per_job"]),
                    "MKL_NUM_THREADS": str(CONFIG["cpu_threads_per_job"]), "TOKENIZERS_PARALLELISM": "false"})
        espeak_library = SETUP_RESULTS[self.job["model"]].get("espeak_library") or env.get("PHONEMIZER_ESPEAK_LIBRARY")
        if espeak_library:
            env["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_library
        else:
            env.pop("PHONEMIZER_ESPEAK_LIBRARY", None)
        command = [job["python"], str(SCRIPTS_DIR / SCRIPT_NAMES[job["model"]]), "--config", job["config_path"]]
        atomic_json(Path(job["job_dir"]) / "command.json", {"argv": command, "gpu_uuid": env["CUDA_VISIBLE_DEVICES"],
                    "logical_device": "cuda:0", "cpu_threads": CONFIG["cpu_threads_per_job"]})
        try:
            self.process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=0, start_new_session=True, cwd=str(ROOT))
            self.monitor.pid = self.process.pid
            self.monitor.process_ids.add(self.process.pid)
        except Exception:
            self.monitor.finish()
            raise

        def read_output():
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            remainder = ""
            try:
                with (Path(job["job_dir"]) / "generation.log").open("w", encoding="utf-8") as log:
                    while True:
                        data = os.read(self.process.stdout.fileno(), 65536)
                        if not data:
                            break
                        decoded = decoder.decode(data)
                        log.write(decoded)
                        log.flush()
                        fragments = re.split(r"[\r\n]", remainder + decoded)
                        remainder = fragments.pop()
                        for fragment in fragments:
                            clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", fragment).strip()
                            if clean:
                                self.latest_line = clean[-500:]
                                self.tail.append(clean)
                    final = remainder + decoder.decode(b"", final=True)
                    if final.strip():
                        self.latest_line = final.strip()[-500:]
                        self.tail.append(final.strip())
            except Exception as exc:
                self.latest_line = "Log reader error: " + str(exc)
        self.reader = threading.Thread(target=read_output, daemon=True)
        self.reader.start()
        TIMELINE.append({"event": "start", "job_id": job["job_id"], "model": job["model"],
                         "gpu_slot": slot, "gpu_uuid": GPU_INVENTORY[slot]["uuid"], "timestamp": time.time()})

    def terminate(self, reason):
        self.abort_reason = reason
        self.monitor.process_ids.update(process_tree(self.process.pid))
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=10)
        for pid in self.monitor.process_ids - {self.process.pid}:
            try:
                descendant = psutil.Process(pid)
                if descendant.create_time() == self.monitor.process_births.get(pid):
                    descendant.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def finish(self):
        self.process.wait()
        # A parent can leave a child holding pipe/GPU handles. Kill only tracked
        # descendants whose creation time still matches, avoiding reused PIDs.
        for pid in self.monitor.process_ids - {self.process.pid}:
            try:
                descendant = psutil.Process(pid)
                if descendant.create_time() == self.monitor.process_births.get(pid):
                    descendant.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.reader.join(timeout=10)
        exit_code = self.process.returncode
        status = "completed" if exit_code == 0 and not self.abort_reason else "failed"
        if self.abort_reason:
            status = "budget_exceeded" if self.abort_reason.startswith("VRAM") else "failed"
        # Reap remaining descendants even after a nominally successful parent exit.
        deadline = time.perf_counter() + CONFIG["gpu_release_timeout_seconds"]
        busy = []
        while time.perf_counter() < deadline:
            _, busy = nvml_process_usage(GPU_HANDLES[self.slot], self.monitor.process_ids)
            if not busy:
                break
            time.sleep(0.05)
        if busy:
            status = "gpu_release_failed"
        memory = self.monitor.finish()
        if memory["budget_violation"] and CONFIG["enforce_observed_budget"]:
            status = "budget_exceeded" if not busy else status
            self.abort_reason = self.abort_reason or "VRAM exceeded observed budget: " + json.dumps(memory["budget_violation"])
        if memory["monitor_error"]:
            status = "monitor_failed"
        result = {**self.job, "gpu_slot": self.slot, "gpu_uuid": GPU_INVENTORY[self.slot]["uuid"],
                  "status": status, "exit_code": exit_code, "process_wall_seconds": time.perf_counter()-self.started,
                  "error": self.abort_reason or ("\n".join(self.tail)[-5000:] if exit_code else None),
                  "gpu_release_failed_pids": busy, **memory}
        atomic_json(Path(self.job["job_dir"]) / "supervisor_result.json", result)
        TIMELINE.append({"event": "finish", "job_id": self.job["job_id"], "model": self.job["model"],
                         "gpu_slot": self.slot, "timestamp": time.time(), "status": status})
        return result

def preferred_job_index(pending, active_models):
    for index, job in enumerate(pending):
        if job["model"] not in active_models:
            return index
    return 0 if pending else None

def live_html(active, completed, total):
    rows = []
    for slot in range(2):
        worker = active.get(slot)
        if worker:
            label = html.escape(worker.job["job_id"])
            line = html.escape(worker.latest_line)
            memory = worker.monitor.last_device / 1e9
            process_memory = worker.monitor.last_process
            process_text = "unavailable" if process_memory is None else f"{process_memory/1e9:.3f} GB"
            rows.append(f"<tr><td>GPU {slot}</td><td>{label}<br><pre>{line}</pre></td><td>{memory:.3f} GB device<br>{process_text} process</td></tr>")
        else:
            rows.append(f"<tr><td>GPU {slot}</td><td>Idle</td><td></td></tr>")
    return f"<p>{completed}/{total} jobs finished. One isolated worker per GPU; distinct models preferred.</p><table><tr><th>Slot</th><th>Model progress</th><th>Live NVML usage</th></tr>{''.join(rows)}</table>"

def run_benchmark(jobs):
    pending = []
    for job in jobs:
        reason = job["blocked_reason"] or (hardware_error if not HARDWARE_READY else None)
        if reason:
            result = {**job, "status": job.get("blocked_status", "setup_failed"), "error": reason, "exit_code": None, "gpu_slot": None,
                      "gpu_uuid": None, "process_wall_seconds": None, "hard_instantaneous_peak_certified": False}
            JOB_RESULTS.append(result)
            atomic_json(Path(job["job_dir"]) / "supervisor_result.json", result)
        else:
            pending.append(job)
    active = {}
    disabled_slots = set()
    progress = tqdm(total=len(jobs), initial=len(JOB_RESULTS), desc="Benchmark jobs", unit="job")
    live = display(HTML(live_html(active, len(JOB_RESULTS), len(jobs))), display_id=True)
    try:
        while pending or active:
            for slot in range(2):
                if slot in active or slot in disabled_slots or not pending:
                    continue
                index = preferred_job_index(pending, {worker.job["model"] for worker in active.values()})
                job = pending.pop(index)
                try:
                    active[slot] = Worker(job, slot)
                except Exception as exc:
                    result = {**job, "status": "launch_failed", "error": f"{type(exc).__name__}: {exc}",
                              "exit_code": None, "gpu_slot": slot, "gpu_uuid": GPU_INVENTORY[slot]["uuid"]}
                    JOB_RESULTS.append(result)
                    atomic_json(Path(job["job_dir"]) / "supervisor_result.json", result)
                    progress.update(1)
            for slot, worker in list(active.items()):
                elapsed = time.perf_counter()-worker.started
                if worker.process.poll() is None:
                    if worker.monitor.error:
                        worker.terminate("NVML monitor failed: " + worker.monitor.error)
                    elif worker.monitor.violation and CONFIG["enforce_observed_budget"]:
                        worker.terminate("VRAM exceeded observed budget: " + json.dumps(worker.monitor.violation))
                    elif elapsed > CONFIG["job_timeout_seconds"]:
                        worker.terminate("Job timeout")
                if worker.process.poll() is not None:
                    result = worker.finish()
                    JOB_RESULTS.append(result)
                    progress.update(1)
                    if result.get("gpu_release_failed_pids"):
                        disabled_slots.add(slot)
                    del active[slot]
            if len(disabled_slots) == 2 and pending:
                for job in pending:
                    result = {**job, "status": "gpu_unavailable", "error": "Both GPUs retain tracked allocations", "exit_code": None}
                    JOB_RESULTS.append(result)
                    atomic_json(Path(job["job_dir"]) / "supervisor_result.json", result)
                    progress.update(1)
                pending.clear()
            live.update(HTML(live_html(active, len(JOB_RESULTS), len(jobs))))
            time.sleep(0.2)
    except BaseException as supervisor_error:
        for worker in active.values():
            worker.terminate("Interrupted by user" if isinstance(supervisor_error, KeyboardInterrupt) else "Supervisor error: " + str(supervisor_error))
            JOB_RESULTS.append(worker.finish())
        for job in pending:
            result = {**job, "status": "cancelled", "error": "Supervisor interrupted: " + str(supervisor_error), "exit_code": None}
            JOB_RESULTS.append(result)
            atomic_json(Path(job["job_dir"]) / "supervisor_result.json", result)
        print("Supervisor stopped:", type(supervisor_error).__name__, str(supervisor_error))
        print("Run the report/ZIP cell to package all completed and partial outputs.")
    finally:
        progress.close()
        atomic_json(RUN_DIR / "supervisor_results.json", JOB_RESULTS)
        atomic_json(RUN_DIR / "concurrency_timeline.json", TIMELINE)
        live.update(HTML(live_html({}, len(JOB_RESULTS), len(jobs))))

jobs = profile_jobs()
atomic_json(RUN_DIR / "experiment_plan.json", {"config": CONFIG, "cases": TEXT_CASES, "profiles": MODEL_PROFILES, "jobs": jobs})
print(f"{len(jobs)} model/profile/text jobs; {sum(j['n'] for j in jobs)} planned WAVs")
if KERNEL_READY:
    run_benchmark(jobs)
else:
    JOB_RESULTS = [{**job, "status": "kernel_failed", "error": hardware_error, "exit_code": None} for job in jobs]
    atomic_json(RUN_DIR / "supervisor_results.json", JOB_RESULTS)
'''

REPORT_CODE = r'''
PLANNER_SCOPE = {
    "kokoro": "First loaded voice-conditioned G2P/duration pass; not a warmed latency guarantee",
    "f5": "Allocated frame plan including CFM minimum-length clamp; no waveform generation",
    "emotivoice": "Warmed speaker/style-conditioned duration preflight",
    "styletts2": "BERT/style diffusion/duration preflight for all requested takes",
    "omnivoice": "Upstream rule token allocation; separate cold preflight/import/reference timing",
    "supertonic": "Warmed voice-conditioned duration predictor; startup probes excluded",
}

def read_json_safely(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

def find_manifest(job):
    raw = Path(job["job_dir"]) / "raw"
    preferred = {"kokoro": ["manifest.json", "preview.json"], "f5": ["manifest.json"],
                 "styletts2": ["manifest.json"], "emotivoice": ["emotivoice_manifest.json"],
                 "supertonic": ["supertonic_manifest.json"], "omnivoice": ["manifest.json", "estimate.json"]}
    for filename in preferred[job["model"]]:
        for path in sorted(raw.rglob(filename)):
            data = read_json_safely(path)
            if data is not None:
                return path, data
    return None, {}

def native_schema(job, data):
    model = job["model"]
    samples = data.get("samples", data.get("outputs", data.get("runs", [])))
    planner = cold = None
    prediction = None
    export_rate = data.get("export_sample_rate", data.get("export_rate"))
    predictions = []
    if model == "kokoro":
        prediction = data.get("predicted_export_samples")
        planner, cold = data.get("preflight_seconds"), data.get("startup_seconds")
    elif model == "f5":
        preview = data.get("preview", {})
        prediction = preview.get("exported_samples")
        planner, cold = preview.get("preflight_seconds"), preview.get("startup_seconds")
        export_rate = data.get("settings", {}).get("sample_rate")
    elif model == "styletts2":
        planner, cold = data.get("preflight_seconds"), data.get("startup_seconds")
        export_rate = data.get("settings", {}).get("sample_rate")
        predictions = [{"seed": p.get("seed"), "predicted_samples": p.get("exported_samples"), "predicted_seconds": p.get("seconds")} for p in data.get("preview", [])]
    elif model in {"emotivoice", "supertonic"}:
        prediction = data.get("predicted_samples")
        planner, cold = data.get("warm_preview_seconds"), data.get("startup_seconds")
    elif model == "omnivoice":
        plan = data.get("plan", {})
        prediction = plan.get("predicted_export_samples")
        planner = plan.get("planner_ms", 0) / 1000 if "planner_ms" in plan else None
        cold = plan.get("cold_preflight_ms", 0) / 1000 if "cold_preflight_ms" in plan else None
        export_rate = plan.get("sample_rate")
    export_rate = export_rate or job.get("settings", {}).get("sample_rate")
    if not export_rate:
        export_rate = {"emotivoice": 16000, "supertonic": 44100}.get(model, 24000)
    if prediction is not None:
        predictions = [{"seed": job["seed"] + index, "predicted_samples": prediction, "predicted_seconds": prediction/export_rate} for index in range(job["n"])]
    rows = []
    for index, record in enumerate(samples):
        seed = record.get("seed", job["seed"] + index)
        predicted = record.get("predicted_samples", prediction)
        if model == "styletts2":
            previews = data.get("preview", [])
            match = next((p for p in previews if p.get("seed") == seed), None)
            if match:
                predicted = match.get("exported_samples")
        rows.append({"seed": seed, "path": record.get("path", record.get("file")),
                     "predicted_samples": predicted, "generation_seconds": record.get("generation_seconds"),
                     "native_record": record})
    return {"rows": rows, "prediction": prediction, "predictions": predictions, "export_rate": export_rate,
            "planner_seconds": planner, "startup_seconds": cold}

def allocator_peaks(data):
    allocated, reserved = [], []
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    if key in {"torch_peak_allocated_bytes", "peak_torch_allocated_bytes"}:
                        allocated.append(item)
                    elif key in {"torch_peak_reserved_bytes", "peak_torch_reserved_bytes"}:
                        reserved.append(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(data)
    return max(allocated, default=None), max(reserved, default=None)

def collect_reports():
    audio_dir = PACKAGE_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    job_rows, sample_rows = [], []
    for job in JOB_RESULTS:
        manifest_path, manifest = find_manifest(job)
        schema = native_schema(job, manifest)
        allocated, reserved = allocator_peaks(manifest)
        listed = {Path(row["path"]).resolve(): row for row in schema["rows"] if row["path"]}
        found = sorted((Path(job["job_dir"]) / "raw").rglob("*.wav"))
        observed_errors = []
        final_count = 0
        for index, source in enumerate(found):
            native = listed.get(source.resolve())
            seed = native["seed"] if native else None
            predicted = native["predicted_samples"] if native else schema["prediction"]
            filename = f"{job['job_id']}__seed{seed if seed is not None else 'unknown'}__take{index+1:03d}"
            row = {"job_id": job["job_id"], "model": job["model"], "profile": job["profile"], "case": job["case"],
                   "word_count": job["word_count"], "seed": seed, "take": index+1,
                   "speaker_source": job.get("speaker_source"),
                   "job_status": job["status"], "original_path": str(source), "predicted_samples": predicted,
                   "planner_seconds": schema["planner_seconds"], "planner_scope": PLANNER_SCOPE[job["model"]],
                   "generation_seconds": native["generation_seconds"] if native else None,
                   "gpu_slot": job.get("gpu_slot"), "gpu_uuid": job.get("gpu_uuid"),
                   "nvml_sampled_process_tree_peak_bytes": job.get("nvml_sampled_process_tree_peak_bytes"),
                   "nvml_sampled_device_peak_bytes": job.get("nvml_sampled_device_peak_bytes"),
                   "torch_peak_allocated_bytes": allocated, "torch_peak_reserved_bytes": reserved,
                   "actual_samples": None, "sample_rate": None, "channels": None,
                   "planned_sample_rate": schema["export_rate"], "sample_rate_match": None,
                   "actual_seconds": None, "predicted_seconds": predicted / schema["export_rate"] if predicted is not None else None,
                   "relative_duration_error": None, "duration_error_percent": None,
                   "exact_sample_count_match": None, "under_one_percent": None}
            try:
                info = sf.info(source)
                actual = info.frames
                actual_seconds = actual / info.samplerate
                planned_rate = schema["export_rate"]
                predicted_seconds = predicted / planned_rate if predicted is not None else None
                error = abs(actual_seconds-predicted_seconds) / actual_seconds if actual_seconds > 0 and predicted_seconds is not None else None
                is_temporary = "temporary" in source.name or bool(re.search(r"\.(?:tmp|part)\.wav$", source.name, re.I))
                valid = native is not None and error is not None and error < 0.01 and actual == predicted and info.samplerate == planned_rate and info.channels == 1 and not is_temporary
                partial = not valid
                destination = audio_dir / (("PARTIAL__" if partial else "") + filename + f"__sr{info.samplerate}.wav")
                shutil.copy2(source, destination)
                row.update({"audio_file": str(destination.relative_to(PACKAGE_DIR)), "actual_samples": actual,
                            "sample_rate": info.samplerate, "channels": info.channels,
                            "planned_sample_rate": planned_rate, "sample_rate_match": info.samplerate == planned_rate,
                            "actual_seconds": actual_seconds, "predicted_seconds": predicted_seconds,
                            "relative_duration_error": error, "duration_error_percent": error*100 if error is not None else None,
                            "exact_sample_count_match": actual == predicted if predicted is not None else None,
                            "under_one_percent": error < 0.01 if error is not None else None,
                            "audio_status": "partial" if partial else "verified", "error": None})
                if error is not None:
                    observed_errors.append(error)
                if valid:
                    final_count += 1
            except Exception as exc:
                destination = audio_dir / ("UNREADABLE__" + filename + ".wav")
                shutil.copy2(source, destination)
                row.update({"audio_file": str(destination.relative_to(PACKAGE_DIR)), "audio_status": "unreadable", "error": str(exc)})
            sample_rows.append(row)
        if job["status"] == "completed" and final_count != job["n"]:
            job["status"] = "report_validation_failed"
            job["error"] = f"Expected {job['n']} exact exports, independently verified {final_count}"
        for row in sample_rows:
            if row["job_id"] == job["job_id"]:
                row["job_status"] = job["status"]
        job_rows.append({"job_id": job["job_id"], "model": job["model"], "profile": job["profile"], "case": job["case"],
            "speaker_source": job.get("speaker_source"),
            "word_count": job["word_count"], "requested_samples": job["n"], "created_wavs": len(found), "verified_wavs": final_count,
            "status": job["status"], "error": job.get("error"), "gpu_slot": job.get("gpu_slot"), "gpu_uuid": job.get("gpu_uuid"),
            "exit_code": job.get("exit_code"), "process_wall_seconds": job.get("process_wall_seconds"),
            "planner_seconds": schema["planner_seconds"], "startup_seconds": schema["startup_seconds"],
            "planner_seconds_per_take_amortized": schema["planner_seconds"]/job["n"] if schema["planner_seconds"] is not None and job["model"] == "styletts2" else None,
            "planner_scope": PLANNER_SCOPE[job["model"]], "max_duration_error_percent": max(observed_errors)*100 if observed_errors else None,
            "planned_samples_json": json.dumps(schema["predictions"]), "planned_sample_rate": schema["export_rate"],
            "nvml_sampled_process_tree_peak_bytes": job.get("nvml_sampled_process_tree_peak_bytes"),
            "nvml_sampled_device_peak_bytes": job.get("nvml_sampled_device_peak_bytes"),
            "torch_peak_allocated_bytes": allocated, "torch_peak_reserved_bytes": reserved,
            "budget_violation": json.dumps(job["budget_violation"]) if job.get("budget_violation") else None,
            "monitor_error": job.get("monitor_error"), "trace_samples": job.get("samples"),
            "monitor_interval_ms": CONFIG["nvml_interval_ms"], "instantaneous_peak_certified": False,
            "native_manifest": str(manifest_path) if manifest_path else None})
    return job_rows, sample_rows

def run_quality_checks(sample_rows):
    """Independent CPU diagnostics; errors never prevent audio/report packaging."""
    quality_dir = RUN_DIR / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    output = quality_dir / "quality_report.json"
    if not CONFIG.get("quality_checks", False) or not sample_rows:
        result = {"status": "disabled" if sample_rows else "no_audio", "rows": []}
        atomic_json(output, result)
        return result
    jobs_by_id = {job["job_id"]: job for job in JOB_RESULTS}
    requests = []
    for row in sample_rows:
        job = jobs_by_id[row["job_id"]]
        text = (Path(job["job_dir"]) / "text.txt").read_text(encoding="utf-8")
        requests.append({"job_id": row["job_id"], "model": row["model"], "seed": row["seed"],
                         "audio_file": str((PACKAGE_DIR / row["audio_file"]).resolve()), "text": text})
    request_path = quality_dir / "requests.json"
    atomic_json(request_path, requests)
    log_path = quality_dir / "quality_check.log"
    setup_error = None
    try:
        python = ENV_DIR / "quality_checks" / "bin" / "python"
        if not python.is_file():
            setup_command(UV + ["venv", "--python", "3.11", "--seed", str(python.parent.parent)], log_path)
        install = UV + ["pip", "install", "--python", str(python), "--link-mode", "hardlink"]
        setup_command(install + ["numpy==1.26.4", "soundfile==0.13.1"], log_path)
        use_asr = bool(CONFIG.get("asr_quality_check", False))
        model = CONFIG.get("asr_model", "base.en")
        if use_asr:
            try:
                setup_command(install + ["faster-whisper==1.2.1", "ctranslate2==4.6.0",
                              "onnxruntime==1.23.2", "huggingface_hub==0.29.3", "av==15.1.0"], log_path)
                if model == "base.en":
                    model = str(ASSETS / "quality_asr" / "base.en")
                    downloader = "from huggingface_hub import snapshot_download; import sys; snapshot_download('Systran/faster-whisper-base.en', revision='3d3d5dee26484f91867d81cb899cfcf72b96be6c',local_dir=sys.argv[1],max_workers=1)"
                    setup_command([python, "-c", downloader, model], log_path)
            except Exception as exc:
                use_asr = False
                setup_error = str(exc)
                print("ASR setup failed; continuing with waveform diagnostics:", exc)
        command = [str(python), str(SCRIPTS_DIR / "audio_quality_check.py"), "--requests-json", str(request_path),
                   "--output-json", str(output), "--model", str(model), "--language", CONFIG.get("asr_language", "en"),
                   "--threads", str(CONFIG["cpu_threads_per_job"])]
        if use_asr:
            command.append("--asr")
        print(f"Checking {len(requests)} WAVs on CPU; ASR={use_asr}. Details: {log_path}")
        # Explicit CUDA hiding and CPU/int8 ASR keep diagnostics out of TTS VRAM/timing measurements.
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(command, env=BASE_ENV, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=CONFIG.get("quality_timeout_seconds", 7200))
        result = read_json_safely(output)
        if not isinstance(result, dict):
            raise RuntimeError("Quality helper returned no valid report")
        result["asr_setup_error"] = setup_error
        result["asr_completed_files"] = sum(row.get("asr_status") == "completed" for row in result.get("rows", []))
        result["asr_failed_files"] = sum(row.get("asr_status") == "failed" for row in result.get("rows", []))
        if CONFIG.get("asr_model", "base.en") == "base.en" and use_asr:
            result["asr_weight_revision"] = "3d3d5dee26484f91867d81cb899cfcf72b96be6c"
        atomic_json(output, result)
    except Exception as exc:
        result = {"status": "failed", "error": str(exc), "rows": [], "asr_setup_error": setup_error}
        atomic_json(output, result)
        print("Quality diagnostics failed; WAVs and timing reports will still be packaged:", exc)
    lookup = {(r.get("job_id"), str(Path(r.get("audio_file", "")).resolve())): r for r in result.get("rows", [])}
    for row in sample_rows:
        diagnostics = lookup.get((row["job_id"], str((PACKAGE_DIR / row["audio_file"]).resolve())))
        if diagnostics:
            row["quality_status"] = diagnostics.get("status")
            for field in ("asr_status", "asr_error", "longest_interior_silence_seconds", "clipped_sample_fraction", "word_error_rate", "deletions", "insertions", "substitutions", "transcript"):
                row[field] = diagnostics.get(field)
    return result

def package_results(job_rows, sample_rows):
    reports = PACKAGE_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    quality_report = run_quality_checks(sample_rows)
    atomic_json(reports / "quality.json", quality_report)
    pd.DataFrame(quality_report.get("rows", [])).to_csv(reports / "quality.csv", index=False)
    shutil.copytree(RUN_DIR / "quality", PACKAGE_DIR / "quality_details", dirs_exist_ok=True)
    jobs_frame, samples_frame = pd.DataFrame(job_rows), pd.DataFrame(sample_rows)
    jobs_frame.to_csv(reports / "jobs.csv", index=False)
    samples_frame.to_csv(reports / "samples.csv", index=False)
    atomic_json(reports / "jobs.json", job_rows)
    atomic_json(reports / "samples.json", sample_rows)
    atomic_json(reports / "experiment.json", {"config": CONFIG, "cases": TEXT_CASES, "profiles": MODEL_PROFILES,
                "hardware": GPU_INVENTORY, "setup": SETUP_RESULTS, "preflight": globals().get("PREFLIGHT_RESULTS", {}),
                "runtime_validation": "This run's data only; failures are included. Sampled NVML does not certify instantaneous peaks."})
    sources = PACKAGE_DIR / "source"
    sources.mkdir(exist_ok=True)
    for path in SCRIPTS_DIR.glob("*.py"):
        shutil.copy2(path, sources / path.name)
    if (SCRIPTS_DIR / "source_manifest.json").exists():
        shutil.copy2(SCRIPTS_DIR / "source_manifest.json", sources / "source_manifest.json")
    for name in ("hardware.json", "supervisor_results.json", "concurrency_timeline.json", "experiment_plan.json", "reference_report.json"):
        source = RUN_DIR / name
        if source.exists():
            shutil.copy2(source, reports / name)
    for source in (ROOT / "setup_report.json", ROOT / "preflight_report.json", ROOT / "reference_report.json", ROOT / "emotivoice_checkpoint_provenance.json"):
        if source.exists():
            shutil.copy2(source, reports / source.name)
    reference_root = ROOT / "reference"
    if reference_root.exists():
        for source in reference_root.rglob("*.json"):
            destination = reports / "reference_provenance" / source.relative_to(reference_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    logs_dir = ROOT / "logs" / "setup"
    if logs_dir.exists():
        shutil.copytree(logs_dir, PACKAGE_DIR / "setup_logs", dirs_exist_ok=True)
    for job in JOB_RESULTS:
        base = Path(job["job_dir"])
        target = PACKAGE_DIR / "job_details" / job["job_id"]
        target.mkdir(parents=True, exist_ok=True)
        for source in base.rglob("*"):
            if source.is_file() and source.suffix.lower() in {".json", ".txt", ".log", ".csv", ".pt"}:
                # .pt here is the small StyleTTS2 cached duration plan, never model checkpoints.
                relative = source.relative_to(base)
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    if CONFIG["include_reference_in_zip"] and globals().get("REFERENCE_AUDIO"):
        reference = Path(REFERENCE_AUDIO)
        if reference.exists():
            target = PACKAGE_DIR / "reference"
            target.mkdir(exist_ok=True)
            shutil.copy2(reference, target / reference.name)
            (target / "transcript.txt").write_text(REFERENCE_TEXT or "", encoding="utf-8")
    # A static report links to every named clip; suitable after unzipping.
    clips = []
    for row in sample_rows:
        href = html.escape(row["audio_file"], quote=True)
        label = html.escape(f"{row['model']} / {row['profile']} / {row['case']} / seed {row.get('seed')} / {row['audio_status']} / speaker: {row.get('speaker_source')}")
        clips.append(f'<p>{label}</p><audio controls preload="none" src="{href}"></audio><a href="{href}">WAV</a>')
    table = jobs_frame.to_html(index=False, escape=True) if not jobs_frame.empty else "<p>No jobs completed.</p>"
    figures_html = ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if job_rows:
            y = list(range(len(job_rows)))
            figure, axis = plt.subplots(figsize=(12, max(4, len(y)*0.42)))
            labels = [f"{r['model']}/{r['profile']}/{r['case']} GPU{r.get('gpu_slot')}" for r in job_rows]
            for field, offset, label in (("nvml_sampled_device_peak_bytes", -.24, "NVML device sampled"),
                                         ("nvml_sampled_process_tree_peak_bytes", 0, "NVML process tree sampled"),
                                         ("torch_peak_allocated_bytes", .24, "PyTorch allocator peak")):
                values = [r[field]/1e9 if r.get(field) is not None else float("nan") for r in job_rows]
                axis.barh([i+offset for i in y], values, height=.22, label=label)
            axis.axvline(CONFIG["max_vram_gb"], color="red", linestyle="--", label="Configured VRAM target")
            axis.set_yticks(y, labels)
            axis.invert_yaxis()
            axis.set_xlabel("GPU memory (decimal GB); absent measurements are blank")
            axis.legend(loc="best")
            figure.tight_layout()
            figure.savefig(reports / "vram_peaks.png", dpi=150)
            plt.close(figure)
            figures_html += '<h2>Memory measurements</h2><img style="max-width:100%" src="reports/vram_peaks.png">'
        timed = [r for r in job_rows if r.get("planner_seconds") is not None and r["planner_seconds"] > 0]
        if timed:
            figure, axis = plt.subplots(figsize=(10, 5))
            for model in sorted({r["model"] for r in timed}):
                rows = sorted([r for r in timed if r["model"] == model], key=lambda r:r["word_count"])
                axis.plot([r["word_count"] for r in rows], [r["planner_seconds"]*1000 for r in rows], "o", label=model)
            axis.set_yscale("log")
            axis.set_xlabel("Words")
            axis.set_ylabel("Measured planning time (ms, log scale)")
            axis.set_title("Model-specific timing scopes differ; StyleTTS2 includes all requested takes")
            axis.legend()
            axis.grid(True, alpha=.3)
            figure.tight_layout()
            figure.savefig(reports / "planning_latency.png", dpi=150)
            plt.close(figure)
            figures_html += '<h2>Planning latency</h2><img style="max-width:100%" src="reports/planning_latency.png">'
    except Exception as exc:
        atomic_json(reports / "plot_failure.json", {"error": str(exc)})
    quality_frame = pd.DataFrame(quality_report.get("rows", []))
    quality_columns = ["job_id", "model", "seed", "status", "asr_status", "asr_error",
                       "word_error_rate", "deletions", "insertions", "substitutions",
                       "longest_interior_silence_seconds", "clipped_sample_fraction", "transcript"]
    quality_table = (quality_frame.reindex(columns=[name for name in quality_columns if name in quality_frame.columns])
                     .to_html(index=False, escape=True) if not quality_frame.empty
                     else html.escape(str(quality_report.get("status"))))
    report_html = ('<!doctype html><meta charset="utf-8"><title>TTS benchmark</title>'
        '<style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}audio{width:500px;max-width:100%}</style>'
        '<h1>TTS timing and GPU memory benchmark</h1><p>NVML peaks are sampled; allocator peaks exclude driver/library memory. Failed and partial jobs are retained.</p>'
        + figures_html + table + '<h2>Word completeness and acoustic diagnostics</h2><p>ASR disagreements are screening results, not proof of a synthesis error. Silence can be intentional; inspect the transcript and listen.</p>'
        + quality_table + '<p>Full word alignments, silence spans, and ASR timestamps are in reports/quality.json.</p>'
        + '<h2>Recordings</h2>' + ''.join(clips))
    (PACKAGE_DIR / "index.html").write_text(report_html, encoding="utf-8")
    (PACKAGE_DIR / "README.txt").write_text(
        "Open index.html to compare reports and play recordings.\n"
        "reports/jobs.csv: every planned job, including setup/runtime/budget failures.\n"
        "reports/samples.csv: independent WAV header duration checks and canonical filenames.\n"
        "reports/quality.csv and quality.json: CPU acoustic checks, ASR transcripts and word edits; ASR is fallible.\n"
        "job_details/: exact config, text, native manifests, progress logs and NVML time series.\n"
        "source/: embedded standalone adapters and SHA256 provenance.\n"
        "PARTIAL/UNREADABLE filenames are not verified final samples.\n"
        "Audio length accuracy does not establish text intelligibility or natural speaking time.\n"
        "Dependency environments and model weights are excluded from this archive.\n", encoding="utf-8")
    archive = ROOT.parent / f"tts_benchmark_{RUN_ID}.zip"
    files = [path for path in PACKAGE_DIR.rglob("*") if path.is_file()]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zipped:
        for source in tqdm(files, desc="Packaging WAVs and reports", unit="file"):
            zipped.write(source, source.relative_to(PACKAGE_DIR))
    return jobs_frame, samples_frame, archive

if KERNEL_READY:
    job_rows, sample_rows = collect_reports()
    JOBS_TABLE, SAMPLES_TABLE, ZIP_PATH = package_results(job_rows, sample_rows)
    print(f"ZIP: {ZIP_PATH} ({ZIP_PATH.stat().st_size/1e6:.1f} MB)")
    display(FileLink(str(ZIP_PATH)))
    display(FileLink(str(PACKAGE_DIR / "index.html")))
    if not JOBS_TABLE.empty:
        display(JOBS_TABLE.reindex(columns=["model", "profile", "word_count", "status", "verified_wavs", "planner_seconds", "max_duration_error_percent", "nvml_sampled_process_tree_peak_bytes", "nvml_sampled_device_peak_bytes"]))
    if not SAMPLES_TABLE.empty:
        display(SAMPLES_TABLE.reindex(columns=["model", "profile", "word_count", "seed", "speaker_source", "predicted_seconds", "actual_seconds", "duration_error_percent", "longest_interior_silence_seconds", "word_error_rate", "audio_status", "audio_file"]))
    played_models = set()
    for row in sample_rows:
        if len(played_models) >= CONFIG["preview_players"]:
            break
        if row["audio_status"] == "verified" and row["model"] not in played_models:
            print(row["model"], row["profile"], row["case"], "seed", row["seed"])
            display(Audio(filename=str(PACKAGE_DIR / row["audio_file"])))
            played_models.add(row["model"])
else:
    ZIP_PATH = ROOT.parent / f"tts_benchmark_{RUN_ID}_failure.zip"
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in RUN_DIR.rglob("*.json"):
            archive.write(file, file.relative_to(RUN_DIR))
        for file in ROOT.glob("*report.json"):
            archive.write(file, "reports/" + file.name)
    print("Kernel dependency setup failed. Diagnostic ZIP saved:", ZIP_PATH)
    from IPython.display import FileLink, display
    display(FileLink(str(ZIP_PATH)))
'''

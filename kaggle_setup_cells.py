"""Setup-cell source embedded into the standalone Kaggle notebook.

This is a notebook build input, not a runtime TTS library. The notebook embeds
the strings below and runs them directly. Installation never allocates CUDA
memory, and never replaces the Kaggle kernel's Torch installation.
"""

from textwrap import dedent

SETUP_CODE = dedent(r'''
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

MODELS = ("kokoro", "f5", "emotivoice", "styletts2", "omnivoice", "supertonic")
RUNTIME = Path("/tmp/tts_benchmark_runtime")
ENV_DIR = RUNTIME / "envs"
UV_CACHE = RUNTIME / "uv_cache"
HF_CACHE = RUNTIME / "huggingface"
ASSETS = RUNTIME / "assets"
REPOS = RUNTIME / "repos"
SETUP_LOGS = ROOT / "logs" / "setup"
for directory in (ENV_DIR, UV_CACHE, HF_CACHE, ASSETS, REPOS, SETUP_LOGS):
    directory.mkdir(parents=True, exist_ok=True)

BASE_ENV = os.environ.copy()
BASE_ENV.update({
    "CUDA_VISIBLE_DEVICES": "", "HF_HOME": str(HF_CACHE),
    "HF_HUB_CACHE": str(HF_CACHE / "hub"), "UV_CACHE_DIR": str(UV_CACHE),
    "UV_PYTHON_INSTALL_DIR": str(RUNTIME / "python"),
    "GIT_LFS_SKIP_SMUDGE": "1", "TOKENIZERS_PARALLELISM": "false",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONUNBUFFERED": "1",
    "NLTK_DATA": str(RUNTIME / "nltk_data"),
    "HF_HUB_DOWNLOAD_TIMEOUT": "120", "HF_HUB_ETAG_TIMEOUT": "60",
})
# Subprocesses get their own environments and therefore cannot contaminate
# the notebook kernel or accidentally initialize CUDA during installation.
os.environ["NLTK_DATA"] = BASE_ENV["NLTK_DATA"]

def setup_command(command, log, *, cwd=None, env=None):
    command = [str(value) for value in command]
    with Path(log).open("a", encoding="utf-8") as stream:
        stream.write("\nCOMMAND " + json.dumps(command) + "\n")
        stream.flush()
        started = time.perf_counter()
        process = subprocess.run(command, cwd=cwd, env=env or BASE_ENV,
                                 stdout=stream, stderr=subprocess.STDOUT, text=True)
        stream.write(f"EXIT {process.returncode}; elapsed={time.perf_counter()-started:.3f}s\n")
    if process.returncode:
        tail = Path(log).read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"Command failed ({process.returncode}); see {log}\n{tail}")

def setup_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

global_log = SETUP_LOGS / "bootstrap.log"
UV = [sys.executable, "-m", "uv"]
BOOTSTRAP_ERROR = None
try:
    if not globals().get("HARDWARE_READY", True):
        raise RuntimeError("Dual-GPU hardware preflight failed; installation and downloads were skipped.")
    if importlib.util.find_spec("uv") is None:
        setup_command([sys.executable, "-m", "pip", "install", "uv==0.6.17"], global_log)
    setup_command(UV + ["python", "install", "3.11"], global_log)
except Exception as exc:
    BOOTSTRAP_ERROR = str(exc)
    print("Python runtime bootstrap FAILED:", BOOTSTRAP_ERROR)

# Kaggle normally already supplies these. espeak-ng is required by StyleTTS2
# and is also the English fallback in Kokoro; the other packages support I/O.
missing_commands = [name for name in ("git", "ffmpeg", "espeak-ng")
                    if shutil.which(name) is None]
SYSTEM_STATUS = {"missing_before_setup": missing_commands}
if missing_commands and not globals().get("HARDWARE_READY", True):
    SYSTEM_STATUS.update(status="skipped", error="Hardware preflight failed; no system packages installed.")
elif missing_commands:
    try:
        apt = ["apt-get"] if os.geteuid() == 0 else ["sudo", "apt-get"]
        setup_command(apt + ["update", "-qq"], global_log)
        setup_command(apt + ["install", "-y", "-qq", "git", "ffmpeg",
                             "espeak-ng", "libespeak-ng1", "libsndfile1"], global_log)
        SYSTEM_STATUS["status"] = "ready"
    except Exception as exc:
        SYSTEM_STATUS.update(status="failed", error=str(exc))
else:
    SYSTEM_STATUS["status"] = "ready"

SOURCES = {
    "kokoro": ("https://github.com/hexgrad/kokoro.git", "dfb907a02bba8152ca444717ca5d78747ccb4bec"),
    "f5": ("https://github.com/SWivid/F5-TTS.git", "283252563dbf91be625e0c27926acfaac449186c"),
    "emotivoice": ("https://github.com/netease-youdao/EmotiVoice.git", "59f0f36de4db12825f4705dd4e0780d79dd6bb01"),
    "styletts2": ("https://github.com/yl4579/StyleTTS2.git", "5cedc71c333f8d8b8551ca59378bdcc7af4c9529"),
    "omnivoice": ("https://github.com/k2-fsa/OmniVoice.git", "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"),
    "supertonic": ("https://github.com/supertone-oss-archive/supertonic.git", "1e9799e964ea4c0dad7cde993b65c3c813a7b373"),
}
COMMON = ["numpy==1.26.4", "scipy==1.14.1", "soundfile==0.13.1", "tqdm==4.67.1",
          "nvidia-ml-py==12.570.86"]
DEPENDENCIES = {
    "kokoro": ["transformers==5.3.0", "misaki[en]==0.9.4", "loguru==0.7.3",
               "spacy==3.8.7", "thinc==8.3.4", "spacy-curated-transformers==0.3.0",
               "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"],
    "f5": ["vocos==0.1.0", "pypinyin==0.54.0", "rjieba==0.1.1", "librosa==0.10.2.post1",
           "einops==0.8.1", "x_transformers==2.16.2", "torchdiffeq==0.2.5",
           "safetensors==0.5.3", "huggingface_hub==0.29.3", "wandb==0.19.8",
           "accelerate==1.6.0", "ema_pytorch==0.7.7", "datasets==3.5.0"],
    "emotivoice": ["transformers==4.26.1", "numba==0.61.2", "librosa==0.10.2.post1",
                   "yacs==0.1.8", "g2p_en==2.1.0", "jieba==0.42.1", "pypinyin==0.54.0",
                   "pypinyin_dict==0.9.0", "cn2an==0.5.23", "nltk==3.9.1",
                   "PyYAML==6.0.2", "huggingface_hub==0.29.3"],
    "styletts2": ["PyYAML==6.0.2", "munch==4.0.0", "transformers==4.44.2",
                  "einops==0.8.1", "einops-exts==0.0.4", "librosa==0.10.2.post1",
                  "phonemizer==3.3.0", "nltk==3.9.1", "matplotlib==3.10.1",
                  "huggingface_hub==0.29.3"],
    "omnivoice": ["transformers==5.3.0", "accelerate==1.12.0", "pydub==0.25.1",
                  "librosa==0.10.2.post1", "num2words==0.5.14", "WeTextProcessing==1.0.3"],
    "supertonic": ["onnxruntime-gpu[cuda,cudnn]==1.23.2", "huggingface_hub==0.29.3"],
}

def checkout_model(model, log):
    url, commit = SOURCES[model]
    destination = REPOS / model
    if not (destination / ".git").exists():
        setup_command(["git", "init", str(destination)], log)
        setup_command(["git", "-C", str(destination), "remote", "add", "origin", url], log)
    setup_command(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", commit], log)
    setup_command(["git", "-C", str(destination), "checkout", "--detach", commit], log)
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"],
                                     text=True, env=BASE_ENV).strip()
    if actual != commit:
        raise RuntimeError(f"Source revision mismatch: {actual} != {commit}")
    return destination

def download_hf(python, model, log):
    destination = ASSETS / model
    destination.mkdir(parents=True, exist_ok=True)
    # Download before concurrent inference, so benchmark timing excludes network
    # and workers do not race on partial weight files.
    specifications = {
        "kokoro": [("hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987",
                    ["config.json", "kokoro-v1_0.pth", "voices/*.pt"], ".")],
        "f5": [("SWivid/F5-TTS", "84e5a410d9cead4de2f847e7c9369a6440bdfaca",
                ["F5TTS_v1_Base/model_1250000.safetensors", "F5TTS_v1_Base/vocab.txt"], "."),
               ("charactr/vocos-mel-24khz", "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21",
                ["config.yaml", "pytorch_model.bin"], "vocos")],
        "emotivoice": [("WangZeJun/simbert-base-chinese", "8096d1f9a7f7c2d9dbd7974511c4bc4a1755932b",
                        ["*.json", "*.txt", "*.bin", "*.safetensors"], "simbert")],
        "styletts2": [("yl4579/StyleTTS2-LibriTTS", "3aa7ba7f8f275ec13dce21682a61494c35089e2a",
                      ["Models/LibriTTS/config.yml", "Models/LibriTTS/epochs_2nd_00020.pth"], ".")],
        "omnivoice": [("k2-fsa/OmniVoice", "c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
                      ["*.json", "*.safetensors", "*.model", "*.txt", "audio_tokenizer/**"], ".")],
        "supertonic": [("supertone-oss-archive/supertonic-3", "aafc6e32416a594460b32413efc49d7fe4ce6d46",
                       ["onnx/**", "voice_styles/**"], ".")],
    }
    script = """import json, sys
from huggingface_hub import snapshot_download
for repo, revision, patterns, subdir in json.loads(sys.argv[1]):
    snapshot_download(repo_id=repo, revision=revision, allow_patterns=patterns,
                      local_dir=str(__import__('pathlib').Path(sys.argv[2])/subdir), max_workers=1)
"""
    setup_command([python, "-c", script, json.dumps(specifications[model]), str(destination)], log)
    return destination

def download_emoti_checkpoints(log):
    # Git stores signed SHA-256 LFS pointers. Download only the two required
    # inference binaries, and validate bytes against their pinned Git pointers.
    commit = "d1332ad9fee7020df5f358fdbe8d257dbf0d1a7c"
    destination = ASSETS / "emotivoice" / "outputs"
    if not (destination / ".git").is_dir():
        setup_command(["git", "init", str(destination)], log)
        setup_command(["git", "-C", str(destination), "remote", "add", "origin",
                       "https://www.modelscope.cn/syq163/outputs.git"], log)
    setup_command(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", commit], log)
    paths = ("prompt_tts_open_source_joint/ckpt/g_00140000", "style_encoder/ckpt/checkpoint_163431")
    records = []
    for relative in paths:
        pointer = subprocess.check_output(["git", "-C", str(destination), "show", f"{commit}:{relative}"],
                                          env=BASE_ENV, text=True)
        oid = re.search(r"^oid sha256:([0-9a-f]{64})$", pointer, re.M)
        size = re.search(r"^size (\d+)$", pointer, re.M)
        if not oid or not size:
            raise RuntimeError(f"Expected an authenticated Git LFS pointer for {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_size, expected_sha = int(size.group(1)), oid.group(1)
        def valid_file(path):
            if not path.is_file() or path.stat().st_size != expected_size:
                return False
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest() == expected_sha
        if not valid_file(target):
            query = urllib.parse.urlencode({"Revision": commit, "FilePath": relative})
            url = "https://www.modelscope.cn/api/v1/models/syq163/outputs/repo?" + query
            temporary = target.with_name(target.name + ".part")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
            if not valid_file(temporary):
                raise RuntimeError(f"Checkpoint SHA-256/size mismatch for {relative}; no inference will run.")
            temporary.replace(target)
        records.append({"file": str(target), "revision": commit,
                        "sha256": expected_sha, "size": expected_size})
    setup_json(ROOT / "emotivoice_checkpoint_provenance.json", records)
    return {"generator_checkpoint": str(destination / paths[0]),
            "style_checkpoint": str(destination / paths[1])}

def resources_for(model, repo, assets):
    resources = {"repo": str(repo)}
    if model == "kokoro":
        resources = {"checkpoint": str(assets / "kokoro-v1_0.pth"),
                     "model_config": str(assets / "config.json"),
                     "cache_dir": str(HF_CACHE / "hub")}
    elif model == "f5":
        resources.update(checkpoint=str(assets / "F5TTS_v1_Base/model_1250000.safetensors"),
                         vocab=str(assets / "F5TTS_v1_Base/vocab.txt"), vocos_dir=str(assets / "vocos"))
    elif model == "emotivoice":
        resources.update(bert_path=str(assets / "simbert"), local_files_only=True)
    elif model == "styletts2":
        resources.update(checkpoint=str(assets / "Models/LibriTTS/epochs_2nd_00020.pth"),
                         model_config=str(assets / "Models/LibriTTS/config.yml"))
    elif model == "omnivoice":
        resources = {"local_model": str(assets), "local_files_only": True}
    elif model == "supertonic":
        resources.update(assets_dir=str(assets))
    return resources

requested = CONFIG.get("install_models", list(MODELS))
if not isinstance(requested, list) or any(model not in MODELS for model in requested):
    raise ValueError("CONFIG['install_models'] must contain known model names.")
SETUP_RESULTS = {}
report_path = ROOT / "setup_report.json"
for model in MODELS:
    if model not in requested:
        SETUP_RESULTS[model] = {"status": "not_requested", "error": "Disabled in install_models"}
        continue
    log = SETUP_LOGS / f"{model}.log"
    python = ENV_DIR / model / "bin" / "python"
    entry = {"status": "installing", "python": str(python), "resources": {}, "log": str(log)}
    SETUP_RESULTS[model] = entry
    started = time.perf_counter()
    print(f"[{model}] installing isolated Python, dependencies, source, and weights...")
    try:
        if BOOTSTRAP_ERROR:
            raise RuntimeError("Python runtime bootstrap failed: " + BOOTSTRAP_ERROR)
        if model in ("styletts2", "kokoro") and shutil.which("espeak-ng") is None:
            raise RuntimeError("espeak-ng is unavailable. See bootstrap.log for the system dependency failure.")
        if shutil.which("git") is None:
            raise RuntimeError("git is unavailable; pinned source verification cannot proceed.")
        if not python.is_file():
            setup_command(UV + ["venv", "--python", "3.11", "--seed", str(ENV_DIR / model)], log)
        # Hardlink packages shared between isolated environments. All runtime
        # trees are under /tmp on one filesystem; zip contains outputs only.
        torch_version = "2.8.0" if model in ("omnivoice", "supertonic") else "2.5.1"
        variant = "cu128" if model in ("omnivoice", "supertonic") else "cu121"
        override = CONFIG.get("torch_cuda_variant")
        if isinstance(override, dict):
            variant = override.get(model, variant)
        elif override not in (None, "auto", ""):
            variant = override
        supported = {"2.5.1": {"cu118", "cu121", "cu124"}, "2.8.0": {"cu126", "cu128", "cu129"}}
        if variant not in supported[torch_version]:
            raise ValueError(f"Torch {torch_version} has no supported wheel variant {variant!r} in this setup.")
        if model == "supertonic" and variant != "cu128":
            raise ValueError("Supertonic ORT1.23.2 requires the configured CUDA12.8/cuDNN9 runtime; retain cu128.")
        if model == "supertonic" and CONFIG.get("strict_cuda_driver", False):
            versions = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, env=BASE_ENV).strip().splitlines()
            if not versions or any(tuple(map(int, version.strip().split("."))) < (570, 26)
                                   for version in versions):
                raise RuntimeError("Strict driver policy requires NVIDIA Linux driver>=570.26 for CUDA12.8.")
        constraints = RUNTIME / f"{model}_constraints.txt"
        constraints.write_text(f"torch=={torch_version}\ntorchaudio=={torch_version}\n"
                               "numpy==1.26.4\nscipy==1.14.1\n", encoding="utf-8")
        install = UV + ["pip", "install", "--python", str(python), "--link-mode", "hardlink"]
        reinstall = ["--reinstall"] if CONFIG.get("force_setup", False) else []
        setup_command(install + reinstall + [f"torch=={torch_version}", f"torchaudio=={torch_version}",
                                 "--index-url", f"https://download.pytorch.org/whl/{variant}"], log)
        setup_command(install + ["--constraint", str(constraints)] + COMMON + DEPENDENCIES[model], log)
        repo = checkout_model(model, log)
        if model in ("kokoro", "omnivoice"):
            url, commit = SOURCES[model]
            setup_command(install + reinstall + ["--no-deps", f"git+{url}@{commit}"], log)
        if model == "emotivoice":
            setup_command([python, "-m", "nltk.downloader", "averaged_perceptron_tagger_eng",
                           "averaged_perceptron_tagger", "cmudict", "punkt", "punkt_tab"], log)
        assets = download_hf(python, model, log)
        resources = resources_for(model, repo, assets)
        if model == "emotivoice":
            resources.update(download_emoti_checkpoints(log))
        entry.update(status="ready", resources=resources, source_revision=SOURCES[model][1],
                     torch_version=torch_version, torch_cuda_variant=variant,
                     elapsed_seconds=time.perf_counter()-started)
        # Exact installed package versions are useful when reproducing failures.
        with (SETUP_LOGS / f"{model}_packages.txt").open("w", encoding="utf-8") as stream:
            subprocess.run(UV + ["pip", "freeze", "--python", str(python)], env=BASE_ENV,
                           stdout=stream, stderr=subprocess.STDOUT, check=True, text=True)
        print(f"[{model}] ready ({entry['elapsed_seconds']:.1f}s); details: {log}")
    except Exception as exc:
        entry.update(status="failed", error=str(exc), elapsed_seconds=time.perf_counter()-started)
        print(f"[{model}] FAILED: {exc}")
    setup_json(report_path, {"models": SETUP_RESULTS, "system": SYSTEM_STATUS,
                             "runtime_directory": str(RUNTIME), "cuda_allocations_during_setup": False})
print("Setup statuses:", {model: item["status"] for model, item in SETUP_RESULTS.items()})
''').strip() + "\n"


PREFLIGHT_CODE = dedent(r'''
# CPU-only import/API checks. CUDA context/session initialization is performed
# later by monitored benchmark jobs, not hidden in dependency setup.
PREFLIGHT_RESULTS = {}
API_CHECKS = {
    "kokoro": "from kokoro import KModel, KPipeline; from misaki import en; en.G2P(trf=False, british=False)",
    "f5": "from f5_tts.model import CFM, DiT; from f5_tts.model.utils import get_tokenizer; from vocos import Vocos",
    "emotivoice": "from frontend_en import G2p; from frontend import g2p_cn_en; from models.prompt_tts_modified.jets import JETSGenerator; from models.prompt_tts_modified.simbert import StyleEncoder; G2p()",
    "styletts2": "from models import build_model; from Utils.PLBERT.util import CustomAlbert; from Modules.diffusion.sampler import DiffusionSampler; from phonemizer.backend import EspeakBackend; EspeakBackend('en-us')",
    "omnivoice": "from omnivoice import OmniVoice; from transformers import HiggsAudioV2TokenizerModel",
    "supertonic": "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()",
}
SCRIPT_NAMES = {"f5": "f5_samples.py", "styletts2": "styletts2_samples.py",
                "emotivoice": "emotivoice_samples.py", "kokoro": "kokoro_samples.py",
                "omnivoice": "omnivoice_samples.py", "supertonic": "supertonic_samples.py"}
for model, item in SETUP_RESULTS.items():
    if item["status"] != "ready":
        PREFLIGHT_RESULTS[model] = {"status": "unavailable", "error": item.get("error")}
        continue
    log = SETUP_LOGS / f"{model}_preflight.log"
    try:
        repo = REPOS / model
        sys_path = str(repo / "src") if model == "f5" else str(repo)
        script = ("import os,sys,torch; assert os.environ.get('CUDA_VISIBLE_DEVICES')==''; "
                  f"sys.path.insert(0,{sys_path!r}); os.chdir({str(repo)!r}); " + API_CHECKS[model] +
                  "; print('API_IMPORTS_OK'); print('Torch:',torch.__version__,'CUDA runtime:',torch.version.cuda)")
        setup_command([item["python"], "-c", script], log)
        setup_command([item["python"], str(SCRIPTS_DIR / SCRIPT_NAMES[model]), "--help"], log)
        PREFLIGHT_RESULTS[model] = {"status": "passed", "log": str(log)}
        print(f"[{model}] CPU API preflight passed")
    except Exception as exc:
        PREFLIGHT_RESULTS[model] = {"status": "failed", "error": str(exc), "log": str(log)}
        item.update(status="failed", error="API preflight failed: " + str(exc))
        print(f"[{model}] API preflight FAILED: {exc}")
setup_json(ROOT / "preflight_report.json", PREFLIGHT_RESULTS)
setup_json(ROOT / "setup_report.json", {"models": SETUP_RESULTS, "system": SYSTEM_STATUS,
           "runtime_directory": str(RUNTIME), "cuda_allocations_during_setup": False})
''').strip() + "\n"


BOOTSTRAP_REFERENCE_CODE = dedent(r'''
# F5/StyleTTS2 need reference audio. Supply your own exact transcript for voice
# cloning, or generate this disclosed synthetic reference with Kokoro on CPU.
REFERENCE_AUDIO = CONFIG.get("reference_audio")
REFERENCE_TEXT = CONFIG.get("reference_text")
if CONFIG.get("speaker_mode") == "default":
    REFERENCE_AUDIO = REFERENCE_TEXT = None
REFERENCE_STATUS = {"status": "pending"}
if REFERENCE_AUDIO:
    try:
        REFERENCE_AUDIO = str(Path(REFERENCE_AUDIO).expanduser().resolve())
        if not Path(REFERENCE_AUDIO).is_file():
            raise ValueError("Set an existing reference_audio WAV.")
        if REFERENCE_TEXT is not None and (not isinstance(REFERENCE_TEXT, str) or not REFERENCE_TEXT.strip()):
            raise ValueError("reference_text must be the exact nonempty transcript or None.")
        checker_python = next((item["python"] for item in SETUP_RESULTS.values()
                               if item.get("status") == "ready"), sys.executable)
        checker = """import json,sys,numpy as np,soundfile as sf
wave,rate=sf.read(sys.argv[1],dtype='float32',always_2d=True)
assert wave.shape[1]==1, 'Reference must be mono.'
duration=wave.shape[0]/rate
assert 1<=duration<=8, f'Reference duration {duration:.3f}s outside 1-8s.'
assert np.isfinite(wave).all(), 'Reference contains nonfinite samples.'
assert np.max(np.abs(wave))>1e-5, 'Reference is silent.'
print(json.dumps({'duration_seconds':duration,'sample_rate':rate,'channels':1}))
"""
        checked = subprocess.run([checker_python, "-c", checker, REFERENCE_AUDIO], env=BASE_ENV,
                                 text=True, capture_output=True, check=True)
        facts = json.loads(checked.stdout.strip())
        REFERENCE_STATUS.update(status="provided", path=REFERENCE_AUDIO, transcript=REFERENCE_TEXT, **facts)
    except Exception as exc:
        REFERENCE_AUDIO = None
        details = (getattr(exc, "stderr", None) or str(exc)).strip()
        REFERENCE_STATUS.update(status="failed", error=details)
        print("Reference validation FAILED:", details)
elif CONFIG.get("speaker_mode") == "reference":
    REFERENCE_STATUS.update(status="failed", error="speaker_mode='reference' requires your reference_audio WAV; add its transcript for F5/OmniVoice.")
    print(REFERENCE_STATUS["error"])
else:
    REFERENCE_TEXT = "The morning sun brings gentle warmth to the quiet garden."
    item = SETUP_RESULTS.get("kokoro", {})
    if item.get("status") != "ready":
        REFERENCE_STATUS.update(status="failed", error="Kokoro unavailable; set reference_audio and reference_text.")
        print(REFERENCE_STATUS["error"])
    else:
        try:
            ref_root = ROOT / "reference"
            ref_root.mkdir(parents=True, exist_ok=True)
            parameters = {**item["resources"], "text": REFERENCE_TEXT,
                          "voice": str(ASSETS / "kokoro" / "voices" / "am_fenrir.pt"),
                          "language": "a", "device": "cpu", "n": 1, "seed": 42, "speed": 1.0,
                          "sample_rate": 24000, "output_dir": str(ref_root / "kokoro_runs")}
            settings = ref_root / "bootstrap_config.json"
            setup_json(settings, parameters)
            setup_command([item["python"], str(SCRIPTS_DIR / "kokoro_samples.py"),
                           "--config", str(settings)], SETUP_LOGS / "reference_bootstrap.log")
            candidates = sorted((ref_root / "kokoro_runs").glob("**/*.wav"), key=lambda path: path.stat().st_mtime)
            if not candidates:
                raise RuntimeError("Kokoro created no reference WAV.")
            reference = ref_root / "reference_kokoro_am_fenrir.wav"
            shutil.copy2(candidates[-1], reference)
            # WAV PCM frame/rate header gives total duration without loading a
            # model or removing content-dependent silence.
            import wave
            with wave.open(str(reference), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            if not 1 <= duration <= 8:
                raise RuntimeError(f"Synthetic reference duration {duration:.3f}s outside1–8s; supply a clean reference.")
            REFERENCE_AUDIO = str(reference)
            REFERENCE_STATUS.update(status="synthetic", path=REFERENCE_AUDIO,
                                    transcript=REFERENCE_TEXT, duration_seconds=duration,
                                    generator="Kokoro am_fenrir, CPU; complete untrimmed male waveform")
            print(f"Synthetic reference ready: {REFERENCE_AUDIO} ({duration:.3f}s)")
        except Exception as exc:
            REFERENCE_AUDIO = None
            REFERENCE_STATUS.update(status="failed", error=str(exc))
            print(f"Reference bootstrap FAILED: {exc}")
setup_json(ROOT / "reference_report.json", REFERENCE_STATUS)
''').strip() + "\n"


def validate_cell_syntax() -> None:
    """Validate the exact embedded cells without installation or network I/O."""
    for name, source in (("setup", SETUP_CODE), ("preflight", PREFLIGHT_CODE),
                         ("reference", BOOTSTRAP_REFERENCE_CODE)):
        compile(source, f"<kaggle_{name}_cell>", "exec")


def validate_setup_contract() -> None:
    """Exercise setup/report/reference behavior with no installers or network."""
    import contextlib
    import hashlib
    import io
    import json
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

    def sandbox_source(runtime: Path) -> str:
        return SETUP_CODE.replace(
            'RUNTIME = Path("/tmp/tts_benchmark_runtime")', f"RUNTIME = Path({str(runtime)!r})"
        )

    with tempfile.TemporaryDirectory(prefix="tts_notebook_setup_test_") as directory:
        base = Path(directory)
        disabled = {
            "ROOT": base / "disabled", "SCRIPTS_DIR": base / "scripts",
            "CONFIG": {"install_models": list(("kokoro", "f5", "emotivoice", "styletts2",
                                                "omnivoice", "supertonic")),
                       "torch_cuda_variant": "auto"}, "HARDWARE_READY": False,
        }
        with patch("subprocess.run", side_effect=AssertionError("No installation may run")), \
                patch("subprocess.check_output", side_effect=AssertionError("No subprocess may run")), \
                patch("shutil.which", return_value="/mock/bin/command"), \
                contextlib.redirect_stdout(io.StringIO()):
            exec(compile(sandbox_source(base / "runtime_disabled"), "<mock_setup>", "exec"), disabled)
        assert len(disabled["SETUP_RESULTS"]) == 6
        assert all(item["status"] == "failed" for item in disabled["SETUP_RESULTS"].values())
        assert (disabled["ROOT"] / "setup_report.json").is_file()

        runtime = base / "runtime_success"
        outputs = runtime / "assets" / "emotivoice" / "outputs"
        (outputs / ".git").mkdir(parents=True)
        payload = b"Mock checkpoint bytes for a no-network contract test."
        for name in ("prompt_tts_open_source_joint/ckpt/g_00140000",
                     "style_encoder/ckpt/checkpoint_163431"):
            path = outputs / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        pointer = ("version https://git-lfs.github.com/spec/v1\n"
                   f"oid sha256:{hashlib.sha256(payload).hexdigest()}\nsize {len(payload)}\n")
        successful = {
            "ROOT": base / "success", "SCRIPTS_DIR": base / "scripts",
            "CONFIG": {"install_models": list(disabled["SETUP_RESULTS"]), "torch_cuda_variant": "auto"},
            "HARDWARE_READY": True,
        }

        def fake_output(command, **_kwargs):
            if "show" in command:
                return pointer
            if "rev-parse" in command:
                return successful["SOURCES"][Path(command[2]).name][1] + "\n"
            raise AssertionError(f"Unexpected subprocess read: {command}")

        commands = []

        def fake_run(command, **kwargs):
            assert kwargs.get("env", {}).get("CUDA_VISIBLE_DEVICES") == ""
            commands.append(command)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run), \
                patch("subprocess.check_output", side_effect=fake_output), \
                patch("shutil.which", return_value="/mock/bin/command"), \
                contextlib.redirect_stdout(io.StringIO()):
            exec(compile(sandbox_source(runtime), "<mock_setup>", "exec"), successful)
        assert all(item["status"] == "ready" for item in successful["SETUP_RESULTS"].values())
        assert successful["SETUP_RESULTS"]["kokoro"]["torch_cuda_variant"] == "cu121"
        assert successful["SETUP_RESULTS"]["supertonic"]["torch_cuda_variant"] == "cu128"
        assert successful["SETUP_RESULTS"]["omnivoice"]["resources"]["local_files_only"] is True
        assert any("--link-mode" in command and "hardlink" in command for command in commands)
        report = json.loads((successful["ROOT"] / "setup_report.json").read_text(encoding="utf-8"))
        assert report["cuda_allocations_during_setup"] is False

        # Invalid supplied references must become reportable failures, so later
        # audio/report/ZIP cells remain usable after a partial benchmark failure.
        successful["CONFIG"].update(reference_audio=str(base / "absent.wav"), reference_text="Exact text.")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(BOOTSTRAP_REFERENCE_CODE, "<mock_reference>", "exec"), successful)
        assert successful["REFERENCE_AUDIO"] is None
        assert successful["REFERENCE_STATUS"]["status"] == "failed"
        assert (successful["ROOT"] / "reference_report.json").is_file()

        provided = base / "provided_reference.wav"
        provided.write_bytes(b"Reference decoding is mocked in this contract test.")
        successful["CONFIG"].update(reference_audio=str(provided), reference_text="Exact text.")
        checked_audio = SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
            {"duration_seconds": 3.25, "sample_rate": 24000, "channels": 1}))
        with patch("subprocess.run", return_value=checked_audio) as checked, \
                contextlib.redirect_stdout(io.StringIO()):
            exec(compile(BOOTSTRAP_REFERENCE_CODE, "<mock_reference>", "exec"), successful)
        assert successful["REFERENCE_STATUS"]["status"] == "provided"
        assert successful["REFERENCE_STATUS"]["duration_seconds"] == 3.25
        assert checked.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""

        strict = {
            "ROOT": base / "strict", "SCRIPTS_DIR": base / "scripts", "HARDWARE_READY": True,
            "CONFIG": {"install_models": ["supertonic"], "torch_cuda_variant": "auto",
                       "strict_cuda_driver": True},
        }
        with patch("subprocess.run", side_effect=fake_run), \
                patch("subprocess.check_output", return_value="550.54.14\n550.54.14\n"), \
                patch("shutil.which", return_value="/mock/bin/command"), \
                contextlib.redirect_stdout(io.StringIO()):
            exec(compile(sandbox_source(base / "runtime_strict"), "<mock_setup>", "exec"), strict)
        assert strict["SETUP_RESULTS"]["supertonic"]["status"] == "failed"
        assert "570.26" in strict["SETUP_RESULTS"]["supertonic"]["error"]


if __name__ == "__main__":
    validate_cell_syntax()
    validate_setup_contract()
    print("Kaggle setup cells compile; mocked setup/reference contract checks pass.")

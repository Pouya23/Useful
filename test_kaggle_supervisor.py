"""Focused CPU tests for the delivered notebook's orchestration and reports.

These inspect the authoring cells without executing installers, touching CUDA,
or pretending that synthetic fixtures validate pretrained speech synthesis.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import html
import json
import math
import os
import re
import runpy
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import soundfile as sf

SOURCE = Path(__file__).with_name("kaggle_benchmark_cells.py")


class Progress:
    def __init__(self, iterable=None, **_):
        self.iterable = iterable
        self.count = 0

    def update(self, count=1):
        self.count += count

    def close(self):
        pass

    def __iter__(self):
        return iter(self.iterable)


def extract_cell(code, names=None):
    """Load definitions and selected constants, never notebook tail execution."""
    tree = ast.parse(code)
    statements = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            statements.append(node)
        elif isinstance(node, ast.Assign) and names:
            if any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
                statements.append(node)
    return compile(ast.Module(body=statements, type_ignores=[]), str(SOURCE), "exec")


@pytest.fixture
def env(tmp_path):
    cells = runpy.run_path(str(SOURCE))
    root = tmp_path / "benchmark"
    run = root / "runs" / "test-run"
    package = run / "download"
    scripts = root / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir()
    config = {
        "install_models": ["kokoro", "f5"], "n": 2, "seed": 100,
        "sample_rate": 48000, "pause_ms": 100, "max_vram_gb": 5,
        "cpu_threads_per_job": 2, "nvml_interval_ms": 10,
        "enforce_observed_budget": True, "budget_scope": "device_total",
        "job_timeout_seconds": 10, "gpu_release_timeout_seconds": 1,
        "include_reference_in_zip": False,
    }
    namespace = {
        "Path": Path, "csv": csv, "hashlib": hashlib, "html": html,
        "json": json, "math": math, "os": os, "re": re, "shutil": shutil,
        "signal": signal, "subprocess": subprocess, "sys": sys,
        "threading": threading, "time": time, "zipfile": zipfile, "deque": deque,
        "sf": sf, "ROOT": root, "RUN_DIR": run, "RUN_ID": "test-run",
        "PACKAGE_DIR": package, "SCRIPTS_DIR": scripts, "CONFIG": config,
        "JOB_RESULTS": [], "SAMPLE_RESULTS": [], "TIMELINE": [],
        "TEXT_CASES": [{"name": "short", "text": "one two three four five six seven eight nine ten"}],
        "MODEL_PROFILES": {
            "kokoro": [{"name": "base", "parameters": {"voice": "af_heart"}}],
            "f5": [{"name": "clone", "parameters": {"steps": 16}}],
        },
        "SETUP_RESULTS": {
            "kokoro": {"status": "ready", "python": "python", "resources": {}},
            "f5": {"status": "ready", "python": "python", "resources": {"repo": root / "f5", "local_files_only": True}},
        },
        "REFERENCE_AUDIO": root / "reference.wav", "REFERENCE_TEXT": "Reference transcript.",
        "HARDWARE_READY": True, "hardware_error": None,
        "GPU_INVENTORY": [{"uuid": "GPU-A"}, {"uuid": "GPU-B"}],
        "GPU_HANDLES": {0: "A", 1: "B"}, "tqdm": Progress,
        "HTML": lambda value: value,
        "display": lambda *args, **kwargs: SimpleNamespace(update=lambda *_: None),
    }
    exec(extract_cell(cells["KERNEL_CODE"]), namespace)
    exec(extract_cell(cells["SUPERVISOR_CODE"], {"SCRIPT_NAMES"}), namespace)
    exec(extract_cell(cells["REPORT_CODE"], {"PLANNER_SCOPE"}), namespace)
    return namespace


def make_job(env, model="kokoro", index=0, status="completed", n=1):
    directory = env["RUN_DIR"] / "jobs" / f"{model}-{index}"
    (directory / "raw").mkdir(parents=True)
    return {
        "job_id": f"{model}-{index}", "model": model, "profile": "base",
        "case": "ten", "word_count": 10, "n": n, "seed": 100,
        "status": status, "settings": {"sample_rate": 48000},
        "job_dir": str(directory), "blocked_reason": None,
        "gpu_slot": index % 2, "gpu_uuid": f"GPU-{index % 2}",
        "exit_code": 0,
    }


def test_distinct_model_preference(env):
    choose = env["preferred_job_index"]
    pending = [{"model": "kokoro"}, {"model": "kokoro"}, {"model": "f5"}]
    assert choose(pending, {"kokoro"}) == 2
    assert choose(pending, {"kokoro", "f5"}) == 0
    assert choose([], set()) is None


def test_profile_jobs_identity_reference_and_native_types(env):
    env["TEXT_CASES"] *= 2
    jobs = env["profile_jobs"]()
    assert len(jobs) == 4
    assert len({job["job_id"] for job in jobs}) == 4
    assert all(job["settings"]["device"] == "cuda:0" for job in jobs)
    f5 = next(job for job in jobs if job["model"] == "f5")
    assert f5["settings"]["local_files_only"] is True
    assert f5["settings"]["repo"] == str(env["ROOT"] / "f5")
    assert f5["settings"]["reference_text"] == env["REFERENCE_TEXT"]
    assert json.loads(Path(f5["config_path"]).read_text())["local_files_only"] is True


def add_omni_profile(env):
    env["CONFIG"]["install_models"].append("omnivoice")
    env["SETUP_RESULTS"]["omnivoice"] = {"status": "ready", "python": "python", "resources": {}}
    env["MODEL_PROFILES"]["omnivoice"] = [{"name": "male_or_reference", "reference": "auto",
        "parameters": {"instruct": "male, low pitch", "dtype": "float16"}}]


def test_user_reference_automatically_selects_omni_clone_without_conflicting_design(env):
    add_omni_profile(env)
    env["CONFIG"].update(speaker_mode="auto", reference_audio=str(env["REFERENCE_AUDIO"]))
    jobs = env["profile_jobs"]()
    omni = next(job for job in jobs if job["model"] == "omnivoice")
    assert omni["settings"]["ref_audio"] == str(env["REFERENCE_AUDIO"])
    assert omni["settings"]["ref_text"] == env["REFERENCE_TEXT"]
    assert "instruct" not in omni["settings"]
    assert omni["speaker_source"] == "user_reference"
    assert next(job for job in jobs if job["model"] == "kokoro")["speaker_source"] == "preset"


def test_default_voice_mode_uses_male_design_even_if_user_path_is_configured(env):
    add_omni_profile(env)
    env["CONFIG"].update(speaker_mode="default", reference_audio="user.wav")
    jobs = env["profile_jobs"]()
    omni = next(job for job in jobs if job["model"] == "omnivoice")
    assert omni["settings"]["instruct"] == "male, low pitch"
    assert "ref_audio" not in omni["settings"]
    assert omni["speaker_source"] == "voice_design"
    assert next(job for job in jobs if job["model"] == "f5")["speaker_source"] == "synthetic_male_reference"


def test_reference_only_mode_reports_preset_models_as_unsupported(env):
    add_omni_profile(env)
    env["CONFIG"].update(speaker_mode="reference", reference_audio="user.wav")
    jobs = env["profile_jobs"]()
    preset = next(job for job in jobs if job["model"] == "kokoro")
    assert preset["blocked_status"] == "reference_unsupported"
    assert "preset speakers" in preset["blocked_reason"]
    assert next(job for job in jobs if job["model"] == "omnivoice")["blocked_reason"] is None


def test_reference_without_transcript_still_allows_style_but_blocks_f5_and_omni(env):
    add_omni_profile(env)
    env["CONFIG"]["install_models"].append("styletts2")
    env["SETUP_RESULTS"]["styletts2"] = {"status": "ready", "python": "python", "resources": {}}
    env["MODEL_PROFILES"]["styletts2"] = [{"name": "reference", "parameters": {}}]
    env["CONFIG"]["reference_audio"] = "user.wav"
    env["REFERENCE_TEXT"] = None
    jobs = env["profile_jobs"]()
    assert next(job for job in jobs if job["model"] == "styletts2")["blocked_reason"] is None
    for model in ("f5", "omnivoice"):
        assert "transcript" in next(job for job in jobs if job["model"] == model)["blocked_reason"]


def test_clone_parameters_cannot_override_supervisor_identity(env):
    add_omni_profile(env)
    env["CONFIG"]["speaker_mode"] = "reference"
    env["MODEL_PROFILES"]["omnivoice"][0]["clone_parameters"] = {"n": 9}
    with pytest.raises(ValueError, match="supervisor-controlled"):
        env["profile_jobs"]()


def test_per_profile_omni_reference_is_respected_without_global_reference(env):
    add_omni_profile(env)
    parameters = env["MODEL_PROFILES"]["omnivoice"][0]["parameters"]
    parameters.update(ref_audio="custom.wav", ref_text="Custom exact transcript.")
    omni = next(job for job in env["profile_jobs"]() if job["model"] == "omnivoice")
    assert omni["settings"]["ref_audio"] == "custom.wav"
    assert omni["settings"]["ref_text"] == "Custom exact transcript."
    assert "instruct" not in omni["settings"]
    assert omni["speaker_source"] == "user_reference"


def test_cached_omni_reference_does_not_require_global_wav_or_transcript(env):
    add_omni_profile(env)
    cache = env["ROOT"] / "prepared_prompt.pt"
    cache.write_bytes(b"test path sentinel; model deserialization is tested separately")
    env["MODEL_PROFILES"]["omnivoice"][0]["parameters"]["prompt_cache"] = str(cache)
    env["REFERENCE_AUDIO"] = env["REFERENCE_TEXT"] = None
    omni = next(job for job in env["profile_jobs"]() if job["model"] == "omnivoice")
    assert omni["blocked_reason"] is None
    assert omni["speaker_source"] == "cached_reference_prompt"
    assert "ref_audio" not in omni["settings"]
    assert "instruct" not in omni["settings"]


def test_invalid_speaker_mode_fails_before_creating_job_folders(env):
    env["CONFIG"]["speaker_mode"] = "random"
    with pytest.raises(ValueError, match="speaker_mode"):
        env["profile_jobs"]()
    assert not (env["RUN_DIR"] / "jobs").exists()


def test_f5_environment_covers_eager_imports_of_actual_pinned_source():
    source = SOURCE.parent / "_source_checks" / "f5_pinned" / "src" / "f5_tts" / "model"
    if not source.exists():
        pytest.skip("Optional pinned F5 checkout absent")
    setup = runpy.run_path(str(SOURCE.with_name("kaggle_setup_cells.py")))
    tree = ast.parse(setup["SETUP_CODE"])
    declared = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "DEPENDENCIES" for target in node.targets))
    available = {package.split("==")[0].replace("-", "_") for package in declared["f5"]}
    available.update({"torch", "torchaudio", "numpy", "scipy", "soundfile", "tqdm"})
    imported = set()
    for path in source.rglob("*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    required = imported - sys.stdlib_module_names - {"f5_tts"}
    assert required <= available, f"Missing eager dependencies: {required - available}"


@pytest.mark.parametrize("parameter", ["device", "output_dir", "estimate_only", "n", "seed"])
def test_profile_cannot_override_supervisor_identity(env, parameter):
    env["MODEL_PROFILES"]["kokoro"][0]["parameters"][parameter] = 1
    with pytest.raises(ValueError, match="supervisor-controlled"):
        env["profile_jobs"]()


@pytest.mark.parametrize("word_count", [9, 501])
def test_profile_rejects_outside_requested_word_range(env, word_count):
    env["TEXT_CASES"] = [{"name": "outside", "text": " ".join(["word"] * word_count)}]
    with pytest.raises(ValueError, match="outside 10-500"):
        env["profile_jobs"]()


def test_nvml_deduplicates_pid_and_ignores_unavailable_values(env):
    class NvmlError(Exception):
        pass

    env["pynvml"] = SimpleNamespace(
        NVMLError=NvmlError,
        nvmlDeviceGetComputeRunningProcesses=lambda _: [
            SimpleNamespace(pid=1, usedGpuMemory=100),
            SimpleNamespace(pid=2, usedGpuMemory=200),
            SimpleNamespace(pid=3, usedGpuMemory=2**64 - 1),
            SimpleNamespace(pid=999, usedGpuMemory=4000),
        ],
        nvmlDeviceGetGraphicsRunningProcesses=lambda _: [SimpleNamespace(pid=1, usedGpuMemory=120)],
    )
    assert env["nvml_process_usage"]("handle", {1, 2, 3}) == (320, [1, 2])
    assert env["nvml_process_usage"]("handle", {3}) == (None, [])


def test_monitor_latches_budget_and_falls_back_when_process_accounting_unavailable(env):
    class OneTick:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _):
            self.stopped = True

    job = make_job(env)
    env["CONFIG"]["budget_scope"] = "process_tree"
    env["pynvml"] = SimpleNamespace(
        nvmlDeviceGetMemoryInfo=lambda _: SimpleNamespace(used=5_100_000_000),
        nvmlDeviceGetUtilizationRates=lambda _: SimpleNamespace(gpu=50, memory=20),
    )
    env["process_tree"] = lambda _: {10}
    env["nvml_process_usage"] = lambda *_: (None, [])
    env["psutil"] = SimpleNamespace(Process=lambda _: SimpleNamespace(create_time=lambda: 1.))
    monitor = env["JobMonitor"](job, 0)
    monitor.pid = 10
    monitor.stop_event = OneTick()
    monitor.observe()
    assert monitor.error is None
    assert monitor.device_peak == 5_100_000_000
    assert monitor.process_peak is None
    assert monitor.violation["scope"] == "device_total"
    assert monitor.violation["budget_bytes"] == 5_000_000_000
    trace = Path(job["job_dir"], "vram_trace.csv").read_text()
    assert "5100000000" in trace


def test_worker_finish_catches_late_budget_violation_and_avoids_reused_pids(env):
    killed = []

    class ProcessGone(Exception):
        pass

    def fake_process(pid):
        return SimpleNamespace(
            create_time=lambda: {2: 999., 3: 33.}[pid],
            kill=lambda: killed.append(pid),
        )

    env["psutil"] = SimpleNamespace(Process=fake_process, NoSuchProcess=ProcessGone, AccessDenied=ProcessGone)
    env["nvml_process_usage"] = lambda *_: (0, [])
    job = make_job(env)
    worker = env["Worker"].__new__(env["Worker"])
    worker.job, worker.slot, worker.started = job, 0, time.perf_counter()
    worker.process = SimpleNamespace(pid=1, returncode=0, wait=lambda: 0)
    worker.reader = SimpleNamespace(join=lambda **_: None)
    worker.abort_reason, worker.tail = None, deque()
    violation = {"observed_bytes": 5_100_000_000, "budget_bytes": 5_000_000_000, "scope": "device_total"}
    worker.monitor = SimpleNamespace(process_ids={1, 2, 3}, process_births={2: 22., 3: 33.},
        finish=lambda: {"monitor_error": None, "budget_violation": violation})
    result = worker.finish()
    assert result["status"] == "budget_exceeded"
    assert result["exit_code"] == 0
    assert result["budget_violation"] == violation
    assert killed == [3], "A tracked PID reused by another process must never be killed"
    assert Path(job["job_dir"], "supervisor_result.json").exists()


@pytest.mark.parametrize("model", ["kokoro", "f5", "emotivoice", "styletts2", "omnivoice", "supertonic"])
def test_native_schema_six_models_and_exact_preview_mapping(env, model):
    job = make_job(env, model)
    record = {"seed": 100, "file": "/clip.wav", "generation_seconds": 3}
    if model == "kokoro":
        data = {"predicted_export_samples": 12000, "export_sample_rate": 48000,
                "preflight_seconds": .2, "startup_seconds": 2, "samples": [record]}
    elif model == "f5":
        data = {"preview": {"exported_samples": 12000, "preflight_seconds": .2, "startup_seconds": 2},
                "settings": {"sample_rate": 48000}, "outputs": [record]}
    elif model == "styletts2":
        data = {"preview": [{"seed": 101, "exported_samples": 24000}, {"seed": 100, "exported_samples": 12000}],
                "settings": {"sample_rate": 48000}, "preflight_seconds": .2,
                "startup_seconds": 2, "outputs": [record, dict(record, seed=101)]}
    elif model in {"emotivoice", "supertonic"}:
        data = {"predicted_samples": 12000, "export_rate": 48000,
                "warm_preview_seconds": .2, "startup_seconds": 2, "runs": [record]}
    else:
        data = {"plan": {"predicted_export_samples": 12000, "sample_rate": 48000,
                         "planner_ms": 200, "cold_preflight_ms": 2000}, "samples": [record]}
    result = env["native_schema"](job, data)
    assert result["planner_seconds"] == .2
    assert result["startup_seconds"] == 2
    assert result["export_rate"] == 48000
    assert result["rows"][0]["predicted_samples"] == 12000
    assert result["rows"][0]["seed"] == 100
    if model == "styletts2":
        assert result["rows"][1]["predicted_samples"] == 24000
        assert [preview["predicted_samples"] for preview in result["predictions"]] == [24000, 12000]


def test_allocator_peak_aliases_and_nested_samples(env):
    data = {"memory": {"torch_peak_allocated_bytes": 10, "torch_peak_reserved_bytes": 20},
            "runs": [{"peak_torch_allocated_bytes": 30, "peak_torch_reserved_bytes": 40}],
            "irrelevant": {"torch_peak_reserved_bytes": True}}
    assert env["allocator_peaks"](data) == (30, 40)
    assert env["allocator_peaks"]({}) == (None, None)


def write_kokoro_manifest(job, filename="final.wav", frames=12000, samplerate=48000):
    raw = Path(job["job_dir"]) / "raw"
    source = raw / filename
    sf.write(source, [0.] * frames, samplerate)
    manifest = {"predicted_export_samples": 12000, "predicted_seconds": .25,
                "export_sample_rate": 48000, "preflight_seconds": .1,
                "samples": [{"seed": 100, "path": str(source), "generation_seconds": .5}]}
    (raw / "manifest.json").write_text(json.dumps(manifest))
    return source


def test_collect_reports_independently_reads_wav_and_keeps_failed_jobs(env):
    complete = make_job(env)
    write_kokoro_manifest(complete)
    failed = make_job(env, "f5", 1, status="failed")
    (Path(failed["job_dir"]) / "raw" / "manifest.json").write_text(json.dumps({
        "preview": {"exported_samples": 24000, "preflight_seconds": .01},
        "settings": {"sample_rate": 48000}, "outputs": [],
    }))
    env["JOB_RESULTS"] = [complete, failed]
    jobs, samples = env["collect_reports"]()
    assert len(jobs) == 2 and len(samples) == 1
    assert samples[0]["audio_status"] == "verified"
    assert samples[0]["actual_seconds"] == .25
    assert samples[0]["duration_error_percent"] == 0
    assert jobs[0]["verified_wavs"] == 1
    assert jobs[1]["status"] == "failed"
    assert json.loads(jobs[1]["planned_samples_json"]) == [
        {"seed": 100, "predicted_samples": 24000, "predicted_seconds": .5}
    ]


def test_wrong_wav_sampling_frequency_cannot_be_false_verified(env):
    job = make_job(env)
    write_kokoro_manifest(job, samplerate=24000)
    env["JOB_RESULTS"] = [job]
    jobs, samples = env["collect_reports"]()
    assert samples[0]["predicted_seconds"] == .25
    assert samples[0]["actual_seconds"] == .5
    assert samples[0]["duration_error_percent"] == 50
    assert samples[0]["audio_status"] == "partial"
    assert jobs[0]["status"] == "report_validation_failed"


def test_temporary_wave_is_never_certified_even_if_in_manifest(env):
    job = make_job(env)
    write_kokoro_manifest(job, filename="final.tmp.wav")
    env["JOB_RESULTS"] = [job]
    jobs, samples = env["collect_reports"]()
    assert samples[0]["audio_status"] == "partial"
    assert jobs[0]["verified_wavs"] == 0
    assert jobs[0]["status"] == "report_validation_failed"


def test_unreadable_only_outputs_keep_null_duration_fields_and_download_link(env):
    job = make_job(env, status="failed")
    source = Path(job["job_dir"], "raw", "broken.wav")
    source.write_bytes(b"An interrupted, unreadable WAV must still be archived.")
    Path(job["job_dir"], "raw", "manifest.json").write_text(json.dumps({
        "predicted_export_samples": 12000, "export_sample_rate": 48000,
        "samples": [{"seed": 100, "path": str(source)}],
    }))
    env["JOB_RESULTS"] = [job]
    _, samples = env["collect_reports"]()
    assert samples[0]["audio_status"] == "unreadable"
    for field in ("actual_seconds", "duration_error_percent", "predicted_seconds"):
        assert field in samples[0], f"Unreadable output needs stable report column {field}"
    assert samples[0]["actual_seconds"] is None
    assert samples[0]["duration_error_percent"] is None

    class Frame:
        def __init__(self, rows):
            self.rows = rows
            self.empty = not rows

        def to_csv(self, path, index=False):
            Path(path).write_text("report\n")

        def to_html(self, **_):
            return "<table></table>"

        def reindex(self, columns):
            return Frame([{key: row.get(key) for key in columns} for row in self.rows])

        def __getitem__(self, columns):
            return Frame([{key: row[key] for key in columns} for row in self.rows])

    displayed = []
    env.update(pd=SimpleNamespace(DataFrame=Frame), KERNEL_READY=True,
               FileLink=lambda path: "DOWNLOAD " + path,
               display=lambda value, **_: displayed.append(value))
    env["CONFIG"]["preview_players"] = 6
    exec(runpy.run_path(str(SOURCE))["REPORT_CODE"], env)
    assert env["ZIP_PATH"].is_file()
    assert any(isinstance(value, str) and value.startswith("DOWNLOAD ") for value in displayed)
    with zipfile.ZipFile(env["ZIP_PATH"]) as zipped:
        audio_names = [name for name in zipped.namelist() if name.startswith("audio/")]
        assert len(audio_names) == 1 and "UNREADABLE__" in audio_names[0]
        assert zipped.read(audio_names[0]) == source.read_bytes()


def test_scheduler_two_slots_distinct_models_and_no_overlap(env, monkeypatch):
    running, starts = {}, []

    class FakeWorker:
        def __init__(self, job, slot):
            assert slot not in running, "Two workers cannot share a GPU"
            assert len(running) < 2
            self.job, self.slot, self.started = job, slot, time.perf_counter()
            self.monitor = SimpleNamespace(error=None, violation=None)
            self.calls = 0
            self.process = SimpleNamespace(poll=self.poll)
            running[slot] = self
            starts.append((job["model"], slot, {worker.job["model"] for worker in running.values()}))

        def poll(self):
            self.calls += 1
            return 0 if self.calls >= 3 else None

        def finish(self):
            del running[self.slot]
            return dict(self.job, status="completed", gpu_slot=self.slot, gpu_release_failed_pids=[])

        def terminate(self, _):
            self.calls = 3

    jobs = [make_job(env, model, index) for index, model in enumerate(["kokoro", "kokoro", "f5", "f5", "omnivoice"])]
    env["Worker"] = FakeWorker
    env["live_html"] = lambda *_: "test"
    monkeypatch.setattr(time, "sleep", lambda _: None)
    env["run_benchmark"](jobs)
    assert starts[:2] == [("kokoro", 0, {"kokoro"}), ("f5", 1, {"kokoro", "f5"})]
    assert not running
    assert len(env["JOB_RESULTS"]) == 5
    assert all(result["status"] == "completed" for result in env["JOB_RESULTS"])


def test_package_contains_reports_logs_and_audio_but_not_weight_cache(env):
    class Frame:
        def __init__(self, rows):
            self.rows = rows
            self.empty = not rows

        def to_csv(self, path, index=False):
            Path(path).write_text("test_csv\n")

        def to_html(self, **_):
            return "<table></table>"

    env["pd"] = SimpleNamespace(DataFrame=Frame)
    (env["SCRIPTS_DIR"] / "kokoro_samples.py").write_text("# adapter provenance")
    cache = env["ROOT"] / "assets" / "hf_cache"
    cache.mkdir(parents=True)
    (cache / "huge_model.safetensors").write_bytes(b"model weight sentinel")
    venv = env["ROOT"] / "venvs"
    venv.mkdir()
    (venv / "torch.so").write_bytes(b"dependency sentinel")
    setup_logs = env["ROOT"] / "logs" / "setup"
    setup_logs.mkdir(parents=True)
    (setup_logs / "bootstrap.log").write_text("managed Python and package setup")
    (env["ROOT"] / "reference_report.json").write_text('{"status":"synthetic"}')
    job = make_job(env)
    write_kokoro_manifest(job)
    Path(job["job_dir"], "generation.log").write_text("actual model progress\r50%\n")
    env["JOB_RESULTS"] = [job]
    jobs, samples = env["collect_reports"]()
    _, _, archive = env["package_results"](jobs, samples)
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert any(name.startswith("audio/") and name.endswith(".wav") for name in names)
        assert "reports/jobs.csv" in names
        assert "reports/samples.json" in names
        assert "source/kokoro_samples.py" in names
        assert "setup_logs/bootstrap.log" in names
        assert "reports/reference_report.json" in names
        assert any(name.endswith("generation.log") for name in names)
        assert not any("huge_model" in name or "torch.so" in name or "hf_cache" in name for name in names)
        assert b"actual model progress" in zipped.read(f"job_details/{job['job_id']}/generation.log")


def test_quality_dependency_failure_is_reported_without_interrupting_exports(env):
    job = make_job(env)
    Path(job["job_dir"], "text.txt").write_text("one two three four five six seven eight nine ten")
    write_kokoro_manifest(job)
    env["JOB_RESULTS"] = [job]
    env["CONFIG"]["quality_checks"] = True
    env["ENV_DIR"] = env["ROOT"] / "envs"
    env["UV"] = ["uv"]

    def offline(*_args, **_kwargs):
        raise RuntimeError("Installer unavailable in this test")

    env["setup_command"] = offline
    _, samples = env["collect_reports"]()
    result = env["run_quality_checks"](samples)
    assert result["status"] == "failed"
    assert result["error"] == "Installer unavailable in this test"
    assert samples[0]["audio_status"] == "verified"
    assert Path(env["PACKAGE_DIR"], samples[0]["audio_file"]).is_file()
    assert (env["RUN_DIR"] / "quality" / "quality_report.json").is_file()


def test_quality_results_merge_by_job_and_absolute_audio_path(env, monkeypatch):
    job = make_job(env)
    Path(job["job_dir"], "text.txt").write_text("one two three four five six seven eight nine ten")
    write_kokoro_manifest(job)
    env["JOB_RESULTS"] = [job]
    env["CONFIG"].update(quality_checks=True, asr_quality_check=False)
    env["ENV_DIR"] = env["ROOT"] / "envs"
    python = env["ENV_DIR"] / "quality_checks" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    env["UV"] = ["uv"]
    env["BASE_ENV"] = {"CUDA_VISIBLE_DEVICES": ""}
    env["setup_command"] = lambda *_args, **_kwargs: None

    def checked(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        requests = json.loads(Path(command[command.index("--requests-json") + 1]).read_text())
        output = Path(command[command.index("--output-json") + 1])
        row = {**requests[0], "status": "completed", "longest_interior_silence_seconds": .2,
               "clipped_sample_fraction": 0, "asr_status": "disabled"}
        output.write_text(json.dumps({"status": "completed", "rows": [row]}))

    monkeypatch.setattr(subprocess, "run", checked)
    _, samples = env["collect_reports"]()
    result = env["run_quality_checks"](samples)
    assert result["status"] == "completed"
    assert samples[0]["longest_interior_silence_seconds"] == .2
    assert samples[0]["clipped_sample_fraction"] == 0
    assert samples[0]["audio_status"] == "verified"


def test_authoring_cells_parse_and_gpu_isolation_is_explicit():
    cells = runpy.run_path(str(SOURCE))
    for name, code in cells.items():
        if name.endswith("_CODE"):
            ast.parse(code)
    code = cells["SUPERVISOR_CODE"]
    assert '"CUDA_VISIBLE_DEVICES": GPU_INVENTORY[slot]["uuid"]' in code
    assert "start_new_session=True" in code
    assert "os.killpg" in code
    assert r're.split(r"[\r\n]"' in code


def test_default_passages_are_complete_nonrepeating_and_exact_length():
    cells = runpy.run_path(str(SOURCE))
    tree = ast.parse(cells["CONFIG_CODE"])
    passages = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "SAMPLE_PASSAGES" for target in node.targets))
    assert set(passages) == {"10", "100", "500"}
    for count, passage in passages.items():
        assert len(passage.split()) == int(count)
        assert passage.endswith(".")
        sentences = [s.strip().casefold() for s in re.split(r"[.!?]+", passage) if s.strip()]
        assert len(sentences) == len(set(sentences))
        assert passage == (SOURCE.parent / f"sample_text_{count}.txt").read_text(encoding="utf-8").strip()
    assert len(passages["500"].split("\n\n")) == 5

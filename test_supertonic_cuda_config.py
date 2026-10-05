"""GPU configuration regressions that do not require a GPU or model weights."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_script():
    path = Path(__file__).with_name("supertonic_samples.py")
    spec = importlib.util.spec_from_file_location("supertonic_gpu_configuration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUPERTONIC = load_script()
BASE_ARGS = ["--repo", ".", "--assets-dir", ".", "--text", "one two three four five six seven eight nine ten"]


class FakeSession:
    def __init__(self, providers):
        self.providers = providers

    def get_providers(self):
        return self.providers

    def get_provider_options(self):
        return {name: {} for name in self.providers}


class SupertonicCudaConfigTests(unittest.TestCase):
    def test_cpu_defaults_need_no_cuda_acknowledgement(self):
        args = SUPERTONIC.parse_args(BASE_ARGS)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(SUPERTONIC.execution_providers(args, 4), ["CPUExecutionProvider"])

    def test_cuda_requires_explicit_unverified_total_budget_acknowledgement(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            SUPERTONIC.parse_args(BASE_ARGS + ["--device", "cuda:0"])
        args = SUPERTONIC.parse_args(BASE_ARGS + ["--device", "cuda:0", "--allow-unverified-cuda-budget"])
        self.assertEqual(args.device, "cuda:0")

    def test_visible_cuda_device_ordinal_is_used_in_provider_options(self):
        for device, expected in (("cuda", 0), ("cuda:0", 0), ("cuda:1", 1), ("cuda:12", 12)):
            args = SUPERTONIC.parse_args(BASE_ARGS + ["--device", device, "--allow-unverified-cuda-budget"])
            providers = SUPERTONIC.execution_providers(args, 4)
            self.assertEqual(providers[0][1]["device_id"], expected)

    def test_independent_session_limits_do_not_multiply_aggregate_budget(self):
        for budget in (1, 2.5, 5):
            args = SUPERTONIC.parse_args(
                BASE_ARGS + ["--device", "cuda:0", "--allow-unverified-cuda-budget", "--max-vram-gb", str(budget)]
            )
            providers = SUPERTONIC.execution_providers(args, len(SUPERTONIC.WEIGHT_HASHES))
            arena = providers[0][1]["gpu_mem_limit"]
            aggregate = arena * len(SUPERTONIC.WEIGHT_HASHES)
            self.assertGreater(arena, 0)
            self.assertLessEqual(aggregate + args.cuda_reserve_mb * 1_000_000, budget * 1_000_000_000)
            self.assertGreater(aggregate, budget * 1_000_000_000 - args.cuda_reserve_mb * 1_000_000 - 4)

    def test_budget_and_device_validation_reject_invalid_cli_settings(self):
        for extra in (
            ["--device", "gpu"],
            ["--device", "cuda:-1"],
            ["--max-vram-gb", "5.1"],
            ["--cuda-reserve-mb", "-1"],
            ["--device", "cuda:0", "--allow-unverified-cuda-budget", "--cuda-reserve-mb", "5000"],
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                SUPERTONIC.parse_args(BASE_ARGS + extra)

    def test_json_cuda_configuration_obeys_cli_boolean_override(self):
        settings = {
            "repo": ".", "assets_dir": ".", "text": "one two three four five six seven eight nine ten",
            "device": "cuda:0", "allow_unverified_cuda_budget": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps(settings), encoding="utf-8")
            args = SUPERTONIC.parse_args(["--config", str(path)])
            self.assertTrue(args.allow_unverified_cuda_budget)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                SUPERTONIC.parse_args(["--config", str(path), "--no-allow-unverified-cuda-budget"])

    def test_provider_failure_is_not_silently_accepted_as_cpu_execution(self):
        SUPERTONIC.validate_session_providers(FakeSession(["CPUExecutionProvider"]), "cpu", "vocoder")
        SUPERTONIC.validate_session_providers(
            FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"]), "cuda:0", "vocoder"
        )
        with self.assertRaisesRegex(RuntimeError, "cannot silently fall back"):
            SUPERTONIC.validate_session_providers(FakeSession(["CPUExecutionProvider"]), "cuda:0", "vocoder")

    def test_cuda_report_does_not_mislabel_allocator_limits_as_measured_peak(self):
        args = SUPERTONIC.parse_args(BASE_ARGS + ["--device", "cuda:0", "--allow-unverified-cuda-budget"])
        engine = SUPERTONIC.Engine.__new__(SUPERTONIC.Engine)
        engine.args = args
        engine.providers = SUPERTONIC.execution_providers(args, 4)
        engine.sessions = {name: FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"]) for name in SUPERTONIC.WEIGHT_HASHES}
        report = engine.memory_report()
        self.assertIsNone(report["peak_gpu_vram_bytes"])
        self.assertLess(report["cuda_arena_limit_aggregate_bytes"], report["requested_total_budget_bytes"])
        self.assertIn("externally", report["gpu_budget_guarantee"])

    def test_cpu_report_remains_zero_gpu_allocation(self):
        engine = SUPERTONIC.Engine.__new__(SUPERTONIC.Engine)
        engine.args = SimpleNamespace(device="cpu")
        engine.providers = ["CPUExecutionProvider"]
        report = engine.memory_report()
        self.assertEqual(report["peak_gpu_vram_bytes"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

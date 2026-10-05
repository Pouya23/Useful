"""Focused planning regressions; run with pytest or `python this_file.py`.

No model weights are required. NumPy/SciPy verify the actual resampler's output
length; optional CPU PyTorch verifies convolution length formulas independently.
Real checkpoint smoke results are stored separately under .validation/.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_script(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EMOTIVOICE = load_script("emotivoice_samples")
SUPERTONIC = load_script("supertonic_samples")


class PlanningTests(unittest.TestCase):
    def test_fallback_pronunciation_retains_final_phone_before_upstream_punctuation(self):
        """Exercise the real pinned tokenizer, extracted without model imports."""
        source = Path(__file__).parent / ".validation" / "emotivoice_source" / "frontend_en.py"
        if not source.exists():
            self.skipTest("Pinned frontend source is an optional local regression fixture")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_eng_phoneme"
        )
        import re

        namespace = {"re": re}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        tokenizer = namespace["get_eng_phoneme"]

        def backend(token):
            return ["K", "AE1", "G", "AH0", "L"] if token == "Kaggle" else [token]

        unguarded = tokenizer("Kaggle.", backend, {})
        self.assertNotIn("[L]", unguarded)
        repaired = tokenizer("Kaggle.", EMOTIVOICE.GuardedEnglishG2P(backend), {})
        self.assertIn("[L]", repaired)
        self.assertEqual(repaired.split(), ["<sos/eos>", "[K]", "[AE1]", "[G]", "[AH0]", "[L]", "<sos/eos>"])
        self.assertEqual(
            tokenizer("Kaggle!!!", EMOTIVOICE.GuardedEnglishG2P(backend), {}).split(),
            repaired.split(),
        )
        with self.assertRaisesRegex(ValueError, "refusing to drop a word"):
            EMOTIVOICE.GuardedEnglishG2P(lambda _: [])("missing")

    def test_short_lexical_phonemes_survive_speed_rounding_but_pauses_are_not_inflated(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU PyTorch is optional for this check")
        phones = ["<sos/eos>", "[D]", "[AH0]", "engsp1", "sh", "a1", "cn_eng_sp", ".", "_"]
        base = torch.tensor([[0, 0.2, 1, 0, 0, 2, 0, 0, 0]], dtype=torch.float32)
        factors = torch.full_like(base, 0.25)
        durations, zeros, changed = EMOTIVOICE.apply_duration_controls(base, factors, phones, 1)
        self.assertEqual(durations.tolist(), [[0, 1, 1, 0, 1, 1, 0, 0, 0]])
        self.assertEqual(zeros, [1, 2, 4, 5])
        self.assertEqual(changed, [1, 2, 4, 5])
        unguarded, _, changed = EMOTIVOICE.apply_duration_controls(base, factors, phones, 0)
        self.assertEqual(unguarded.tolist(), [[0] * len(phones)])
        self.assertEqual(changed, [])

    def test_sentence_and_clause_boundaries_preferred_without_text_loss(self):
        for module in (EMOTIVOICE, SUPERTONIC):
            text = "This sentence ends. The following sentence goes on for several more words."
            chunks = module.chunk_text(text, 35)
            self.assertEqual(chunks[0], "This sentence ends.")
            self.assertEqual(" ".join(chunks), text)
            self.assertTrue(all(len(chunk) <= 35 for chunk in chunks))
            self.assertEqual(
                module.chunk_text("We reach a clause, and this still continues onward", 30)[0],
                "We reach a clause,",
            )
            self.assertEqual(module.boundary_strength("Dr."), 0)
            self.assertEqual(module.boundary_strength("J."), 0)
            self.assertEqual(module.boundary_strength("ends.”"), 2)
            self.assertEqual(
                module.chunk_text("Short paragraph.\nNext paragraph.", 300),
                ["Short paragraph.", "Next paragraph."],
            )

    def test_edge_fades_and_peak_protection_preserve_lengths_and_relative_levels(self):
        import numpy as np

        for module in (EMOTIVOICE, SUPERTONIC):
            signal = np.full(1000, 0.6, dtype=np.float32)
            faded = module.edge_fade(signal, 16000, 3)
            self.assertEqual(len(faded), len(signal))
            self.assertEqual(faded[0], 0)
            self.assertEqual(faded[-1], 0)
            self.assertTrue(np.array_equal(faded[48:-48], signal[48:-48]))
            self.assertTrue(np.array_equal(module.edge_fade(signal, 16000, 0), signal))
            peaks = np.array([-2.0, 0, 1.0, 0.5], dtype=np.float32)
            output, levels = module.protect_peak(peaks, 0, True)
            self.assertEqual(len(output), len(peaks))
            self.assertAlmostEqual(float(np.max(np.abs(output))), 0.98, places=6)
            self.assertTrue(np.allclose(output, peaks * 0.49))
            self.assertAlmostEqual(levels["peak_scale"], 0.49)
            output, levels = module.protect_peak(peaks, 0, False)
            self.assertTrue(np.array_equal(output, peaks))
            quiet, _ = module.protect_peak(signal, 0, True)
            self.assertTrue(np.array_equal(quiet, signal))

    def test_chunking_preserves_order_and_all_words_at_both_boundaries(self):
        for module in (EMOTIVOICE, SUPERTONIC):
            for count in (10, 73, 500):
                text = " ".join(f"word{index}" for index in range(count))
                chunks = module.chunk_text(text, 90)
                self.assertEqual(" ".join(chunks), text)
                self.assertTrue(all(0 < len(chunk) <= 90 for chunk in chunks))
            with self.assertRaises(ValueError):
                module.chunk_text("oversized" * 100, 90)

    def test_actual_resampler_sample_count_matches_plan(self):
        import numpy as np

        for module in (EMOTIVOICE, SUPERTONIC):
            native_rate = 16000 if module is EMOTIVOICE else 44100
            for length in (1, 17, 256, 3072, 12001):
                signal = np.zeros(length, dtype=np.float32)
                signal[length // 2] = 0.5
                for export_rate in (8000, 16000, 22050, 24000, 44100, 48000, 96000):
                    result = module.resample_audio(signal, native_rate, export_rate)
                    expected = module.predicted_export_samples(length, native_rate, export_rate)
                    self.assertEqual(len(result), expected, (module.__name__, length, export_rate))
                    self.assertTrue(np.isfinite(result).all())

    def test_convolution_formula_matches_actual_torch_layers(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU PyTorch is optional for this check")
        torch.set_num_threads(1)
        with torch.inference_mode():
            for length in (1, 2, 7, 31, 333):
                for kernel, stride, padding in ((7, 1, 3), (3, 1, 2), (3, 2, 1)):
                    layer = torch.nn.Conv1d(2, 3, kernel, stride=stride, padding=padding)
                    result = layer(torch.zeros(1, 2, length))
                    self.assertEqual(result.shape[-1], EMOTIVOICE.conv_output_length(length, layer))
                for kernel, stride, padding in ((16, 8, 4), (4, 2, 1), (5, 2, 2)):
                    layer = torch.nn.ConvTranspose1d(2, 3, kernel, stride=stride, padding=padding)
                    result = layer(torch.zeros(1, 2, length))
                    self.assertEqual(
                        result.shape[-1], EMOTIVOICE.conv_output_length(length, layer, transposed=True)
                    )

    def test_json_types_cli_overrides_and_mutually_exclusive_inputs(self):
        for module in (EMOTIVOICE, SUPERTONIC):
            base = {"text": "one two three four five six seven eight nine ten", "repo": "."}
            if module is SUPERTONIC:
                base["assets_dir"] = "."
            else:
                base.update(generator_checkpoint="dummy", style_checkpoint="dummy")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "settings.json"
                path.write_text(json.dumps(base | {"n": 2, "estimate_only": True}), encoding="utf-8")
                args = module.parse_args(["--config", str(path), "--n", "3", "--no-estimate-only"])
                self.assertEqual(args.n, 3)
                self.assertFalse(args.estimate_only)
                args = module.parse_args(["--config", str(path), "--text-file", "alternate.txt"])
                self.assertIsNone(args.text)
                self.assertEqual(args.text_file, Path("alternate.txt"))
                for invalid in (
                    {"n": True},
                    {"n": 1.5},
                    {"estimate_only": "yes"},
                    {"speed": None},
                    {"text": 12},
                    {"device": True},
                    {"max_vram_gb": 5.1},
                    {"speed": float("nan")},
                    {"sample_rate": 7999},
                    {"seed": -1},
                    {"edge_fade_ms": -1},
                    {"edge_fade_ms": 21},
                    {"edge_fade_ms": float("nan")},
                    {"gain_db": float("inf")},
                    {"normalize_peak": "yes"},
                    {"min_phoneme_frames": -1} if module is EMOTIVOICE else {"steps": 0},
                    {"min_phoneme_frames": 11} if module is EMOTIVOICE else {"steps": -1},
                ):
                    path.write_text(json.dumps(base | invalid), encoding="utf-8")
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                        module.parse_args(["--config", str(path)])
                    self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

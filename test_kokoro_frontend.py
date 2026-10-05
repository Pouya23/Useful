"""Ensure Kokoro frontend guards cannot silently drop or truncate words."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).with_name("kokoro_samples.py")
SPEC = importlib.util.spec_from_file_location("tested_kokoro_frontend", SCRIPT)
kokoro = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kokoro)


def token(text, phonemes):
    return SimpleNamespace(text=text, phonemes=phonemes)


@pytest.mark.parametrize("phonemes", [None, "", " "])
def test_unpronounced_lexical_word_is_rejected(phonemes):
    with pytest.raises(ValueError, match="no pronunciation"):
        kokoro.validate_english_tokens([token("unrecognizedword", phonemes)], 510)


def test_empty_punctuation_does_not_require_pronunciation():
    kokoro.validate_english_tokens([token("—", None), token("hello", "həlˈO")], 510)


def test_indivisible_overlong_word_is_rejected_before_chunker():
    with pytest.raises(ValueError, match="refusing to truncate"):
        kokoro.validate_english_tokens([token("longword", "a" * 511)], 510)


def test_exact_context_boundary_and_multiple_complete_words_are_valid():
    kokoro.validate_english_tokens([token("one", "a" * 510), token("two", "b" * 509)], 510)

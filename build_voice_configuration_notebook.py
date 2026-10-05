"""Build a new self-contained benchmark with complete voice/configuration guides."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import build_kaggle_notebook as benchmark
import voice_guide_common as common
import voice_guide_compact as compact
import voice_guide_duration as duration
import voice_guide_f5_style as f5_style
import voice_guide_omni as omni

OUTPUT_NAME = "kaggle_dual_t4_tts_voice_configuration_guide.ipynb"
MODEL_FILES = {
    "kokoro": "kokoro_samples.py", "f5": "f5_samples.py",
    "emotivoice": "emotivoice_samples.py", "styletts2": "styletts2_samples.py",
    "omnivoice": "omnivoice_samples.py", "supertonic": "supertonic_samples.py",
}


class ParserCaptured(Exception):
    def __init__(self, parser):
        self.parser = parser


def capture_parser(parser, *_args, **_kwargs):
    """Stop before parsing/validation; no inference or model imports occur."""
    raise ParserCaptured(parser)


def option_inventory(root: Path) -> dict:
    """Read the actual stdlib-only parsers, including BooleanOptionalAction flags."""
    inventory = {}
    for model, filename in MODEL_FILES.items():
        name = f"_voice_doc_{model}"
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            if model == "kokoro":
                parser = module.parser()
            else:
                parse = module.parse_arguments if model == "omnivoice" else module.parse_args
                with patch.object(argparse.ArgumentParser, "parse_known_args", capture_parser):
                    try:
                        parse() if model == "omnivoice" else parse([])
                    except ParserCaptured as captured:
                        parser = captured.parser
                    else:
                        raise RuntimeError(f"Could not capture {model}'s argument parser")
            rows = []
            for action in parser._actions:
                if action.dest == "help":
                    continue
                default = action.default
                if isinstance(default, Path):
                    try:
                        default = str(default.relative_to(root))
                    except ValueError:
                        default = str(default)
                boolean = isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction))
                rows.append({
                    "key": action.dest, "cli_flags": action.option_strings,
                    "type": "bool" if boolean else getattr(action.type, "__name__", "str"),
                    "script_default": default,
                    "choices": list(action.choices) if action.choices is not None else None,
                    "help": action.help,
                    "supervisor_controlled": action.dest in {
                        "config", "text", "text_file", "device", "n", "seed", "output_dir",
                        "max_vram_gb", "estimate_only", "self_test",
                    },
                })
            inventory[model] = rows
        finally:
            sys.modules.pop(name, None)
    json.dumps(inventory, allow_nan=False)
    return inventory


def algorithm_locations(root: Path) -> list[dict]:
    rows = []
    for model, names in duration.ALGORITHM_FUNCTIONS.items():
        tree = ast.parse((root / MODEL_FILES[model]).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in names:
                rows.append({"model": model, "file": MODEL_FILES[model], "function": node.name, "line": node.lineno})
    return rows


CATALOG_CODE = r'''
# No weight downloads or GPU allocations. Run the experiment configuration cell first.
import csv as catalog_csv
import html as catalog_markup
import json

VOICE_CATALOG = json.loads(__VOICE_JSON__)
ADAPTER_OPTIONS = json.loads(__OPTION_JSON__)
ALGORITHM_LOCATIONS = json.loads(__ALGORITHM_JSON__)
REFERENCE_GUIDE = __GUIDE_REPR__
CATALOG_DIR = PACKAGE_DIR / "catalog"
CATALOG_DIR.mkdir(parents=True, exist_ok=True)
for filename, payload in (("voices.json", VOICE_CATALOG), ("adapter_options.json", ADAPTER_OPTIONS),
                          ("notebook_default_profiles.json", MODEL_PROFILES), ("duration_algorithm_locations.json", ALGORITHM_LOCATIONS)):
    (CATALOG_DIR / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
(CATALOG_DIR / "REFERENCE_GUIDE.md").write_text(REFERENCE_GUIDE, encoding="utf-8")

def write_catalog_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = catalog_csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)

for model, catalog in VOICE_CATALOG.items():
    if isinstance(catalog, list):
        write_catalog_csv(CATALOG_DIR / f"{model}_all_voices.csv", catalog)
for model, rows in ADAPTER_OPTIONS.items():
    write_catalog_csv(CATALOG_DIR / f"{model}_all_options.csv", rows)

# Edit these filters and rerun this cell to explore every embedded speaker row.
CATALOG_MODEL = "kokoro"       # kokoro / emotivoice / supertonic
CATALOG_QUERY = ""             # Case-insensitive substring across fields; e.g. a speaker label/name.
CATALOG_GENDER = "all"         # all / male / female (exact normalized gender match).
CATALOG_LIMIT = 100            # None displays every matched row, including all 2,014 EmotiVoice speakers.
OPTION_MODEL = "omnivoice"    # Any of the six model keys.

if CATALOG_MODEL not in VOICE_CATALOG or not isinstance(VOICE_CATALOG[CATALOG_MODEL], list):
    raise ValueError("CATALOG_MODEL must select a finite preset catalog: kokoro, emotivoice, or supertonic")
if CATALOG_GENDER not in {"all", "male", "female"}:
    raise ValueError("CATALOG_GENDER must be all, male, or female")
if CATALOG_LIMIT is not None and (type(CATALOG_LIMIT) is not int or CATALOG_LIMIT < 1):
    raise ValueError("CATALOG_LIMIT must be a positive integer or None")
if OPTION_MODEL not in ADAPTER_OPTIONS:
    raise ValueError("OPTION_MODEL must select one of the six models")

def normalized_gender(row):
    gender = str(row.get("gender", "")).casefold().strip()
    return {"m": "male", "f": "female", "man": "male", "woman": "female"}.get(gender, gender)

catalog_rows = [row for row in VOICE_CATALOG[CATALOG_MODEL]
                if (CATALOG_GENDER == "all" or normalized_gender(row) == CATALOG_GENDER)
                and CATALOG_QUERY.casefold() in " ".join(str(value) for value in row.values()).casefold()]
shown_rows = catalog_rows if CATALOG_LIMIT is None else catalog_rows[:CATALOG_LIMIT]

def catalog_html(rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    headers = "".join(f"<th>{catalog_markup.escape(key)}</th>" for key in columns)
    body = "".join("<tr>" + "".join(f"<td>{catalog_markup.escape(str(row.get(key, '')))}</td>" for key in columns) + "</tr>" for row in rows)
    return '<div style="max-height:520px;overflow:auto"><table border="1" cellpadding="5"><thead><tr>' + headers + '</tr></thead><tbody>' + body + '</tbody></table></div>'

print("Complete preset catalogs:", {key: len(value) for key, value in VOICE_CATALOG.items() if isinstance(value, list)})
print(f"Showing {len(shown_rows)} / {len(catalog_rows)} matched {CATALOG_MODEL} voices; all rows are saved to CSV.")
print("Option counts:", {key: len(value) for key, value in ADAPTER_OPTIONS.items()})
try:
    from IPython.display import HTML, FileLink, display
    display(HTML(catalog_html(shown_rows)))
    print(f"Actual {OPTION_MODEL} parser defaults (not profile overrides):")
    display(HTML(catalog_html(ADAPTER_OPTIONS[OPTION_MODEL])))
    print("Pre-synthesis planning and generation functions in the embedded Python files:")
    display(HTML(catalog_html(ALGORITHM_LOCATIONS)))
    for filename in (f"{CATALOG_MODEL}_all_voices.csv", f"{OPTION_MODEL}_all_options.csv", "voices.json", "REFERENCE_GUIDE.md"):
        display(FileLink(str(CATALOG_DIR / filename)))
except ImportError:
    print(json.dumps(shown_rows, ensure_ascii=False, indent=2))
    print(json.dumps(ADAPTER_OPTIONS[OPTION_MODEL], ensure_ascii=False, indent=2))

# Catch misspelled/unsupported profile keys before downloading model weights.
# Numeric bounds and reference/checkpoint compatibility remain adapter checks.
for model, profiles in MODEL_PROFILES.items():
    if model not in ADAPTER_OPTIONS:
        raise ValueError(f"Unknown profile model {model}")
    options = {row["key"]: row for row in ADAPTER_OPTIONS[model]}
    for profile in profiles:
        for section in ("parameters", "clone_parameters"):
            for key, value in profile.get(section, {}).items():
                if key not in options:
                    raise ValueError(f"Unknown {model}/{profile['name']} {section} key: {key}")
                if options[key]["supervisor_controlled"]:
                    raise ValueError(f"Set {key} through its supervisor/profile control, not {section}")
                if value is not None and options[key]["choices"] is not None and value not in options[key]["choices"]:
                    raise ValueError(f"Unsupported {model}/{profile['name']} value for {key}: {value!r}")
print("Profile key/choice audit passed. Catalog files and the guide will enter the final WAV/report ZIP.")
'''


def build() -> Path:
    root = Path(__file__).resolve().parent
    guides = [common.GUIDE_MARKDOWN, duration.GUIDE_MARKDOWN, compact.GUIDE_MARKDOWN,
              f5_style.GUIDE_MARKDOWN, omni.GUIDE_MARKDOWN]
    voices = {}
    for module in (compact, f5_style, omni):
        catalog = module.VOICE_CATALOG
        model_catalogs = {catalog["model"]: catalog} if isinstance(catalog.get("model"), str) else catalog
        for key, value in model_catalogs.items():
            if key in voices:
                raise ValueError(f"Duplicate voice catalog key: {key}")
            voices[key] = value
    options = option_inventory(root)
    assert {key: len(voices[key]) for key in ("kokoro", "emotivoice", "supertonic")} == {
        "kokoro": 54, "emotivoice": 2014, "supertonic": 10,
    }
    assert set(voices) == set(MODEL_FILES)
    # A new filename preserves the previous delivered notebook.
    output = benchmark.build(OUTPUT_NAME)
    notebook = json.loads(output.read_text(encoding="utf-8"))
    documentation = [benchmark.markdown(text, cell_id) for text, cell_id in zip(
        guides, ("voice-guide-common", "duration-algorithms", "voice-guide-compact",
                 "voice-guide-f5-style", "voice-guide-omni"), strict=True)]
    notebook["cells"][1:1] = documentation
    guide_text = "\n\n".join(guides)
    catalog_code = CATALOG_CODE.replace("__VOICE_JSON__", repr(json.dumps(voices, ensure_ascii=False)))
    catalog_code = catalog_code.replace("__OPTION_JSON__", repr(json.dumps(options, ensure_ascii=False)))
    catalog_code = catalog_code.replace("__ALGORITHM_JSON__", repr(json.dumps(algorithm_locations(root))))
    catalog_code = catalog_code.replace("__GUIDE_REPR__", repr(guide_text))
    index = next(i for i, cell in enumerate(notebook["cells"]) if cell["id"] == "configuration") + 1
    notebook["cells"][index:index] = [benchmark.markdown(
        """## Explore and export the complete voice/option catalogs

Run the configuration cell, then this cell to inspect catalogs without installing
TTS models. Set `CATALOG_MODEL`, `CATALOG_QUERY`, `CATALOG_GENDER`, and
`CATALOG_LIMIT` inside the cell; set `OPTION_MODEL` to see another adapter's
actual CLI defaults. Use `CATALOG_LIMIT=None` for every matching speaker.
All rows, model metadata, option CSVs, and this reference guide are saved under
`catalog/` and included in the final ZIP. To generate a selected voice, copy its
selection key into a model profile; catalog browsing does not add jobs.
""", "catalog-explorer-notes"), benchmark.code(catalog_code, "voice-option-catalog")]
    notebook["metadata"]["tts_benchmark"].update(
        revision="complete-voice-and-configuration-guide", guide_models=list(MODEL_FILES),
        documented_option_counts={key: len(value) for key, value in options.items()},
        preset_catalog_counts={key: len(voices[key]) for key in ("kokoro", "emotivoice", "supertonic")},
    )
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Extended {output}: {len(notebook['cells'])} cells; {output.stat().st_size:,} bytes")
    return output


if __name__ == "__main__":
    build()

import json
import shutil
from pathlib import Path

import pytest

from src.llm_analysis.prompts import build_user_prompt, get_prompt, load_prompt
from src.llm_analysis.diff_context import RenderedDiff
from src.llm_analysis.schemas import BUG_FIX, NON_BUG_FIX, normalize_label, parse_llm_response

PROMPT_V1 = get_prompt("v1")

VALID = {
    "label": "BUG_FIX",
    "confidence": 0.8,
    "message_evidence": "Fix invalid subnet mask",
    "diff_evidence": "cidr_block /16 -> /24",
    "reason": "Corrects a wrong CIDR.",
    "uncertainty": "",
}


def test_parses_valid_json():
    parsed = parse_llm_response(json.dumps(VALID))
    assert parsed.parse_success and parsed.parse_error is None
    assert (parsed.label, parsed.confidence) == (BUG_FIX, 0.8)
    assert parsed.diff_evidence == "cidr_block /16 -> /24"
    assert parsed.warnings == []


def test_parses_json_inside_code_fence_with_warning():
    parsed = parse_llm_response("Here you go:\n```json\n" + json.dumps(VALID) + "\n```")
    assert parsed.parse_success
    assert parsed.warnings


@pytest.mark.parametrize("text, error", [
    ("", "empty response"),
    ("not json at all", "no JSON object"),
    ('{"label": "BUG_FIX", "confidence": 0.5', "no JSON object"),  # truncado (max_tokens)
    ("[1, 2]", "no JSON object"),
    (json.dumps({**VALID, "label": "MAYBE"}), "invalid or missing label"),
    (json.dumps({k: v for k, v in VALID.items() if k != "label"}), "invalid or missing label"),
    (json.dumps({**VALID, "confidence": 1.5}), "invalid or missing confidence"),
    (json.dumps({**VALID, "confidence": "high"}), "invalid or missing confidence"),
    (json.dumps({**VALID, "confidence": True}), "invalid or missing confidence"),
    (json.dumps({k: v for k, v in VALID.items() if k != "confidence"}), "invalid or missing confidence"),
])
def test_malformed_responses(text, error):
    parsed = parse_llm_response(text)
    assert not parsed.parse_success
    assert error in parsed.parse_error
    assert parsed.label is None or error.startswith("invalid or missing confidence")


def test_missing_text_fields_are_warnings_not_failures():
    parsed = parse_llm_response(json.dumps({"label": "NON_BUG_FIX", "confidence": 0, "extra": 1}))
    assert parsed.parse_success and parsed.label == NON_BUG_FIX
    assert parsed.reason == ""
    assert any("missing field 'reason'" in w for w in parsed.warnings)
    assert any("unexpected fields: extra" in w for w in parsed.warnings)


@pytest.mark.parametrize("raw, expected", [
    ("BUG_FIX", BUG_FIX), ("bug fix", BUG_FIX), (" Non-Bug-Fix ", NON_BUG_FIX),
    ("FIX", None), ("", None), (None, None), (1, None),
])
def test_label_validation(raw, expected):
    assert normalize_label(raw) == expected


def test_user_prompt_per_input_mode():
    rendered = RenderedDiff(text="FILE: main.tf\n```diff\n+x\n```", diff_truncated=False,
                            original_diff_chars=3, submitted_diff_chars=3, files_submitted=1)
    msg_only = build_user_prompt(PROMPT_V1, "message_only", "Fix cidr", rendered)
    diff_only = build_user_prompt(PROMPT_V1, "diff_only", "Fix cidr", rendered)
    both = build_user_prompt(PROMPT_V1, "message_and_diff", "Fix cidr", rendered)

    assert "Fix cidr" in msg_only and "FILE: main.tf" not in msg_only
    assert "Fix cidr" not in diff_only and "FILE: main.tf" in diff_only
    assert "Fix cidr" in both and "FILE: main.tf" in both
    assert "(empty commit message)" in build_user_prompt(PROMPT_V1, "message_only", "  ", None)
    with pytest.raises(ValueError):
        build_user_prompt(PROMPT_V1, "diff_only", "x", None)


def test_prompt_registry_and_hash():
    assert get_prompt("v1") is PROMPT_V1
    assert len(PROMPT_V1.sha256) == 64
    assert "BUG_FIX" in PROMPT_V1.system and PROMPT_V1.version == "v1"
    with pytest.raises(ValueError):
        get_prompt("v999")


def _copy_v1(tmp_path, name="v2"):
    src = Path(__file__).resolve().parents[1] / "src" / "llm_analysis" / "prompt_templates" / "v1"
    dst = tmp_path / name
    shutil.copytree(src, dst)
    return dst


def test_prompt_loaded_from_files_and_edits_change_hash(tmp_path):
    folder = _copy_v1(tmp_path)
    copy = load_prompt(folder)
    assert copy.version == "v2" and copy.system == PROMPT_V1.system

    (folder / "system.txt").write_text(PROMPT_V1.system + "\nExtra rule.\n", encoding="utf-8")
    assert load_prompt(folder).sha256 != copy.sha256


def test_prompt_file_validation(tmp_path):
    folder = _copy_v1(tmp_path)
    (folder / "user.yaml").write_text("message_header: x\n")
    with pytest.raises(ValueError, match="missing"):
        load_prompt(folder)


def test_primary_intent_is_optional_and_stored():
    with_intent = parse_llm_response(json.dumps({**VALID, "primary_intent": "Fix wrong subnet mask"}))
    assert with_intent.parse_success and with_intent.primary_intent == "Fix wrong subnet mask"
    assert with_intent.warnings == []

    without = parse_llm_response(json.dumps(VALID))  # respostas do prompt v1
    assert without.parse_success and without.primary_intent is None and without.warnings == []


def test_v2_system_prompt_file():
    system = (Path(__file__).resolve().parents[1] / "src" / "llm_analysis" / "prompt_templates" / "v2"
              / "system.txt").read_text()
    assert system.startswith("You are a software maintenance researcher")
    assert '"primary_intent"' in system and system.rstrip().endswith("Return only the final JSON object.")


def _template_folder(tmp_path, templates_yaml):
    folder = tmp_path / "vt"
    folder.mkdir()
    (folder / "system.txt").write_text("system", encoding="utf-8")
    (folder / "user.yaml").write_text(templates_yaml, encoding="utf-8")
    return folder


def test_placeholder_templates(tmp_path):
    template = load_prompt(_template_folder(tmp_path, (
        "empty_message: '(empty)'\n"
        "templates:\n"
        "  message_and_diff: |\n"
        "    Repo {{ repo }} @ {{commit_hash}}\n"
        "    <<<M\n    {{ message }}\n    M\n"
        "    Files:\n    {{ file_list }}\n"
        "    {{ combined_diff }}\n"
        "  message_only: 'Only {{ message }}'\n"
    )))
    assert template.input_modes == ("message_only", "message_and_diff")
    rendered = RenderedDiff(text="FILE: a.tf\n```diff\n+x\n```", diff_truncated=False,
                            original_diff_chars=3, submitted_diff_chars=3, files_submitted=1)
    prompt = build_user_prompt(template, "message_and_diff", "Fix {{ repo }} typo", rendered,
                               repo="o_r", commit_hash="abc", files=["a.tf", "b.tf", "c.tf"],
                               max_listed_file_names=2)
    assert "Repo o_r @ abc" in prompt
    assert "Fix {{ repo }} typo" in prompt  # texto do commit nunca é expandido
    assert "- a.tf\n- b.tf\n- ... and 1 more" in prompt
    assert "FILE: a.tf" in prompt and prompt.endswith("\n")
    assert build_user_prompt(template, "message_only", "  ", None) == "Only (empty)\n"
    with pytest.raises(ValueError, match="no user template for diff_only"):
        build_user_prompt(template, "diff_only", "x", rendered)


@pytest.mark.parametrize("mode, placeholder", [
    ("message_only", "combined_diff"), ("message_only", "file_list"), ("diff_only", "message"),
    ("message_and_diff", "author"),
])
def test_templates_cannot_leak_withheld_evidence(tmp_path, mode, placeholder):
    folder = _template_folder(tmp_path, f"templates:\n  {mode}: 'x {{{{ {placeholder} }}}}'\n")
    with pytest.raises(ValueError, match="must not see"):
        load_prompt(folder)


def test_v2_prompt_loads():
    v2 = get_prompt("v2")
    assert v2.system.startswith("You are a software maintenance researcher")
    assert "message_and_diff" in v2.input_modes
    assert v2.sha256 != PROMPT_V1.sha256

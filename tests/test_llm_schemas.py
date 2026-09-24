import json

import pytest

from src.llm_analysis.prompts import PROMPT_V1, build_user_prompt, get_prompt
from src.llm_analysis.diff_context import RenderedDiff
from src.llm_analysis.schemas import BUG_FIX, NON_BUG_FIX, normalize_label, parse_llm_response

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
    with pytest.raises(ValueError):
        get_prompt("v999")

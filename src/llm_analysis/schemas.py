"""Data structures, label vocabulary and validation of LLM responses."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

BUG_FIX = "BUG_FIX"
NON_BUG_FIX = "NON_BUG_FIX"
LABELS = (BUG_FIX, NON_BUG_FIX)

MESSAGE_ONLY = "message_only"
DIFF_ONLY = "diff_only"
MESSAGE_AND_DIFF = "message_and_diff"
INPUT_MODES = (MESSAGE_ONLY, DIFF_ONLY, MESSAGE_AND_DIFF)

CANDIDATE_VALIDATION = "candidate_validation"
FULL_CLASSIFICATION = "full_classification"
EXPERIMENT_MODES = (CANDIDATE_VALIDATION, FULL_CLASSIFICATION)

# Value the model must put in an evidence field whose input was withheld (e.g. diff_evidence in message_only).
NOT_PROVIDED = "NOT_PROVIDED"

RESPONSE_TEXT_FIELDS = ("message_evidence", "diff_evidence", "reason", "uncertainty")
RESPONSE_FIELDS = ("label", "confidence", *RESPONSE_TEXT_FIELDS)

# Field order of each line in classifications_<input_mode>.jsonl.
RESULT_FIELDS = (
    "record_id",
    "experiment_id",
    "experiment_mode",
    "repo",
    "commit_hash",
    "input_mode",
    "provider",
    "model",
    "response_model",
    "temperature",
    "max_output_tokens",
    "prompt_version",
    "prompt_sha256",
    "timestamp",
    "label",
    "confidence",
    "reason",
    "message_evidence",
    "diff_evidence",
    "uncertainty",
    "diff_truncated",
    "original_diff_chars",
    "submitted_diff_chars",
    "files_submitted",
    "files_omitted",
    "prepared_commit_sha256",
    "user_prompt_sha256",
    "user_prompt",
    "raw_response",
    "parse_success",
    "parse_error",
    "parse_warnings",
    "error",
    "error_type",
    "attempts",
    "input_tokens",
    "output_tokens",
    "stop_reason",
    "latency_seconds",
)


@dataclass
class FileDiff:
    """One distinct patch of a commit. `also_in` lists other files that received the identical patch."""
    file: str
    patch: str
    also_in: list[str] = field(default_factory=list)


@dataclass
class PreparedCommit:
    """A commit aggregated from all its per-file miner records: the unit of classification."""
    repo: str
    commit_hash: str
    author: str | None
    date: str | None
    message: str
    head_hash: str | None
    files: list[str]
    diffs: list[FileDiff]
    files_without_patch: list[str]
    n_records: int
    duplicate_patches_removed: int
    source_files: list[str]
    content_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreparedCommit:
        values = dict(data)
        values["diffs"] = [FileDiff(**d) for d in values.get("diffs", [])]
        return cls(**values)


@dataclass
class ParsedResponse:
    label: str | None = None
    confidence: float | None = None
    message_evidence: str = ""
    diff_evidence: str = ""
    reason: str = ""
    uncertainty: str = ""
    parse_success: bool = False
    parse_error: str | None = None
    warnings: list[str] = field(default_factory=list)


_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Find the JSON object in a response. Returns (object or None, warnings)."""
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        return (obj if isinstance(obj, dict) else None), []
    except json.JSONDecodeError:
        pass

    candidates = [m.group(1) for m in _FENCED_JSON.finditer(stripped)]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start:end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj, ["response contained text around the JSON object"]
    return None, []


def normalize_label(value: Any) -> str | None:
    """Map 'BUG_FIX', 'bug fix', 'Non-Bug-Fix', ... to a canonical label; None if not a valid label."""
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[\s\-]+", "_", value.strip()).upper()
    return normalized if normalized in LABELS else None


def parse_llm_response(text: str | None) -> ParsedResponse:
    """
    Parse and validate a model response against the required schema.

    `label` (one of LABELS) and `confidence` (a number in [0, 1]) are mandatory: without them
    parse_success is False. Missing or non-string text fields are tolerated but reported in warnings.
    """
    result = ParsedResponse()
    if not text or not text.strip():
        result.parse_error = "empty response"
        return result

    obj, warnings = _extract_json_object(text)
    result.warnings.extend(warnings)
    if obj is None:
        result.parse_error = "no JSON object found in response"
        return result

    label = normalize_label(obj.get("label"))
    if label is None:
        result.parse_error = f"invalid or missing label: {obj.get('label')!r}"
        return result
    if obj["label"] != label:
        result.warnings.append(f"label normalized from {obj['label']!r}")
    result.label = label

    raw_confidence = obj.get("confidence")
    confidence: float | None = None
    if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
        confidence = float(raw_confidence)
    elif isinstance(raw_confidence, str):
        try:
            confidence = float(raw_confidence)
            result.warnings.append("confidence given as a string")
        except ValueError:
            confidence = None
    if confidence is None or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        result.parse_error = f"invalid or missing confidence: {raw_confidence!r}"
        return result
    result.confidence = confidence

    for name in RESPONSE_TEXT_FIELDS:
        value = obj.get(name)
        if value is None:
            result.warnings.append(f"missing field {name!r}")
            value = ""
        elif not isinstance(value, str):
            result.warnings.append(f"field {name!r} is not a string")
            value = json.dumps(value, ensure_ascii=False)
        setattr(result, name, value)

    extra = sorted(set(obj) - set(RESPONSE_FIELDS))
    if extra:
        result.warnings.append(f"unexpected fields: {', '.join(extra)}")

    result.parse_success = True
    return result

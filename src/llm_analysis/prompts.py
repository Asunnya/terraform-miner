"""Load versioned prompts and build user prompts."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from src.llm_analysis.diff_context import RenderedDiff
from src.llm_analysis.schemas import DIFF_ONLY, INPUT_MODES, MESSAGE_AND_DIFF, MESSAGE_ONLY
from src.llm_analysis.utils import MODULE_DIR, stable_hash

PROMPT_TEMPLATES_DIR = MODULE_DIR / "prompt_templates"
SYSTEM_FILE = "system.txt"
USER_FILE = "user.yaml"
USER_TEMPLATE_KEYS = ("message_header", "diff_header", "empty_message", "mode_notes")


@dataclass(frozen=True)
class PromptTemplate:
    """A prompt version: v1 sections or v2+ per-mode templates."""
    version: str
    system: str
    message_header: str | None = None
    diff_header: str | None = None
    empty_message: str | None = None
    mode_notes: dict[str, str] | None = None
    user_templates: dict[str, str] | None = None

    @property
    def sha256(self) -> str:
        # Skip unset fields so new fields keep old versions' hashes.
        return stable_hash({k: v for k, v in asdict(self).items() if v is not None})

    @property
    def input_modes(self) -> tuple[str, ...]:
        """Input modes this version can build a user prompt for."""
        if self.user_templates is not None:
            return tuple(m for m in INPUT_MODES if m in self.user_templates)
        return INPUT_MODES


# Placeholders each mode may use (a mode never sees the evidence it withholds).
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
MODE_PLACEHOLDERS = {
    MESSAGE_ONLY: frozenset({"repo", "commit_hash", "message"}),
    DIFF_ONLY: frozenset({"repo", "commit_hash", "file_list", "combined_diff"}),
    MESSAGE_AND_DIFF: frozenset({"repo", "commit_hash", "message", "file_list", "combined_diff"}),
}


def available_versions(templates_dir: Path = PROMPT_TEMPLATES_DIR) -> list[str]:
    return sorted(p.name for p in templates_dir.iterdir() if (p / SYSTEM_FILE).is_file())


def load_prompt(version_dir: Path) -> PromptTemplate:
    """Load and validate one prompt version folder."""
    system = (version_dir / SYSTEM_FILE).read_text(encoding="utf-8")
    if not system.strip():
        raise ValueError(f"{version_dir / SYSTEM_FILE} is empty")
    user_file = version_dir / USER_FILE
    if not user_file.is_file():
        raise ValueError(f"{user_file} is missing")
    with open(user_file, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}

    if "templates" in user:
        templates = user["templates"] or {}
        unknown_modes = sorted(set(templates) - set(INPUT_MODES))
        if unknown_modes or not templates:
            raise ValueError(f"{user_file}: templates must map input modes {INPUT_MODES} to text "
                             f"(unknown: {unknown_modes})")
        for mode, text in templates.items():
            used = set(PLACEHOLDER.findall(text))
            not_allowed = sorted(used - MODE_PLACEHOLDERS[mode])
            if not_allowed:
                raise ValueError(f"{user_file}: the {mode} template uses {not_allowed}, which that mode "
                                 f"must not see (allowed: {sorted(MODE_PLACEHOLDERS[mode])})")
        return PromptTemplate(version=version_dir.name, system=system, empty_message=user.get("empty_message"),
                              user_templates={m: templates[m] for m in INPUT_MODES if m in templates})

    missing = [k for k in USER_TEMPLATE_KEYS if k not in user]
    if missing:
        raise ValueError(f"{user_file} is missing {missing} (or a `templates` mapping)")
    missing_modes = [m for m in INPUT_MODES if m not in user["mode_notes"]]
    if missing_modes:
        raise ValueError(f"{user_file}: mode_notes is missing {missing_modes}")
    return PromptTemplate(
        version=version_dir.name,
        system=system,
        message_header=user["message_header"],
        diff_header=user["diff_header"],
        empty_message=user["empty_message"],
        mode_notes={m: user["mode_notes"][m] for m in INPUT_MODES},
    )


@lru_cache(maxsize=None)
def get_prompt(version: str, templates_dir: Path = PROMPT_TEMPLATES_DIR) -> PromptTemplate:
    version_dir = templates_dir / version
    if not (version_dir / SYSTEM_FILE).is_file():
        raise ValueError(f"Unknown prompt_version {version!r}; available: {available_versions(templates_dir)}")
    return load_prompt(version_dir)


def format_file_list(files: list[str], max_names: int) -> str:
    shown = [f"- {f}" for f in files[:max_names]]
    if len(files) > max_names:
        shown.append(f"- ... and {len(files) - max_names} more")
    return "\n".join(shown) or "(none)"


def build_user_prompt(
    template: PromptTemplate,
    input_mode: str,
    message: str,
    rendered_diff: RenderedDiff | None,
    *,
    repo: str = "",
    commit_hash: str = "",
    files: list[str] | None = None,
    max_listed_file_names: int = 100,
) -> str:
    """User prompt for one commit and mode."""
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unknown input mode {input_mode!r}")
    if input_mode not in template.input_modes:
        raise ValueError(f"Prompt {template.version} has no user template for {input_mode}")
    if input_mode in (DIFF_ONLY, MESSAGE_AND_DIFF) and rendered_diff is None:
        raise ValueError(f"{input_mode} requires a rendered diff")
    text = message.strip() or (template.empty_message or "")

    if template.user_templates is not None:
        values = {
            "repo": repo,
            "commit_hash": commit_hash,
            "message": text,
            "file_list": format_file_list(files or [], max_listed_file_names),
            "combined_diff": rendered_diff.text if rendered_diff else "",
        }
        # Single pass: {{ }} inside commit text is not expanded.
        prompt = PLACEHOLDER.sub(lambda m: values[m.group(1)], template.user_templates[input_mode])
        return prompt if prompt.endswith("\n") else prompt + "\n"

    sections = [template.mode_notes[input_mode]]
    if input_mode in (MESSAGE_ONLY, MESSAGE_AND_DIFF):
        sections.append(f"{template.message_header}\n{text}")
    if input_mode in (DIFF_ONLY, MESSAGE_AND_DIFF):
        sections.append(f"{template.diff_header}\n{rendered_diff.text}")
    return "\n\n".join(sections) + "\n"

"""Deterministic rendering of a commit's diffs within the context budget."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.llm_analysis.schemas import FileDiff

# Part of the experiment id: bump when the rendering changes.
DIFF_FORMAT_VERSION = "1"

HUNK_PREFIX = "@@"


@dataclass(frozen=True)
class ContextLimits:
    max_diff_chars: int
    max_chars_per_file: int
    max_files_per_commit: int
    max_listed_file_names: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ContextLimits:
        limits = cls(
            max_diff_chars=int(cfg["max_diff_chars"]),
            max_chars_per_file=int(cfg["max_chars_per_file"]),
            max_files_per_commit=int(cfg["max_files_per_commit"]),
            max_listed_file_names=int(cfg["max_listed_file_names"]),
        )
        for name, value in limits.__dict__.items():
            if value <= 0:
                raise ValueError(f"context.{name} must be a positive integer, got {value}")
        return limits


@dataclass
class RenderedDiff:
    text: str
    diff_truncated: bool
    original_diff_chars: int
    submitted_diff_chars: int
    files_submitted: int
    files_omitted: list[str] = field(default_factory=list)
    files_truncated: list[str] = field(default_factory=list)


def allocate_budget(sizes: list[int], total: int, per_file: int) -> list[int]:
    """Water-filling split of `total` chars, each file capped at `per_file`."""
    caps = [min(size, per_file) for size in sizes]
    allocation = [0] * len(sizes)
    remaining = total
    order = sorted(range(len(sizes)), key=lambda i: (caps[i], i))
    for position, i in enumerate(order):
        share = remaining // (len(order) - position)
        allocation[i] = min(caps[i], share)
        remaining -= allocation[i]
    return allocation


def split_hunks(patch: str) -> list[str]:
    """Split a patch at each `@@` line."""
    chunks: list[str] = []
    current: list[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith(HUNK_PREFIX) and current:
            chunks.append("".join(current))
            current = []
        current.append(line)
    if current:
        chunks.append("".join(current))
    return chunks


def truncate_patch(patch: str, limit: int) -> tuple[str, int, int]:
    """Cut at hunk, then line boundaries. Returns (text, whole hunks kept, total hunks)."""
    hunks = split_hunks(patch)
    if len(patch) <= limit:
        return patch, len(hunks), len(hunks)

    kept: list[str] = []
    used = 0
    full_hunks = 0
    for hunk in hunks:
        if used + len(hunk) <= limit:
            kept.append(hunk)
            used += len(hunk)
            full_hunks += 1
            continue
        for line in hunk.splitlines(keepends=True):
            if used + len(line) > limit:
                break
            kept.append(line)
            used += len(line)
        break

    if not kept and limit > 0:
        kept.append(patch[:limit])
    return "".join(kept), full_hunks, len(hunks)


def _format_names(names: list[str], max_names: int) -> str:
    shown = ", ".join(names[:max_names])
    hidden = len(names) - max_names
    return f"{shown}, ... and {hidden} more" if hidden > 0 else shown


def render_diffs(
    diffs: list[FileDiff],
    files_without_patch: list[str],
    limits: ContextLimits,
) -> RenderedDiff:
    shown = diffs[:limits.max_files_per_commit]
    omitted = diffs[limits.max_files_per_commit:]
    budgets = allocate_budget([len(d.patch) for d in shown], limits.max_diff_chars, limits.max_chars_per_file)

    blocks: list[str] = []
    files_truncated: list[str] = []
    submitted = 0
    for diff, budget in zip(shown, budgets):
        kept, full_hunks, total_hunks = truncate_patch(diff.patch, budget)
        submitted += len(kept)
        header = f"FILE: {diff.file}"
        if diff.also_in:
            header += (f"\n(the identical patch was also applied to: "
                       f"{_format_names(diff.also_in, limits.max_listed_file_names)})")
        body = kept if kept.endswith("\n") or not kept else kept + "\n"
        block = f"{header}\n```diff\n{body}```"
        if len(kept) < len(diff.patch):
            files_truncated.append(diff.file)
            block += (f"\nNOTE: patch truncated to fit the context budget: showing {len(kept)} of "
                      f"{len(diff.patch)} characters ({full_hunks} of {total_hunks} hunks complete).")
        blocks.append(block)

    omitted_names = [d.file for d in omitted]
    if omitted_names:
        blocks.append(
            f"FILES CHANGED BUT NOT SHOWN (limit of {limits.max_files_per_commit} patches per commit): "
            f"{_format_names(omitted_names, limits.max_listed_file_names)}"
        )
    if files_without_patch:
        blocks.append(
            "FILES CHANGED WITHOUT A TEXTUAL PATCH (e.g. binary, mode-only or pure rename): "
            f"{_format_names(files_without_patch, limits.max_listed_file_names)}"
        )
    if not diffs and not files_without_patch:
        blocks.append("(no Terraform changes available for this commit)")

    return RenderedDiff(
        text="\n\n".join(blocks),
        diff_truncated=bool(files_truncated or omitted_names),
        original_diff_chars=sum(len(d.patch) for d in diffs),
        submitted_diff_chars=submitted,
        files_submitted=len(shown),
        files_omitted=omitted_names,
        files_truncated=files_truncated,
    )

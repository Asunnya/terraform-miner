"""Stage 01 scope filter: keep commits with in-scope Terraform infrastructure changes."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import Any, Iterable

from src.llm_analysis.preparation import commit_content_hash
from src.llm_analysis.schemas import FileDiff, PreparedCommit

# File scopes.
INFRASTRUCTURE = "INFRASTRUCTURE"
TEST = "TEST"
EXAMPLE = "EXAMPLE"
DOCUMENTATION = "DOCUMENTATION"
VENDOR = "VENDOR"
GENERATED = "GENERATED"
REPOSITORY_METADATA = "REPOSITORY_METADATA"
# config.yaml key -> scope
SCOPE_KEYS = {"test": TEST, "example": EXAMPLE, "documentation": DOCUMENTATION, "vendor": VENDOR,
              "generated": GENERATED, "repository_metadata": REPOSITORY_METADATA}

# Commit states.
IN_SCOPE = "IN_SCOPE"
OUT_OF_SCOPE = "OUT_OF_SCOPE"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
STATES = (IN_SCOPE, OUT_OF_SCOPE, INSUFFICIENT_DATA)

# Reasons.
ELIGIBLE = "eligible"
GENERIC_MERGE = "generic_merge"
ONLY_EXCLUDED_FILES = "only_excluded_files"
NO_IN_SCOPE_TERRAFORM_FILES = "no_in_scope_terraform_files"
NO_IN_SCOPE_PATCH = "no_in_scope_patch"
ONLY_SCOPE_REASON = {TEST: "only_test_files", EXAMPLE: "only_example_files",
                     DOCUMENTATION: "only_documentation_files", VENDOR: "only_vendor_files",
                     GENERATED: "only_generated_files", REPOSITORY_METADATA: "only_repository_metadata_files"}
REASONS = (ELIGIBLE, GENERIC_MERGE, *ONLY_SCOPE_REASON.values(), ONLY_EXCLUDED_FILES,
           NO_IN_SCOPE_TERRAFORM_FILES, NO_IN_SCOPE_PATCH)


@dataclass(frozen=True)
class EligibilityConfig:
    enabled: bool
    exclude_generic_merges: bool
    merge_patterns: tuple[re.Pattern, ...]
    scope_segments: dict[str, str]  # lowercase directory segment -> scope
    require_in_scope_patch: bool

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> EligibilityConfig:
        segments: dict[str, str] = {}
        for key, names in (cfg.get("excluded_scope_segments") or {}).items():
            if key not in SCOPE_KEYS:
                raise ValueError(f"eligibility.excluded_scope_segments: unknown scope {key!r}; use {sorted(SCOPE_KEYS)}")
            for name in names:
                if name.lower() in segments:
                    raise ValueError(f"eligibility: segment {name!r} is listed under two scopes")
                segments[name.lower()] = SCOPE_KEYS[key]
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            exclude_generic_merges=bool(cfg.get("exclude_generic_merges", True)),
            merge_patterns=tuple(re.compile(p, re.IGNORECASE) for p in cfg.get("merge_patterns") or ()),
            scope_segments=segments,
            require_in_scope_patch=bool(cfg.get("require_in_scope_patch", True)),
        )

    def snapshot(self) -> dict[str, Any]:
        """Settings as plain data, for prepare_stats.json."""
        return {
            "enabled": self.enabled,
            "exclude_generic_merges": self.exclude_generic_merges,
            "merge_patterns": [p.pattern for p in self.merge_patterns],
            "scope_segments": dict(sorted(self.scope_segments.items())),
            "require_in_scope_patch": self.require_in_scope_patch,
        }


def file_scope(path: str, config: EligibilityConfig) -> str:
    """Scope of a file: its outermost excluded directory segment, else INFRASTRUCTURE."""
    for segment in PurePosixPath(path).parts[:-1]:
        scope = config.scope_segments.get(segment.lower())
        if scope:
            return scope
    return INFRASTRUCTURE


def is_generic_merge(message: str, config: EligibilityConfig) -> bool:
    return any(p.match(message or "") for p in config.merge_patterns)


@dataclass
class EligibilityDecision:
    commit: PreparedCommit
    state: str
    reason: str
    in_scope_files: list[str]
    excluded_files: list[dict[str, str]]
    prepared: PreparedCommit | None = None  # filtered commit, only when IN_SCOPE

    @property
    def mixed(self) -> bool:
        return bool(self.in_scope_files and self.excluded_files)

    def audit_record(self) -> dict[str, Any]:
        lines = (self.commit.message or "").strip().splitlines()
        return {
            "repo": self.commit.repo,
            "commit_hash": self.commit.commit_hash,
            "eligibility_state": self.state,
            "eligibility_reason": self.reason,
            "message_first_line": lines[0] if lines else "",
            "original_files": self.commit.files,
            "in_scope_files": self.in_scope_files,
            "excluded_files": self.excluded_files,
        }


def _filter_diffs(diffs: list[FileDiff], in_scope: set[str]) -> list[FileDiff]:
    """Keep patches touching an in-scope path; canonical file = first in-scope path."""
    kept = []
    for diff in diffs:
        paths = sorted(p for p in (diff.file, *diff.also_in) if p in in_scope)
        if paths:
            kept.append(FileDiff(file=paths[0], patch=diff.patch, also_in=paths[1:]))
    kept.sort(key=lambda d: (d.file, d.patch))
    return kept


def _out_of_scope_reason(excluded: list[dict[str, str]]) -> str:
    scopes = {e["scope"] for e in excluded}
    if not scopes:
        return NO_IN_SCOPE_TERRAFORM_FILES
    return ONLY_SCOPE_REASON[scopes.pop()] if len(scopes) == 1 else ONLY_EXCLUDED_FILES


def assess_commit(commit: PreparedCommit, config: EligibilityConfig) -> EligibilityDecision:
    scopes = {f: file_scope(f, config) for f in commit.files}
    in_scope = sorted(f for f, s in scopes.items() if s == INFRASTRUCTURE)
    excluded = [{"file": f, "scope": s} for f, s in sorted(scopes.items()) if s != INFRASTRUCTURE]

    def decision(state: str, reason: str, prepared: PreparedCommit | None = None) -> EligibilityDecision:
        return EligibilityDecision(commit, state, reason, in_scope, excluded, prepared)

    if config.exclude_generic_merges and is_generic_merge(commit.message, config):
        return decision(OUT_OF_SCOPE, GENERIC_MERGE)
    if not in_scope:
        return decision(OUT_OF_SCOPE, _out_of_scope_reason(excluded))

    diffs = _filter_diffs(commit.diffs, set(in_scope))
    if not diffs and config.require_in_scope_patch:
        return decision(INSUFFICIENT_DATA, NO_IN_SCOPE_PATCH)

    prepared = replace(
        commit,
        files=in_scope,
        diffs=diffs,
        files_without_patch=[f for f in commit.files_without_patch if f in set(in_scope)],
        excluded_files=excluded,
    )
    prepared.content_sha256 = commit_content_hash(prepared)
    return decision(IN_SCOPE, ELIGIBLE, prepared)


@dataclass
class EligibilityResult:
    decisions: list[EligibilityDecision] = field(default_factory=list)

    @property
    def eligible(self) -> list[PreparedCommit]:
        return [d.prepared for d in self.decisions if d.state == IN_SCOPE]

    def audit_records(self) -> list[dict[str, Any]]:
        return [d.audit_record() for d in self.decisions if d.state != IN_SCOPE]

    def stats(self) -> dict[str, Any]:
        states = Counter(d.state for d in self.decisions)
        reasons = Counter(d.reason for d in self.decisions)
        excluded_scopes = Counter(e["scope"] for d in self.decisions for e in d.excluded_files)
        stats = {
            "commits_before_eligibility": len(self.decisions),
            "commits_in_scope": states[IN_SCOPE],
            "commits_out_of_scope": states[OUT_OF_SCOPE],
            "commits_insufficient_data": states[INSUFFICIENT_DATA],
            "commits_after_eligibility": states[IN_SCOPE],
            "by_reason": {r: reasons[r] for r in REASONS if r != ELIGIBLE},
            "mixed_scope_commits": sum(1 for d in self.decisions if d.state == IN_SCOPE and d.mixed),
            "excluded_file_count": sum(excluded_scopes.values()),
            "excluded_files_by_scope": dict(sorted(excluded_scopes.items())),
            "retained_in_scope_file_count": sum(len(d.prepared.files) for d in self.decisions if d.prepared),
        }
        total = stats["commits_in_scope"] + stats["commits_out_of_scope"] + stats["commits_insufficient_data"]
        if total != stats["commits_before_eligibility"]:
            raise AssertionError("eligibility counts do not add up to the commits assessed")
        return stats


def filter_commits_by_eligibility(commits: Iterable[PreparedCommit], config: EligibilityConfig) -> EligibilityResult:
    """Assess every commit; with filtering disabled, every commit is IN_SCOPE unchanged."""
    result = EligibilityResult()
    for commit in commits:
        if config.enabled:
            result.decisions.append(assess_commit(commit, config))
        else:
            result.decisions.append(EligibilityDecision(commit, IN_SCOPE, ELIGIBLE, list(commit.files), [], commit))
    return result

"""Stage 01: group the per-file miner records by commit."""
from __future__ import annotations

import glob
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.llm_analysis.schemas import FileDiff, PreparedCommit
from src.llm_analysis.utils import read_jsonl, stable_hash

logger = logging.getLogger(__name__)

TERRAFORM_EXTENSION = ".tf"
# Must match across a commit's records (mismatches are counted).
COMMIT_LEVEL_FIELDS = ("author", "date", "message", "head_hash")


@dataclass
class PreparationStats:
    input_files: int = 0
    records_read: int = 0
    records_skipped_missing_key: int = 0
    records_skipped_non_tf: int = 0
    duplicate_records: int = 0
    empty_patches: int = 0
    duplicate_patches_removed: int = 0
    commits: int = 0
    commits_without_any_patch: int = 0
    field_conflicts: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["field_conflicts"] = dict(self.field_conflicts)
        return data


def find_input_files(pattern: str | Path) -> list[Path]:
    """Files matching the glob, sorted (deterministic order)."""
    return sorted(Path(p) for p in glob.glob(str(pattern), recursive=True) if Path(p).is_file())


def iter_miner_records(paths: Iterable[Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in paths:
        for _, record in read_jsonl(path):
            yield path, record


def is_terraform_file(path: str | None) -> bool:
    return bool(path) and path.lower().endswith(TERRAFORM_EXTENSION)


def commit_content_hash(commit: PreparedCommit) -> str:
    """Hash of the prompt-relevant content (detects stale results)."""
    return stable_hash({
        "repo": commit.repo,
        "commit_hash": commit.commit_hash,
        "message": commit.message,
        "files": commit.files,
        "files_without_patch": commit.files_without_patch,
        "diffs": [[d.file, d.patch, d.also_in] for d in commit.diffs],
    })


class _CommitGroup:
    def __init__(self, record: dict[str, Any]):
        self.first = record
        self.file_patches: list[tuple[str, str]] = []
        self.seen: set[tuple[str, str]] = set()
        self.source_files: set[str] = set()
        self.n_records = 0


def aggregate_commits(
    records: Iterable[tuple[Path, dict[str, Any]]],
    stats: PreparationStats | None = None,
) -> list[PreparedCommit]:
    """Group by (repo, commit_hash), dropping empty and duplicate patches."""
    stats = stats if stats is not None else PreparationStats()
    groups: dict[tuple[str, str], _CommitGroup] = {}

    for source, record in records:
        stats.records_read += 1
        repo, commit_hash = record.get("repo"), record.get("commit_hash")
        if not repo or not commit_hash:
            stats.records_skipped_missing_key += 1
            continue
        file_path = record.get("file")
        if not is_terraform_file(file_path):
            stats.records_skipped_non_tf += 1
            continue

        key = (repo, commit_hash)
        group = groups.get(key)
        if group is None:
            group = groups[key] = _CommitGroup(record)
        else:
            for name in COMMIT_LEVEL_FIELDS:
                if record.get(name) != group.first.get(name):
                    stats.field_conflicts[name] += 1
        group.n_records += 1
        group.source_files.add(str(source))

        patch = record.get("patch") or ""
        if (file_path, patch) in group.seen:
            stats.duplicate_records += 1
            continue
        group.seen.add((file_path, patch))
        group.file_patches.append((file_path, patch))

    commits = []
    for (repo, commit_hash), group in sorted(groups.items()):
        commit = _build_commit(repo, commit_hash, group, stats)
        commits.append(commit)
    stats.commits = len(commits)
    return commits


def _build_commit(repo: str, commit_hash: str, group: _CommitGroup, stats: PreparationStats) -> PreparedCommit:
    files = sorted({f for f, _ in group.file_patches})
    patch_to_files: dict[str, list[str]] = {}
    non_empty = 0
    for file_path, patch in group.file_patches:
        if not patch.strip():
            stats.empty_patches += 1
            continue
        non_empty += 1
        patch_to_files.setdefault(patch, []).append(file_path)

    diffs = []
    for patch, paths in patch_to_files.items():
        paths = sorted(set(paths))
        diffs.append(FileDiff(file=paths[0], patch=patch, also_in=paths[1:]))
    diffs.sort(key=lambda d: (d.file, d.patch))

    files_with_patch = {d.file for d in diffs} | {p for d in diffs for p in d.also_in}
    duplicates_removed = non_empty - len(diffs)
    stats.duplicate_patches_removed += duplicates_removed
    if not diffs:
        stats.commits_without_any_patch += 1

    first = group.first
    commit = PreparedCommit(
        repo=repo,
        commit_hash=commit_hash,
        author=first.get("author"),
        date=first.get("date"),
        message=first.get("message") or "",
        head_hash=first.get("head_hash"),
        files=files,
        diffs=diffs,
        files_without_patch=[f for f in files if f not in files_with_patch],
        n_records=group.n_records,
        duplicate_patches_removed=duplicates_removed,
        source_files=sorted(group.source_files),
    )
    commit.content_sha256 = commit_content_hash(commit)
    return commit


def sample_commits(commits: list[PreparedCommit], sample_size: int | None, seed: int) -> list[PreparedCommit]:
    """Seeded simple random sample, kept in sorted order."""
    if sample_size is None or sample_size >= len(commits):
        return commits
    chosen = random.Random(seed).sample(range(len(commits)), sample_size)
    return [commits[i] for i in sorted(chosen)]


def keyword_match_fraction(commits: list[PreparedCommit], keywords: Iterable[str]) -> float | None:
    """Share of messages containing a miner keyword."""
    keywords = [k.lower() for k in keywords if k]
    if not commits or not keywords:
        return None
    matches = sum(1 for c in commits if any(k in c.message.lower() for k in keywords))
    return matches / len(commits)

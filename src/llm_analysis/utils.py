"""Shared helpers: configuration, paths, hashing and JSONL/CSV I/O."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
DEFAULT_CONFIG_PATH = MODULE_DIR / "config.yaml"

logger = logging.getLogger("llm_analysis")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config. A relative path is tried against the CWD, the repo root and this module."""
    if path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        candidates = [Path(path), REPO_ROOT / path, MODULE_DIR / path]
        config_path = next((p for p in candidates if p.is_file()), None)
        if config_path is None:
            raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    """Resolve a config path. Relative paths are relative to the repository root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def cli_or_config_path(cli_value: str | None, config_value: str | Path | None) -> Path | None:
    """A path given on the command line is relative to the CWD; one from config.yaml, to the repo root."""
    if cli_value:
        return resolve_path(cli_value, base=Path.cwd())
    return resolve_path(config_value) if config_value else None


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization used for hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """
    Yield (line_number, record) for each JSON object in the file. Blank lines are skipped;
    malformed lines (e.g. a line cut short by an interrupted run) are logged and skipped.
    """
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON at %s:%d: %s", path, lineno, e)
                continue
            if isinstance(record, dict):
                yield lineno, record
            else:
                logger.warning("Skipping non-object JSON at %s:%d", path, lineno)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

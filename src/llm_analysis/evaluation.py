"""
Stage 03 logic: descriptive summaries, cross-mode disagreements and (optional) gold-label evaluation.

Supervised metrics are only computed when a gold-label file is supplied. The keyword-based miner
selection is never used as ground truth. In candidate_validation experiments every metric is
conditional on the keyword-prefiltered population and is labeled as such in the output.
"""
from __future__ import annotations

import csv
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.llm_analysis.schemas import (
    BUG_FIX,
    CANDIDATE_VALIDATION,
    DIFF_ONLY,
    INPUT_MODES,
    LABELS,
    MESSAGE_AND_DIFF,
    MESSAGE_ONLY,
    NON_BUG_FIX,
    normalize_label,
)
from src.llm_analysis.utils import read_jsonl

PARSE_FAILURE = "PARSE_FAILURE"
API_FAILURE = "API_FAILURE"
CONFIDENCE_BIN_WIDTH = 0.1
# Pairs compared in disagreements.csv (spec: message_only vs. joint, diff_only vs. joint).
DISAGREEMENT_PAIRS = ((MESSAGE_ONLY, MESSAGE_AND_DIFF), (DIFF_ONLY, MESSAGE_AND_DIFF))

SCOPE_NOTES = {
    CANDIDATE_VALIDATION: (
        "candidate_validation: the input corpus was pre-filtered by the miner's bug-fix keywords. "
        "Label distributions estimate the precision of that keyword heuristic (share of candidates "
        "that are real bug fixes). Any recall, accuracy, F1, specificity or MCC reported here is "
        "conditional on keyword-selected commits and is NOT representative of arbitrary Terraform commits."
    ),
    "full_classification": (
        "full_classification: metrics describe the supplied corpus. They generalize to arbitrary "
        "Terraform commits only if that corpus was collected without keyword pre-filtering."
    ),
}

_COMMIT_HASH_IN_URL = re.compile(r"/commit/([0-9a-fA-F]{7,40})")
_HEX_HASH = re.compile(r"^[0-9a-fA-F]{7,40}$")
GOLD_TRUE_VALUES = {"true", "1", "yes", "y", "bug_fix", "bugfix"}
GOLD_FALSE_VALUES = {"false", "0", "no", "n", "non_bug_fix", "not_bug_fix"}
GOLD_COMMIT_COLUMNS = ("commit_hash", "commit_sha", "sha", "commit_url")
GOLD_LABEL_COLUMNS = ("gold_label", "label", "is_bugfix", "is_bug_fix")


# ---------------------------------------------------------------- loading results

def load_experiment_results(output_dir: Path, experiment_id: str,
                            input_modes: Iterable[str] = INPUT_MODES) -> dict[str, dict[tuple[str, str], dict]]:
    """{input_mode: {(repo, commit_hash): latest record}} for one experiment."""
    results: dict[str, dict[tuple[str, str], dict]] = {}
    for mode in input_modes:
        path = output_dir / f"classifications_{mode}.jsonl"
        latest: dict[tuple[str, str], dict] = {}
        if path.exists():
            for _, rec in read_jsonl(path):
                if rec.get("experiment_id") == experiment_id and rec.get("input_mode") == mode:
                    latest[(rec["repo"], rec["commit_hash"])] = rec
        results[mode] = latest
    return results


def outcome(record: dict[str, Any]) -> str:
    """BUG_FIX / NON_BUG_FIX, or API_FAILURE / PARSE_FAILURE."""
    if record.get("error"):
        return API_FAILURE
    if not record.get("parse_success"):
        return PARSE_FAILURE
    return record["label"]


# ---------------------------------------------------------------- descriptive summary

def confidence_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) > 1 else [ordered[0]] * 3
    n_bins = round(1 / CONFIDENCE_BIN_WIDTH)
    histogram = Counter(min(int(v / CONFIDENCE_BIN_WIDTH), n_bins - 1) for v in ordered)
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 4),
        "min": ordered[0],
        "q1": round(quartiles[0], 4),
        "median": round(quartiles[1], 4),
        "q3": round(quartiles[2], 4),
        "max": ordered[-1],
        "histogram": {
            f"[{b * CONFIDENCE_BIN_WIDTH:.1f}, {(b + 1) * CONFIDENCE_BIN_WIDTH:.1f}{']' if b == n_bins - 1 else ')'}":
                histogram.get(b, 0)
            for b in range(n_bins)
        },
    }


def summarize_mode(records: dict[tuple[str, str], dict]) -> dict[str, Any]:
    outcomes = Counter(outcome(r) for r in records.values())
    parsed = [r for r in records.values() if outcome(r) in LABELS]
    n_parsed = len(parsed)
    return {
        "unique_commits": len(records),
        "bug_fix": outcomes[BUG_FIX],
        "non_bug_fix": outcomes[NON_BUG_FIX],
        "parse_failures": outcomes[PARSE_FAILURE],
        "api_failures": outcomes[API_FAILURE],
        "bug_fix_share_of_parsed": round(outcomes[BUG_FIX] / n_parsed, 4) if n_parsed else None,
        "distribution": {k: outcomes[k] for k in (*LABELS, PARSE_FAILURE, API_FAILURE)},
        "truncated_commits": sum(1 for r in records.values() if r.get("diff_truncated")),
        "confidence": confidence_distribution([r["confidence"] for r in parsed]),
        "confidence_by_label": {
            label: confidence_distribution([r["confidence"] for r in parsed if r["label"] == label])
            for label in LABELS
        },
        "input_tokens": sum(r.get("input_tokens") or 0 for r in records.values()),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in records.values()),
    }


def cross_mode_comparison(results: dict[str, dict[tuple[str, str], dict]]) -> dict[str, Any]:
    modes = [m for m in INPUT_MODES if results.get(m)]
    all_keys = set().union(*(results[m].keys() for m in modes)) if modes else set()
    labeled_in_all = [k for k in all_keys if all(outcome(results[m].get(k, {"error": "missing"})) in LABELS
                                                  for m in modes)]
    differing = [k for k in labeled_in_all if len({results[m][k]["label"] for m in modes}) > 1]
    pairwise = {}
    for a, b in DISAGREEMENT_PAIRS:
        if a in modes and b in modes:
            pairwise[f"{a}_vs_{b}"] = sum(1 for k in labeled_in_all
                                          if results[a][k]["label"] != results[b][k]["label"])
    return {
        "modes_compared": modes,
        "unique_commits_any_mode": len(all_keys),
        "commits_labeled_in_all_modes": len(labeled_in_all),
        "commits_classified_differently_across_modes": len(differing),
        "pairwise_disagreements": pairwise,
    }


def classification_count_rows(results: dict[str, dict[tuple[str, str], dict]]) -> list[dict[str, Any]]:
    rows = []
    for mode, records in results.items():
        counts = Counter(outcome(r) for r in records.values())
        total = len(records)
        for label in (*LABELS, PARSE_FAILURE, API_FAILURE):
            rows.append({"input_mode": mode, "label": label, "count": counts[label],
                         "fraction": round(counts[label] / total, 4) if total else None})
    return rows


def error_records(results: dict[str, dict[tuple[str, str], dict]]) -> list[dict[str, Any]]:
    keys = ("record_id", "repo", "commit_hash", "input_mode", "timestamp", "error_type", "error",
            "parse_error", "stop_reason", "attempts", "raw_response")
    return [
        {"failure": outcome(r), **{k: r.get(k) for k in keys}}
        for mode in results for r in results[mode].values()
        if outcome(r) in (PARSE_FAILURE, API_FAILURE)
    ]


def first_line(text: str | None) -> str:
    lines = (text or "").strip().splitlines()
    return lines[0] if lines else ""


def disagreement_rows(results: dict[str, dict[tuple[str, str], dict]], messages: dict[tuple[str, str], str],
                      gold: dict[str, str] | None = None) -> list[dict[str, Any]]:
    rows = []
    joint = results.get(MESSAGE_AND_DIFF, {})
    for key in sorted(joint):
        labels = {m: outcome(results[m][key]) if key in results.get(m, {}) else None for m in INPUT_MODES}
        kinds = [f"{a}!={b}" for a, b in DISAGREEMENT_PAIRS
                 if labels[a] in LABELS and labels[b] in LABELS and labels[a] != labels[b]]
        if not kinds:
            continue
        row: dict[str, Any] = {
            "repo": key[0],
            "commit_hash": key[1],
            "message_first_line": first_line(messages.get(key)),
            "disagreement": ";".join(kinds),
            "gold_label": (gold or {}).get(key[1], ""),
            "diff_truncated": joint[key].get("diff_truncated"),
        }
        for mode in INPUT_MODES:
            rec = results.get(mode, {}).get(key, {})
            row[f"label_{mode}"] = labels[mode] or ""
            row[f"confidence_{mode}"] = rec.get("confidence")
            row[f"reason_{mode}"] = rec.get("reason") or ""
        rows.append(row)
    return rows


DISAGREEMENT_FIELDS = [
    "repo", "commit_hash", "message_first_line", "disagreement", "gold_label", "diff_truncated",
    *[f"{prefix}_{mode}" for mode in INPUT_MODES for prefix in ("label", "confidence", "reason")],
]


# ---------------------------------------------------------------- gold labels and metrics

@dataclass
class GoldLabels:
    labels: dict[str, str] = field(default_factory=dict)  # commit_hash -> label
    rows_read: int = 0
    rows_without_commit: int = 0
    rows_with_unusable_label: int = 0
    conflicting_commits: list[str] = field(default_factory=list)

    def stats(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("labels")
        data["usable_labels"] = len(self.labels)
        data["label_counts"] = dict(Counter(self.labels.values()))
        return data


def normalize_gold_label(value: Any) -> str | None:
    """BUG_FIX/true/1/yes -> BUG_FIX; NON_BUG_FIX/false/0/no -> NON_BUG_FIX; anything else (MAYBE, blank) -> None."""
    if value is None:
        return None
    text = str(value).strip()
    label = normalize_label(text)
    if label:
        return label
    lowered = text.lower().replace(" ", "_").replace("-", "_")
    if lowered in GOLD_TRUE_VALUES:
        return BUG_FIX
    if lowered in GOLD_FALSE_VALUES:
        return NON_BUG_FIX
    return None


def extract_commit_hash(value: Any) -> str | None:
    """Accept a bare SHA, a GitHub commit URL, or a spreadsheet =HYPERLINK("...commit/<sha>") cell."""
    if not value:
        return None
    text = str(value).strip()
    match = _COMMIT_HASH_IN_URL.search(text)
    if match:
        return match.group(1).lower()
    return text.lower() if _HEX_HASH.match(text) else None


def _pick_column(columns: Iterable[str], preferred: str | None, candidates: tuple[str, ...]) -> str | None:
    by_lower = {c.strip().lower(): c for c in columns if c}
    if preferred:
        return by_lower.get(preferred.lower())
    return next((by_lower[c] for c in candidates if c in by_lower), None)


def load_gold_labels(path: Path, label_column: str | None = None, commit_column: str | None = None) -> GoldLabels:
    """
    Load a single, adjudicated gold-label file (.csv or .jsonl). Rows are matched to classifications by
    commit SHA. Commits with contradictory labels in the file are excluded and reported.
    """
    if path.suffix.lower() == ".jsonl":
        rows = [r for _, r in read_jsonl(path)]
    else:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    gold = GoldLabels(rows_read=len(rows))
    if not rows:
        return gold
    columns = list(rows[0].keys())
    commit_col = _pick_column(columns, commit_column, GOLD_COMMIT_COLUMNS)
    label_col = _pick_column(columns, label_column, GOLD_LABEL_COLUMNS)
    if commit_col is None or label_col is None:
        raise ValueError(f"Gold file {path} needs a commit column {GOLD_COMMIT_COLUMNS} and a label column "
                         f"{GOLD_LABEL_COLUMNS} (found {columns}); set evaluation.gold_*_column in config.yaml")

    seen: dict[str, set[str]] = {}
    for row in rows:
        commit_hash = extract_commit_hash(row.get(commit_col))
        if commit_hash is None:
            gold.rows_without_commit += 1
            continue
        label = normalize_gold_label(row.get(label_col))
        if label is None:
            gold.rows_with_unusable_label += 1
            continue
        seen.setdefault(commit_hash, set()).add(label)
    for commit_hash, labels in seen.items():
        if len(labels) == 1:
            gold.labels[commit_hash] = labels.pop()
        else:
            gold.conflicting_commits.append(commit_hash)
    gold.conflicting_commits.sort()
    return gold


def match_gold(gold: dict[str, str], commit_hashes: Iterable[str]) -> dict[str, str]:
    """Map full commit hashes to gold labels; abbreviated gold SHAs match when the prefix is unique."""
    commit_hashes = list(commit_hashes)
    matched = {}
    for gold_hash, label in gold.items():
        if len(gold_hash) == 40:
            candidates = [h for h in commit_hashes if h.lower() == gold_hash]
        else:
            candidates = [h for h in commit_hashes if h.lower().startswith(gold_hash)]
        if len(candidates) == 1:
            matched[candidates[0]] = label
    return matched


def binary_metrics(gold: list[str], predicted: list[str], positive: str = BUG_FIX) -> dict[str, Any]:
    """Confusion matrix, precision, recall, F1 and MCC for `positive`. Undefined ratios are None."""
    tp = sum(1 for g, p in zip(gold, predicted) if g == positive and p == positive)
    fp = sum(1 for g, p in zip(gold, predicted) if g != positive and p == positive)
    fn = sum(1 for g, p in zip(gold, predicted) if g == positive and p != positive)
    tn = sum(1 for g, p in zip(gold, predicted) if g != positive and p != positive)

    def ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = round((tp * tn - fp * fn) / mcc_den, 4) if mcc_den else None
    return {
        "positive_label": positive,
        "n": len(gold),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }


def evaluate_against_gold(results: dict[str, dict[tuple[str, str], dict]], gold: GoldLabels,
                          experiment_mode: str) -> dict[str, Any]:
    """
    Per-mode metrics on the PAIRED set (commits with a gold label and a parsed label in every mode), so
    the three input modes are compared on exactly the same commits. Per-mode metrics on every gold commit
    each mode could label are reported too ("unpaired"), with coverage counts.
    """
    modes = [m for m in INPUT_MODES if results.get(m)]
    all_hashes = {k[1] for m in modes for k in results[m]}
    matched = match_gold(gold.labels, all_hashes)

    def labeled(mode: str) -> dict[str, str]:
        return {k[1]: r["label"] for k, r in results[mode].items() if outcome(r) in LABELS}

    per_mode_labels = {m: labeled(m) for m in modes}
    paired = sorted(h for h in matched if all(h in per_mode_labels[m] for m in modes))
    evaluation: dict[str, Any] = {
        "scope_note": SCOPE_NOTES.get(experiment_mode, ""),
        "gold_file_stats": gold.stats(),
        "gold_commits_matched": len(matched),
        "paired_commits": len(paired),
        "paired": {},
        "unpaired": {},
    }
    for mode in modes:
        evaluation["paired"][mode] = binary_metrics([matched[h] for h in paired],
                                                    [per_mode_labels[mode][h] for h in paired])
        hashes = sorted(h for h in matched if h in per_mode_labels[mode])
        evaluation["unpaired"][mode] = {
            **binary_metrics([matched[h] for h in hashes], [per_mode_labels[mode][h] for h in hashes]),
            "gold_commits_without_parsed_label": len(matched) - len(hashes),
        }
    if experiment_mode == CANDIDATE_VALIDATION and matched:
        evaluation["keyword_heuristic_precision_gold"] = round(
            sum(1 for label in matched.values() if label == BUG_FIX) / len(matched), 4)
    return evaluation


def summary_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mode, data in summary["by_input_mode"].items():
        conf = data["confidence"]
        row = {
            "input_mode": mode,
            "unique_commits": data["unique_commits"],
            "bug_fix": data["bug_fix"],
            "non_bug_fix": data["non_bug_fix"],
            "parse_failures": data["parse_failures"],
            "api_failures": data["api_failures"],
            "bug_fix_share_of_parsed": data["bug_fix_share_of_parsed"],
            "truncated_commits": data["truncated_commits"],
            "confidence_mean": conf.get("mean"),
            "confidence_median": conf.get("median"),
        }
        paired = (summary.get("gold_evaluation") or {}).get("paired", {}).get(mode)
        if paired:
            row.update({f"gold_{k}": paired[k] for k in ("n", "precision", "recall", "f1", "mcc")})
        rows.append(row)
    return rows


SUMMARY_CSV_FIELDS = [
    "input_mode", "unique_commits", "bug_fix", "non_bug_fix", "parse_failures", "api_failures",
    "bug_fix_share_of_parsed", "truncated_commits", "confidence_mean", "confidence_median",
    "gold_n", "gold_precision", "gold_recall", "gold_f1", "gold_mcc",
]

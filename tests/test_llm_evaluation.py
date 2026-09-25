import pytest

from src.llm_analysis.evaluation import (
    API_FAILURE,
    PARSE_FAILURE,
    binary_metrics,
    classification_count_rows,
    cross_mode_comparison,
    disagreement_rows,
    error_records,
    evaluate_against_gold,
    extract_commit_hash,
    load_gold_labels,
    match_gold,
    normalize_gold_label,
    summarize_mode,
)
from src.llm_analysis.schemas import BUG_FIX, NON_BUG_FIX

SHA = {name: name * 40 for name in "abcdef"}


def rec(commit, label=BUG_FIX, confidence=0.9, error=None, parse_success=True, mode="message_and_diff"):
    return {"repo": "r", "commit_hash": SHA[commit], "input_mode": mode, "label": label if parse_success else None,
            "confidence": confidence if parse_success else None, "error": error, "parse_success": parse_success,
            "reason": f"why {commit}", "diff_truncated": commit == "a", "record_id": commit + mode}


def results(**by_mode):
    return {mode: {("r", SHA[r["commit_hash"][0]]): r for r in records} for mode, records in by_mode.items()}


def test_binary_metrics():
    gold = [BUG_FIX, BUG_FIX, BUG_FIX, NON_BUG_FIX, NON_BUG_FIX]
    pred = [BUG_FIX, BUG_FIX, NON_BUG_FIX, BUG_FIX, NON_BUG_FIX]
    m = binary_metrics(gold, pred)
    assert m["confusion_matrix"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["mcc"] == pytest.approx(1 / 6, abs=1e-4)


def test_binary_metrics_undefined_values_are_none():
    m = binary_metrics([NON_BUG_FIX, NON_BUG_FIX], [NON_BUG_FIX, NON_BUG_FIX])
    assert m["precision"] is None and m["recall"] is None and m["mcc"] is None


@pytest.mark.parametrize("raw, expected", [
    ("TRUE", BUG_FIX), ("1", BUG_FIX), ("bug_fix", BUG_FIX), ("False", NON_BUG_FIX), ("0", NON_BUG_FIX),
    ("NON_BUG_FIX", NON_BUG_FIX), ("MAYBE", None), ("", None), (None, None),
])
def test_normalize_gold_label(raw, expected):
    assert normalize_gold_label(raw) == expected


def test_extract_commit_hash_formats():
    sha = "2b0b54290872dcc333cb1c8880fe2e68829ea723"
    assert extract_commit_hash(f"https://github.com/o/r/commit/{sha}") == sha
    assert extract_commit_hash(f'=HYPERLINK("https://github.com/o/r/commit/{sha}")') == sha
    assert extract_commit_hash(sha.upper()) == sha
    assert extract_commit_hash("not a sha") is None


def test_load_gold_labels_csv_like_annotator_samples(tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text(
        "﻿commit_URL,is_bugfix,obs\n"
        f"https://github.com/o/r/commit/{SHA['a']},TRUE,x\n"
        f"https://github.com/o/r/commit/{SHA['b']},MAYBE,merge\n"
        f"https://github.com/o/r/commit/{SHA['c']},0,\n"
        f"https://github.com/o/r/commit/{SHA['d']},1,\n"
        f"https://github.com/o/r/commit/{SHA['d']},0,conflict\n"
        "no url,1,\n",
        encoding="utf-8",
    )
    gold = load_gold_labels(path)
    assert gold.labels == {SHA["a"]: BUG_FIX, SHA["c"]: NON_BUG_FIX}
    assert gold.rows_with_unusable_label == 1
    assert gold.rows_without_commit == 1
    assert gold.conflicting_commits == [SHA["d"]]


def test_load_gold_labels_requires_columns(tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        load_gold_labels(path)


def test_match_gold_accepts_unique_abbreviations():
    assert match_gold({"aaaaaaa": BUG_FIX, SHA["b"]: NON_BUG_FIX}, [SHA["a"], SHA["b"]]) == {
        SHA["a"]: BUG_FIX, SHA["b"]: NON_BUG_FIX}


def test_summarize_mode_counts_failures_and_truncation():
    summary = summarize_mode({
        1: rec("a"), 2: rec("b", NON_BUG_FIX, 0.4), 3: rec("c", error="HTTP 500", parse_success=False),
        4: rec("d", parse_success=False),
    })
    assert (summary["bug_fix"], summary["non_bug_fix"]) == (1, 1)
    assert (summary["api_failures"], summary["parse_failures"]) == (1, 1)
    assert summary["truncated_commits"] == 1
    assert summary["eligible_candidate_bugfix_rate"]["estimate"] == 0.5
    assert summary["eligible_candidate_bugfix_rate"]["n"] == 2  # só as respostas válidas
    assert summary["confidence"]["n"] == 2 and summary["confidence"]["histogram"]["[0.9, 1.0]"] == 1


def test_cross_mode_and_disagreements():
    res = results(
        message_only=[rec("a", BUG_FIX, mode="message_only"), rec("b", BUG_FIX, mode="message_only"),
                      rec("c", BUG_FIX, mode="message_only")],
        diff_only=[rec("a", BUG_FIX, mode="diff_only"), rec("b", NON_BUG_FIX, mode="diff_only"),
                   rec("c", parse_success=False, mode="diff_only")],
        message_and_diff=[rec("a", BUG_FIX), rec("b", NON_BUG_FIX), rec("c", NON_BUG_FIX)],
    )
    cross = cross_mode_comparison(res)
    assert cross["commits_labeled_in_all_modes"] == 2
    assert cross["commits_classified_differently_across_modes"] == 1
    assert cross["pairwise_disagreements"] == {"message_only_vs_message_and_diff": 1,
                                               "diff_only_vs_message_and_diff": 0}

    rows = disagreement_rows(res, {("r", SHA["b"]): "Fix things\n\nbody"}, {SHA["b"]: NON_BUG_FIX})
    by_commit = {r["commit_hash"]: r for r in rows}
    assert set(by_commit) == {SHA["b"], SHA["c"]}
    assert by_commit[SHA["b"]]["disagreement"] == "message_only!=message_and_diff"
    assert by_commit[SHA["b"]]["message_first_line"] == "Fix things"
    assert by_commit[SHA["b"]]["gold_label"] == NON_BUG_FIX
    assert by_commit[SHA["c"]]["label_diff_only"] == PARSE_FAILURE


def test_counts_and_errors():
    res = results(message_only=[rec("a", mode="message_only"),
                                rec("b", error="boom", parse_success=False, mode="message_only")])
    rows = {r["label"]: r for r in classification_count_rows(res)}
    assert rows[BUG_FIX]["count"] == 1 and rows[API_FAILURE]["fraction"] == 0.5
    errors = error_records(res)
    assert [(e["failure"], e["commit_hash"]) for e in errors] == [(API_FAILURE, SHA["b"])]


def test_gold_evaluation_uses_paired_commits():
    res = results(
        message_only=[rec("a", BUG_FIX, mode="message_only"), rec("b", BUG_FIX, mode="message_only")],
        message_and_diff=[rec("a", BUG_FIX), rec("b", parse_success=False)],
    )
    from src.llm_analysis.evaluation import GoldLabels
    gold = GoldLabels(labels={SHA["a"]: BUG_FIX, SHA["b"]: NON_BUG_FIX, SHA["f"]: BUG_FIX})
    ev = evaluate_against_gold(res, gold, "candidate_validation")

    assert ev["gold_commits_matched"] == 2
    assert ev["paired_commits"] == 1
    assert ev["paired"]["message_only"]["n"] == 1
    assert ev["unpaired"]["message_only"]["n"] == 2
    assert ev["unpaired"]["message_and_diff"]["gold_commits_without_parsed_label"] == 1
    assert "NOT representative" in ev["scope_note"]
    assert ev["gold_eligible_candidate_bugfix_rate"]["estimate"] == 0.5

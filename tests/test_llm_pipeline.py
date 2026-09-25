"""Pipeline completo (01 -> 02 -> 03) com o provider mock, sem rede."""
import csv
import importlib.util
import json
from pathlib import Path

import yaml

from src.llm_analysis.schemas import INPUT_MODES, RESULT_FIELDS
from src.llm_analysis.utils import load_config
from tests.helpers import make_record, write_jsonl

MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "llm_analysis"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), MODULE_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(tmp_path, dataset_dir, output_dir, **overrides):
    cfg = load_config()
    cfg["input"]["dataset_glob"] = str(dataset_dir / "**" / "*.jsonl")
    cfg["output"]["directory"] = str(output_dir)
    cfg["llm"]["provider"] = "mock"
    for section, values in overrides.items():
        cfg[section].update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_end_to_end_with_mock_provider(tmp_path):
    dataset, out = tmp_path / "dataset", tmp_path / "out"
    original = [
        make_record(repo="o_r", commit_hash="1" * 40, file="main.tf", patch="@@ -1 +1 @@\n-a\n+b\n"),
        make_record(repo="o_r", commit_hash="1" * 40, file="vars.tf", patch="@@ -1 +1 @@\n-c\n+d\n"),
        make_record(repo="o_r", commit_hash="2" * 40, file="main.tf", patch="", message="add feature"),
        make_record(repo="o_r", commit_hash="1" * 40, file="tests/main.tf", patch="@@ -1 +1 @@\n-t\n+u\n"),
        make_record(repo="o_r", commit_hash="3" * 40, file="x.tf", message="resolve conflicts"),
        make_record(repo="o_r", commit_hash="4" * 40, file="examples/main.tf", message="fix example"),
    ]
    write_jsonl(str(dataset / "o_r" / "data.jsonl"), original)
    raw_before = (dataset / "o_r" / "data.jsonl").read_bytes()
    gold = tmp_path / "gold.csv"
    gold.write_text(f"commit_hash,gold_label\n{'1' * 40},BUG_FIX\n{'2' * 40},NON_BUG_FIX\n")
    config = write_config(tmp_path, dataset, out, evaluation={"gold_labels": str(gold)})

    assert load_script("01_prepare_commits.py").main(["--config", str(config)]) == 0
    prepared = [json.loads(line) for line in (out / "commits_prepared.jsonl").read_text().splitlines()]
    assert [(p["commit_hash"][0], len(p["diffs"])) for p in prepared] == [("1", 2), ("3", 1)]
    assert prepared[0]["files"] == ["main.tf", "vars.tf"]  # tests/main.tf removido
    assert "-t\n+u" not in json.dumps(prepared)
    exclusions = [json.loads(line) for line in (out / "eligibility_exclusions.jsonl").read_text().splitlines()]
    assert [(e["commit_hash"][0], e["eligibility_state"], e["eligibility_reason"]) for e in exclusions] == [
        ("2", "INSUFFICIENT_DATA", "no_in_scope_patch"), ("4", "OUT_OF_SCOPE", "only_example_files")]
    stats = json.loads((out / "prepare_stats.json").read_text())
    assert stats["eligibility"]["commits_before_eligibility"] == 4
    assert stats["eligibility"]["mixed_scope_commits"] == 1
    assert stats["sampling_population"] == stats["commits_sampled"] == 2
    assert (dataset / "o_r" / "data.jsonl").read_bytes() == raw_before  # entrada intocada

    classify = load_script("02_classify_commits.py")
    assert classify.main(["--config", str(config), "--dry-run"]) == 0
    assert not list(out.glob("classifications_*.jsonl"))
    assert classify.main(["--config", str(config), "--limit", "1"]) == 0
    assert classify.main(["--config", str(config)]) == 0  # retoma: só o 2º commit
    runs = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    assert [(r["classified"], r["skipped_existing"]) for r in runs] == [(3, 0), (3, 3)]

    for mode in INPUT_MODES:
        lines = (out / f"classifications_{mode}.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert list(json.loads(lines[0])) == list(RESULT_FIELDS)

    assert load_script("03_summarize_results.py").main(["--config", str(config)]) == 0
    [report_dir] = list((out / "reports").iterdir())
    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["unique_commits"] == 2
    assert summary["experiment_mode"] == "candidate_validation"
    assert set(summary["by_input_mode"]) == set(INPUT_MODES)
    assert summary["gold_evaluation"]["gold_commits_matched"] == 1  # o commit 2 ficou fora (sem patch)
    assert "llm_estimated_keyword_heuristic_precision" in summary
    for name in ("summary.csv", "classification_counts.csv", "disagreements.csv"):
        with open(report_dir / name) as f:
            assert next(csv.reader(f))  # cabeçalho presente
    assert (report_dir / "errors.jsonl").exists()
    assert (out / "experiments" / f"{report_dir.name}.json").exists()


def test_summary_without_gold_labels_has_no_metrics(tmp_path):
    dataset, out = tmp_path / "dataset", tmp_path / "out"
    write_jsonl(str(dataset / "r" / "data.jsonl"), [make_record()])
    config = write_config(tmp_path, dataset, out)
    load_script("01_prepare_commits.py").main(["--config", str(config)])
    load_script("02_classify_commits.py").main(["--config", str(config), "--input-modes", "message_only"])

    assert load_script("03_summarize_results.py").main(["--config", str(config)]) == 0
    [report_dir] = list((out / "reports").iterdir())
    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["gold_evaluation"] is None
    assert "not computed" in summary["gold_evaluation_note"]


def test_prepare_fails_without_input(tmp_path):
    config = write_config(tmp_path, tmp_path / "empty", tmp_path / "out")
    assert load_script("01_prepare_commits.py").main(["--config", str(config)]) == 1


def test_batch_cli_requires_litellm_provider(tmp_path, capsys):
    dataset, out = tmp_path / "dataset", tmp_path / "out"
    write_jsonl(str(dataset / "r" / "data.jsonl"), [make_record()])
    config = write_config(tmp_path, dataset, out)
    load_script("01_prepare_commits.py").main(["--config", str(config)])

    assert load_script("02_classify_commits.py").main(["--config", str(config), "--batch", "submit"]) == 1
    assert "provider: litellm" in capsys.readouterr().err
    assert not list(out.glob("classifications_*.jsonl"))

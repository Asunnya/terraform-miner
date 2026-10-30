"""Custos por chamada e totais (síncrono e batch), reais e estimados."""
import json

import pytest

from src.llm_analysis.costs import (
    BATCH,
    SYNC,
    Price,
    actual_cost_rows,
    estimate_cost_rows,
    load_pricing,
    pricing_for,
    summarize,
)

PRICING = load_pricing({"models": {"gpt-4.1-nano": {"sync": {"input": 0.10, "output": 0.40},
                                                    "batch": {"input": 0.05, "output": 0.20}}}})


def rec(mode="message_only", input_tokens=1_000_000, output_tokens=100_000, batch_id=None, rid="r1"):
    return {"record_id": rid, "repo": "o_r", "commit_hash": "c1", "input_mode": mode, "model": "openai/gpt-4.1-nano",
            "timestamp": "t", "input_tokens": input_tokens, "output_tokens": output_tokens, "batch_id": batch_id}


def test_price_per_million_tokens():
    assert Price(0.10, 0.40).cost(1_000_000, 1_000_000) == pytest.approx(0.50)
    assert Price(0.10, 0.40).cost(2335, 153) == pytest.approx(0.0002947)


def test_pricing_lookup_accepts_vendor_prefix():
    assert pricing_for("openai/gpt-4.1-nano", PRICING) is pricing_for("gpt-4.1-nano", PRICING)
    with pytest.raises(KeyError, match="No price"):
        pricing_for("openai/unknown", PRICING)


def test_actual_rows_use_api_token_counts_and_the_mode_used():
    sync, batch, failed = rec(rid="s"), rec(batch_id="batch_1", rid="b"), rec(input_tokens=None, rid="f")
    rows = {r["record_id"]: r for r in actual_cost_rows([sync, batch, failed], PRICING)}

    assert set(rows) == {"s", "b"}                              # falha de API não é cobrada
    assert rows["s"]["via"] == SYNC and rows["b"]["via"] == BATCH
    assert rows["s"]["cost_usd"] == pytest.approx(0.14)         # 1M × 0,10 + 0,1M × 0,40
    assert rows["b"]["cost_usd"] == pytest.approx(0.07)         # preço de batch
    assert rows["s"]["cost_if_batch_usd"] == rows["b"]["cost_usd"]
    assert rows["b"]["input_tokens"] == 1_000_000               # exatamente o número da API


def test_summary_totals_by_mode_and_via():
    rows = actual_cost_rows([rec(rid="a"), rec(mode="diff_only", rid="b", batch_id="x"),
                             rec(mode="diff_only", rid="c", input_tokens=500_000, output_tokens=0)], PRICING)
    s = summarize(rows)
    assert s["total"]["calls"] == 3 and s["total"]["input_tokens"] == 2_500_000
    assert s["total"]["cost_usd"] == pytest.approx(0.14 + 0.07 + 0.05)
    assert s["total"]["cost_if_sync_usd"] == pytest.approx(0.14 + 0.14 + 0.05)
    assert s["total"]["cost_if_batch_usd"] == pytest.approx(0.07 + 0.07 + 0.025)
    assert s["by_input_mode"]["diff_only"]["calls"] == 2
    assert set(s["by_via"]) == {SYNC, BATCH}
    assert s["by_input_mode"]["message_only"]["mean_output_tokens"] == 100_000


def test_estimate_rows_count_input_and_assume_output():
    requests = [({"record_id": "x", "repo": "o_r", "commit_hash": "c", "input_mode": "diff_only", "timestamp": None},
                 [{"role": "user", "content": "abcd"}])]
    [row] = estimate_cost_rows(requests, "openai/gpt-4.1-nano", PRICING, 200, lambda messages: 1000)
    assert (row["via"], row["input_tokens"], row["output_tokens"]) == ("estimate", 1000, 200)
    assert row["cost_if_batch_usd"] == pytest.approx(1000 * 0.05e-6 + 200 * 0.20e-6)


def test_cost_report_script_end_to_end(tmp_path):
    from tests.test_llm_pipeline import load_script, write_config
    from tests.helpers import make_record, write_jsonl

    dataset, out = tmp_path / "dataset", tmp_path / "out"
    write_jsonl(str(dataset / "r" / "data.jsonl"), [make_record(), make_record(commit_hash="c2")])
    config = write_config(tmp_path, dataset, out)
    assert load_script("01_prepare_commits.py").main(["--config", str(config)]) == 0
    assert load_script("02_classify_commits.py").main(["--config", str(config), "--limit", "1"]) == 0

    # o mock não gasta tokens; simula a contagem devolvida pela API
    for path in out.glob("classifications_*.jsonl"):
        records = [json.loads(line) | {"input_tokens": 2000, "output_tokens": 150, "model": "openai/gpt-4.1-nano"}
                   for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(r) + "\n" for r in records))

    report = load_script("04_cost_report.py")
    assert report.main(["--config", str(config), "--model", "gpt-4.1-nano"]) == 1  # outro experimento: nada feito
    [exp_dir] = [p for p in (out / "experiments").iterdir()]
    assert report.main(["--config", str(config), "--experiment-id", exp_dir.stem]) == 0
    summary = json.loads((out / "reports" / exp_dir.stem / "costs_summary.json").read_text())
    assert summary["total"]["calls"] == 3 and summary["total"]["input_tokens"] == 6000
    assert summary["total"]["cost_if_batch_usd"] == pytest.approx(summary["total"]["cost_if_sync_usd"] / 2)
    assert (out / "reports" / exp_dir.stem / "costs_per_call.csv").exists()

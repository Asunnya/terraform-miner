"""Token usage and cost per call and in total, for synchronous and batch pricing."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable

TOKENS_PER_PRICE_UNIT = 1_000_000  # prices are per 1M tokens
SYNC = "sync"
BATCH = "batch"
ROUND_USD = 6


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_m + output_tokens * self.output_per_m) / TOKENS_PER_PRICE_UNIT


@dataclass(frozen=True)
class ModelPricing:
    sync: Price
    batch: Price


def load_pricing(cfg: dict[str, Any]) -> dict[str, ModelPricing]:
    """`pricing.models` of config.yaml: {model: {sync: {input, output}, batch: {input, output}}}."""
    pricing = {}
    for model, entry in (cfg.get("models") or {}).items():
        pricing[model] = ModelPricing(
            sync=Price(float(entry["sync"]["input"]), float(entry["sync"]["output"])),
            batch=Price(float(entry["batch"]["input"]), float(entry["batch"]["output"])),
        )
    return pricing


def pricing_for(model: str, pricing: dict[str, ModelPricing]) -> ModelPricing:
    """Look up "openai/gpt-4.1-nano", then "gpt-4.1-nano"."""
    for key in (model, model.split("/", 1)[-1]):
        if key in pricing:
            return pricing[key]
    raise KeyError(f"No price for model {model!r} in config.yaml pricing.models (have: {sorted(pricing)})")


def _row(base: dict[str, Any], input_tokens: int, output_tokens: int, via: str, prices: ModelPricing) -> dict[str, Any]:
    sync_cost = prices.sync.cost(input_tokens, output_tokens)
    batch_cost = prices.batch.cost(input_tokens, output_tokens)
    return {
        **base,
        "via": via,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(batch_cost if via == BATCH else sync_cost, ROUND_USD),
        "cost_if_sync_usd": round(sync_cost, ROUND_USD),
        "cost_if_batch_usd": round(batch_cost, ROUND_USD),
    }


def actual_cost_rows(records: Iterable[dict[str, Any]], pricing: dict[str, ModelPricing]) -> list[dict[str, Any]]:
    """One row per call made (records with token counts); via = batch when the record has a batch_id."""
    rows = []
    for rec in records:
        if rec.get("input_tokens") is None:  # API failure: nothing billed
            continue
        base = {k: rec.get(k) for k in ("record_id", "repo", "commit_hash", "input_mode", "model", "timestamp")}
        usage = rec.get("usage") or {}
        row = _row(base, rec["input_tokens"], rec.get("output_tokens") or 0,
                   BATCH if rec.get("batch_id") else SYNC, pricing_for(rec["model"], pricing))
        row.update(
            cached_input_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            cost_at_call_usd=rec.get("cost_usd"),
            litellm_cost_usd=rec.get("litellm_cost_usd"),
        )
        rows.append(row)
    return rows


def estimate_cost_rows(requests: Iterable[tuple[dict[str, Any], list[dict[str, str]]]], model: str,
                       pricing: dict[str, ModelPricing], output_tokens: int,
                       count_tokens: Callable[[list[dict[str, str]]], int]) -> list[dict[str, Any]]:
    """Rows for calls not made yet: input tokens counted locally, output tokens assumed."""
    prices = pricing_for(model, pricing)
    rows = []
    for base, messages in requests:
        rows.append(_row({**base, "model": model}, count_tokens(messages), output_tokens, "estimate", prices))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals overall, per input mode and per via (sync/batch/estimate)."""
    def empty():
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "cost_if_sync_usd": 0.0, "cost_if_batch_usd": 0.0}

    groups: dict[str, dict[str, dict[str, Any]]] = {"total": defaultdict(empty), "by_input_mode": defaultdict(empty),
                                                    "by_via": defaultdict(empty)}
    for row in rows:
        for group, key in (("total", "all"), ("by_input_mode", row["input_mode"]), ("by_via", row["via"])):
            acc = groups[group][key]
            acc["calls"] += 1
            for field in ("input_tokens", "output_tokens"):
                acc[field] += row[field]
            for field in ("cost_usd", "cost_if_sync_usd", "cost_if_batch_usd"):
                acc[field] += row[field]
    result = {}
    for group, accs in groups.items():
        result[group] = {}
        for key, acc in accs.items():
            calls = acc["calls"]
            result[group][key] = {
                **{k: (round(v, ROUND_USD) if k.endswith("_usd") else v) for k, v in acc.items()},
                "mean_input_tokens": round(acc["input_tokens"] / calls, 1) if calls else None,
                "mean_output_tokens": round(acc["output_tokens"] / calls, 1) if calls else None,
            }
    result["total"] = result["total"].get("all", empty())
    return result


COST_CSV_FIELDS = ["record_id", "repo", "commit_hash", "input_mode", "model", "timestamp", "via", "input_tokens",
                   "output_tokens", "cached_input_tokens", "reasoning_tokens", "cost_usd", "cost_if_sync_usd",
                   "cost_if_batch_usd", "cost_at_call_usd", "litellm_cost_usd"]

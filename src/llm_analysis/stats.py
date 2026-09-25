"""Wilson interval, Cohen's kappa, exact McNemar test and Holm correction, in plain Python."""
from __future__ import annotations

import math
from typing import Sequence

Z_95 = 1.959963984540054  # two-sided 95%
DECIMALS = 4
P_VALUE_SIGNIFICANT_DIGITS = 6


def _round_p(p: float) -> float:
    """Round to significant digits, so tiny p-values never become 0."""
    return float(f"{p:.{P_VALUE_SIGNIFICANT_DIGITS}g}")


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval; None when n == 0."""
    if n <= 0:
        return None
    p = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return round(max(0.0, center - half_width), DECIMALS), round(min(1.0, center + half_width), DECIMALS)


def proportion(successes: int, n: int) -> dict[str, float | int | None]:
    """Point estimate with its 95% Wilson interval."""
    interval = wilson_interval(successes, n)
    return {
        "successes": successes,
        "n": n,
        "estimate": round(successes / n, DECIMALS) if n else None,
        "ci95_low": interval[0] if interval else None,
        "ci95_high": interval[1] if interval else None,
    }


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> dict[str, float | int | None]:
    """Cohen's kappa for paired labels; None when undefined (a single shared category)."""
    if len(labels_a) != len(labels_b):
        raise ValueError("kappa needs paired sequences of equal length")
    n = len(labels_a)
    if n == 0:
        return {"n": 0, "observed_agreement": None, "expected_agreement": None, "kappa": None}
    categories = set(labels_a) | set(labels_b)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    expected = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories)
    kappa = None if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)
    return {
        "n": n,
        "observed_agreement": round(observed, DECIMALS),
        "expected_agreement": round(expected, DECIMALS),
        "kappa": round(kappa, DECIMALS) if kappa is not None else None,
    }


def _log_binomial_pmf(k: int, n: int) -> float:
    """log P(X = k) for X ~ Binomial(n, 0.5)."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)


def mcnemar_exact(only_a_correct: int, only_b_correct: int) -> dict[str, float | int | None]:
    """Exact two-sided McNemar test on the discordant pairs; p is None when there are none."""
    b, c = only_a_correct, only_b_correct
    n = b + c
    if n == 0:
        return {"only_a_correct": b, "only_b_correct": c, "discordant": 0, "p_value": None}
    k = min(b, c)
    logs = [_log_binomial_pmf(i, n) for i in range(k + 1)]
    top = max(logs)
    tail = math.exp(top) * sum(math.exp(v - top) for v in logs)
    return {"only_a_correct": b, "only_b_correct": c, "discordant": n,
            "p_value": _round_p(min(1.0, 2 * tail))}


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm-adjusted p-values (None stays None)."""
    indexed = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running_max = 0.0
    for rank, (p, i) in enumerate(indexed):
        running_max = max(running_max, min(1.0, (m - rank) * p))
        adjusted[i] = _round_p(running_max)
    return adjusted

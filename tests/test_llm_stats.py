"""Estatísticas do estágio 03 contra valores de referência conhecidos."""
import math

import pytest

from src.llm_analysis.evaluation import GoldLabels, cross_mode_comparison, evaluate_against_gold
from src.llm_analysis.schemas import BUG_FIX, NON_BUG_FIX
from src.llm_analysis.stats import cohen_kappa, holm_adjust, mcnemar_exact, proportion, wilson_interval

B, N = BUG_FIX, NON_BUG_FIX


def test_wilson_interval_reference_values():
    assert wilson_interval(50, 100) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    low, high = wilson_interval(0, 10)  # nunca negativo, ao contrário do intervalo de Wald
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    assert wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(0, 0) is None
    assert proportion(3, 4) == {"successes": 3, "n": 4, "estimate": 0.75,
                                "ci95_low": pytest.approx(0.3006, abs=1e-4),
                                "ci95_high": pytest.approx(0.9544, abs=1e-4)}


def test_cohen_kappa_reference_example():
    # Exemplo clássico (50 itens): concordância observada 0,7, esperada 0,5 -> kappa 0,4
    a = [B] * 20 + [B] * 5 + [N] * 10 + [N] * 15
    b = [B] * 20 + [N] * 5 + [B] * 10 + [N] * 15
    result = cohen_kappa(a, b)
    assert result["observed_agreement"] == 0.7 and result["expected_agreement"] == 0.5
    assert result["kappa"] == pytest.approx(0.4)


def test_cohen_kappa_edge_cases():
    assert cohen_kappa([B, N], [B, N])["kappa"] == 1.0
    assert cohen_kappa([B, B], [B, B])["kappa"] is None  # uma só categoria: indefinido
    assert cohen_kappa([], [])["kappa"] is None
    with pytest.raises(ValueError):
        cohen_kappa([B], [])


def test_mcnemar_exact_reference_values():
    assert mcnemar_exact(1, 6)["p_value"] == pytest.approx(0.125)  # 2 * (C(7,0)+C(7,1)) / 2^7
    assert mcnemar_exact(5, 5)["p_value"] == 1.0
    assert mcnemar_exact(0, 0)["p_value"] is None
    big = mcnemar_exact(400, 600)["p_value"]  # n grande: sem underflow e sem arredondar para 0
    assert big == pytest.approx(2.728464e-10, rel=1e-5)  # scipy.stats.binomtest(400, 1000).pvalue
    assert math.isfinite(big)


def test_holm_adjust():
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm_adjust([0.5, None, 0.9]) == [1.0, None, 1.0]


def _results(labels_by_mode):
    return {mode: {("r", sha): {"repo": "r", "commit_hash": sha, "label": label, "parse_success": True,
                               "error": None, "confidence": 0.9}
                   for sha, label in labels.items()}
            for mode, labels in labels_by_mode.items()}


def test_kappa_and_mcnemar_in_the_reports():
    shas = [f"{i:040x}" for i in range(8)]
    gold = {s: (B if i < 4 else N) for i, s in enumerate(shas)}
    res = _results({
        "message_only": {s: B for s in shas},                    # acerta 4/8
        "diff_only": dict(gold),                                 # acerta 8/8
        "message_and_diff": {**gold, shas[4]: B},                # acerta 7/8
    })

    kappa = cross_mode_comparison(res)["cohen_kappa"]
    assert set(kappa) == {"message_only_vs_diff_only", "message_only_vs_message_and_diff",
                          "diff_only_vs_message_and_diff"}
    assert kappa["message_only_vs_diff_only"]["kappa"] == 0.0

    ev = evaluate_against_gold(res, GoldLabels(labels=gold), "candidate_validation")
    tests = ev["mcnemar"]["tests"]
    assert tests["message_only_vs_diff_only"]["only_b_correct"] == 4
    assert tests["message_only_vs_diff_only"]["p_value"] == pytest.approx(0.125)
    assert tests["message_only_vs_diff_only"]["p_value_holm"] >= tests["message_only_vs_diff_only"]["p_value"]
    assert ev["paired"]["diff_only"]["precision_ci95"]["n"] == 4
    assert ev["gold_eligible_candidate_bugfix_rate"]["estimate"] == 0.5

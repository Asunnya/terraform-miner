"""
diff_stats.py
=============

Utilities de *analytics* para gerar estatísticas de diffs Terraform.

Este módulo **não** faz parsing de patch (isso já está em ``diff_utils``);
ele apenas consome o DataFrame retornado por ``diff_utils.parse_patch_to_dataframe``
e devolve métricas prontas para notebooks ou para uma pipeline em lote.


"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import pandas as pd

# Importa apenas o que é necessário do core
from diff_utils import (
    CHANGE_CATEGORIES,  # exportado no diff_utils.__all__
    parse_patch_to_dataframe,
)

__all__ = [
    "count_change_categories",
    "summarize_patch",
    "summarize_commit_list",
]


# --------------------------------------------------------------------------- #
# 1. Métricas de nível de diff                                                #
# --------------------------------------------------------------------------- #
def count_change_categories(df: pd.DataFrame) -> pd.Series:
    """
    Conta quantas linhas existem em cada categoria de mudança.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame gerado por ``parse_patch_to_dataframe`` contendo a coluna
        ``'category'`` (resource_definition, value_modification, …).

    Returns
    -------
    pandas.Series
        Índice = categoria, valor = nº de linhas.
    """
    if df.empty or "category" not in df.columns:
        return pd.Series(dtype="int")

    return df["category"].value_counts().sort_index()


def summarize_patch(
    patch: str | None = None, *, df: pd.DataFrame | None = None
) -> Dict[str, int]:
    """
    Gera um dicionário {categoria: contagem} para um único patch de diff.

    Você pode passar **patch** (string) ou **df** já pré-processado; se ambos
    forem fornecidos, **df** tem precedência.

    Parameters
    ----------
    patch : str, optional
        Texto do diff unificado (``git diff``).
    df : pandas.DataFrame, optional
        Resultado de ``parse_patch_to_dataframe`` caso já esteja em memória.

    Returns
    -------
    dict
        Chaves = categorias definidas em ``CHANGE_CATEGORIES`` + ``'other'``.
    """
    if df is None:
        if not patch:
            raise ValueError("É necessário fornecer 'patch' ou 'df'.")
        df = parse_patch_to_dataframe(patch)

    series = count_change_categories(df)
    # Garante que todas as chaves sejam devolvidas (valor 0 se ausente)
    return {cat: int(series.get(cat, 0)) for cat in list(CHANGE_CATEGORIES) + ["other"]}


# --------------------------------------------------------------------------- #
# 2. Métricas agregadas para múltiplos commits                                #
# --------------------------------------------------------------------------- #
def summarize_commit_list(
    commits: List[Dict[str, Any]],
    patch_key: str = "patch",
    extra_fields: List[str] | None = None,
) -> pd.DataFrame:
    """
    Produz DataFrame com estatísticas por commit.

    Parameters
    ----------
    commits : list of dict
        Cada item deve conter ao menos ``patch_key`` (string diff) e pode ter
        metadados extras (``repo``, ``commit_hash``, …).
    patch_key : str, default "patch"
        Nome do campo que contém o diff unificado em cada dict.
    extra_fields : list of str, optional
        Quais outros campos copiar para o resultado.

    Returns
    -------
    pandas.DataFrame
        Colunas:
        * *extra_fields* (se houver)
        * uma coluna por categoria (valor = nº de linhas no commit)
    """
    records: List[Dict[str, Any]] = []

    cats = list(CHANGE_CATEGORIES) + ["other"]
    extra_fields = extra_fields or []

    for commit in commits:
        counts = summarize_patch(commit[patch_key])

        row: Dict[str, Any] = {c: counts[c] for c in cats}
        for fld in extra_fields:
            row[fld] = commit.get(fld)
        records.append(row)

    df_out = pd.DataFrame(records)
    # Ordena colunas: extras primeiro, depois categorias
    return df_out[extra_fields + cats]

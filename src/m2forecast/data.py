"""Ingestão e validação da série M2 (BCB/SGS 27810)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL_SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json"


class SerieInvalida(ValueError):
    pass


def parse_registros(registros: list[dict]) -> pd.Series:
    """Converte o JSON do SGS em série de níveis com PeriodIndex mensal."""
    df = pd.DataFrame(registros)
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    nivel = df.set_index("data")["valor"].dropna().sort_index()
    nivel.index = nivel.index.to_period("M")
    nivel.name = "valor"
    return nivel


def fetch_bcb(codigo: int = 27810, timeout: int = 60) -> pd.Series:
    resp = requests.get(URL_SGS.format(codigo=codigo), timeout=timeout)
    resp.raise_for_status()
    return parse_registros(resp.json())


def carregar_snapshot(caminho: str | Path) -> pd.Series:
    with open(caminho, encoding="utf-8") as f:
        return parse_registros(json.load(f))


def para_crescimento(nivel: pd.Series) -> pd.Series:
    """g_t = (M2_t / M2_{t-1} - 1) * 100  (igual ao notebook)."""
    g = (nivel.pct_change() * 100).dropna()
    g.name = "m2_growth"
    return g


def validar(serie: pd.Series) -> None:
    if serie.empty:
        raise SerieInvalida("Série vazia.")
    if not isinstance(serie.index, pd.PeriodIndex):
        raise SerieInvalida("Índice precisa ser PeriodIndex mensal.")
    if not serie.index.is_unique:
        raise SerieInvalida("Meses duplicados.")
    if not serie.index.is_monotonic_increasing:
        raise SerieInvalida("Série fora de ordem.")
    esperado = pd.period_range(serie.index.min(), serie.index.max(), freq="M")
    if not serie.index.equals(esperado):
        faltando = esperado.difference(serie.index)
        raise SerieInvalida(f"Meses ausentes: {list(map(str, faltando))}")
    if not np.isfinite(serie.to_numpy(dtype=float)).all():
        raise SerieInvalida("Valores não finitos.")


def detectar_novidade(nova: pd.Series, antiga: pd.Series | None, tol: float = 1e-6) -> dict:
    """Compara níveis novos x armazenados: meses novos e revisões."""
    if antiga is None or antiga.empty:
        return {"novos": [str(p) for p in nova.index], "revisados": [], "ultimo": str(nova.index.max())}
    novos = nova.index.difference(antiga.index)
    comuns = nova.index.intersection(antiga.index)
    dif = (nova.loc[comuns] - antiga.loc[comuns]).abs()
    revisados = dif[dif > tol * antiga.loc[comuns].abs().clip(lower=1)].index
    return {
        "novos": [str(p) for p in novos],
        "revisados": [str(p) for p in revisados],
        "ultimo": str(nova.index.max()),
    }

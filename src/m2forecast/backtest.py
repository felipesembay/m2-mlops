"""Avaliação walk-forward (origem móvel) e métricas, como no artigo."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .models import criar


def walk_forward(
    serie: pd.Series,
    spec: str,
    horizonte: int,
    alvos: Iterable[pd.Period],
    janela: int | None = 120,
    **params,
) -> pd.DataFrame:
    """Para cada data-alvo, ajusta na origem = alvo - h e prevê o alvo."""
    linhas = []
    for alvo in alvos:
        alvo = pd.Period(alvo, freq="M")
        origem = alvo - horizonte
        if alvo not in serie.index or origem not in serie.index:
            continue
        try:
            prev = criar(spec, horizonte, janela=janela, **params).fit(serie, origem).predict()
        except Exception as erro:  # noqa: BLE001 - registra falha e segue
            linhas.append({"modelo": spec, "horizonte": horizonte, "origem": origem,
                           "data_alvo": alvo, "real": np.nan, "previsto": np.nan, "falha": str(erro)})
            continue
        linhas.append({
            "modelo": spec,
            "horizonte": horizonte,
            "origem": origem,
            "data_alvo": alvo,
            "real": float(serie.loc[alvo]),
            "previsto": prev,
        })
    df = pd.DataFrame(linhas)
    if not df.empty:
        df["erro"] = df["real"] - df["previsto"]
    return df


def metricas(df: pd.DataFrame, por=("modelo", "horizonte")) -> pd.DataFrame:
    d = df.dropna(subset=["real", "previsto"]).copy()
    d["erro"] = d["real"] - d["previsto"]
    return (
        d.groupby(list(por))
        .agg(
            MAE=("erro", lambda e: float(np.mean(np.abs(e)))),
            RMSE=("erro", lambda e: float(np.sqrt(np.mean(e**2)))),
            Bias=("erro", "mean"),
            N=("erro", "size"),
        )
        .reset_index()
    )

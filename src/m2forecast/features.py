"""Engenharia de atributos (célula ML-13 do notebook, sem alterações de lógica)."""

from __future__ import annotations

import pandas as pd


def criar_features_boost(serie: pd.Series, horizonte: int) -> pd.DataFrame:
    dados = pd.DataFrame(index=serie.index)

    # Defasagens disponíveis na origem
    for lag in [0, 1, 2, 3, 6, 9, 12, 18, 24]:
        dados[f"lag_{lag}"] = serie.shift(lag)

    # Estatísticas móveis
    for janela in [3, 6, 12, 24]:
        dados[f"media_{janela}"] = serie.rolling(janela).mean()
        dados[f"std_{janela}"] = serie.rolling(janela).std()

    # Diferenças históricas
    dados["diff_1"] = serie.diff(1)
    dados["diff_12"] = serie.diff(12)

    # Mês da data-alvo
    dados["mes_alvo"] = [(data + horizonte).month for data in serie.index]

    # Target de previsão direta
    dados["target"] = serie.shift(-horizonte)

    return dados

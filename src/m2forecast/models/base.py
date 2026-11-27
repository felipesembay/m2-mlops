from __future__ import annotations

from typing import Any

import pandas as pd


class Forecaster:
    spec: str = ""
    familia: str = ""

    def __init__(self, horizonte: int, janela: int | None = 120, **params: Any):
        self.horizonte = int(horizonte)
        self.janela = janela
        self.params = params
        self.origem: pd.Period | None = None
        self.dados_ate: pd.Period | None = None

    # ------------------------------------------------------------------
    def _historico(self, serie: pd.Series, origem) -> pd.Series:
        origem = pd.Period(origem, freq="M")
        hist = serie.loc[serie.index <= origem].dropna().astype(float)
        if hist.empty or hist.index[-1] != origem:
            raise ValueError(f"Origem {origem} ausente da série.")
        self.origem = origem
        self.dados_ate = hist.index[-1]
        return hist.tail(self.janela) if self.janela else hist

    @property
    def data_alvo(self) -> pd.Period:
        if self.origem is None:
            raise RuntimeError("Modelo não ajustado.")
        return self.origem + self.horizonte

    def fit(self, serie: pd.Series, origem) -> "Forecaster":  # pragma: no cover
        raise NotImplementedError

    def predict(self) -> float:  # pragma: no cover
        raise NotImplementedError

    def trajetoria(self, n: int) -> list[float] | None:
        """Previsões para origem+1..origem+n com um único ajuste.

        Modelos recursivos (SARIMA, ETS, STL+ETS, Média Sazonal) implementam;
        modelos diretos (Random Forest) devolvem None e são reajustados por passo.
        """
        return None

    def descricao(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "familia": self.familia,
            "horizonte": self.horizonte,
            "janela": self.janela,
            "params": self.params,
            "origem": str(self.origem) if self.origem is not None else None,
        }

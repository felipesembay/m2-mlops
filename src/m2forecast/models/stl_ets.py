"""Híbrido STL + ETS amortecido com sazonalidade média (HYB-08)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

from .base import Forecaster


class STLETS(Forecaster):
    spec = "STL_ETS"
    familia = "Hibrido"

    def fit(self, serie, origem):
        hist = self._historico(serie, origem)
        resultado = STL(hist, period=12, robust=True, seasonal=13).fit()
        sazonalidade = pd.Series(resultado.seasonal, index=hist.index)
        dessaz = hist - sazonalidade
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._ets = ExponentialSmoothing(
                dessaz,
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
        self._saz_mes = sazonalidade.groupby(sazonalidade.index.month).mean().to_dict()
        return self

    def componentes(self) -> dict[str, float]:
        ets = float(np.asarray(self._ets.forecast(self.horizonte))[-1])
        saz = float(self._saz_mes[self.data_alvo.month])
        return {"ets": ets, "sazonalidade": saz, "previsao": ets + saz}

    def predict(self) -> float:
        return self.componentes()["previsao"]

    def trajetoria(self, n):
        ets = np.asarray(self._ets.forecast(n), dtype=float)
        return [float(ets[k - 1] + self._saz_mes[(self.origem + k).month]) for k in range(1, n + 1)]

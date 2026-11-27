"""ETS(A,N,A) — candidato estatístico usado na re-seleção."""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from .base import Forecaster


class ETSANA(Forecaster):
    spec = "ETS_ANA"
    familia = "Estatistico"

    def fit(self, serie, origem):
        hist = self._historico(serie, origem)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._ajuste = ETSModel(
                hist,
                error="add",
                trend=None,
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            ).fit(disp=False, maxiter=1000)
        return self

    def predict(self) -> float:
        return float(np.asarray(self._ajuste.forecast(steps=self.horizonte))[-1])

    def trajetoria(self, n):
        return [float(v) for v in np.asarray(self._ajuste.forecast(steps=n))]

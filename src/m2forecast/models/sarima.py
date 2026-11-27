"""SARIMA(3,0,0)(0,1,1,12) sem constante (FINAL-03)."""

from __future__ import annotations

import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import Forecaster


class SARIMA(Forecaster):
    spec = "SARIMA_300_011"
    familia = "Estatistico"
    order = (3, 0, 0)
    seasonal_order = (0, 1, 1, 12)

    def fit(self, serie, origem):
        hist = self._historico(serie, origem)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._ajuste = SARIMAX(
                hist,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend="n",
                enforce_stationarity=True,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)
        self.convergiu = bool(self._ajuste.mle_retvals.get("converged", False))
        return self

    def predict(self) -> float:
        return float(self._ajuste.forecast(steps=self.horizonte).iloc[-1])

    def trajetoria(self, n):
        return [float(v) for v in self._ajuste.forecast(steps=n)]

    def intervalo(self, alpha: float = 0.05) -> tuple[float, float]:
        ci = self._ajuste.get_forecast(steps=self.horizonte).conf_int(alpha=alpha).iloc[-1]
        return float(ci.iloc[0]), float(ci.iloc[1])

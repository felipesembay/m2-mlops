"""Benchmark: média histórica do mesmo mês-calendário (FINAL-09)."""

from __future__ import annotations

from .base import Forecaster


class MediaSazonal(Forecaster):
    spec = "Media_Sazonal"
    familia = "Benchmark"

    def fit(self, serie, origem):
        hist = self._historico(serie, origem)
        self._medias = hist.groupby(hist.index.month).mean().to_dict()
        return self

    def predict(self) -> float:
        mes = self.data_alvo.month
        if mes not in self._medias:
            raise ValueError(f"Sem histórico para o mês {mes}")
        return float(self._medias[mes])

    def trajetoria(self, n):
        return [float(self._medias[(self.origem + k).month]) for k in range(1, n + 1)]

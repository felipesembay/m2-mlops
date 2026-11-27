"""Modelos de previsão. Cada classe segue a interface Forecaster:

    f = Classe(horizonte, janela=120, **params).fit(serie, origem)
    f.predict() -> float   # previsão para origem + horizonte

`fit` só enxerga dados até a origem (sem vazamento temporal).
"""

from .base import Forecaster
from .registry import SPECS, criar

__all__ = ["Forecaster", "SPECS", "criar"]

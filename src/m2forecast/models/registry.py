from __future__ import annotations

from typing import Any

from .base import Forecaster
from .ets import ETSANA
from .rf import RFBayesian
from .sarima import SARIMA
from .seasonal_mean import MediaSazonal
from .stl_ets import STLETS

SPECS: dict[str, type[Forecaster]] = {
    cls.spec: cls for cls in (MediaSazonal, STLETS, SARIMA, ETSANA, RFBayesian)
}


def criar(spec: str, horizonte: int, janela: int | None = 120, **params: Any) -> Forecaster:
    if spec not in SPECS:
        raise KeyError(f"Spec desconhecida: {spec}. Opções: {sorted(SPECS)}")
    return SPECS[spec](horizonte, janela=janela, **params)

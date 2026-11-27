import pandas as pd
import pytest

from m2forecast import data


def test_validar_detecta_mes_ausente(serie):
    with pytest.raises(data.SerieInvalida):
        data.validar(serie.drop(serie.index[10]))


def test_detectar_novidade(serie):
    nivel = pd.Series(range(1, 6), index=pd.period_range("2026-01", periods=5, freq="M"), dtype=float)
    antiga = nivel.iloc[:4].copy()
    antiga.iloc[1] = 99
    nov = data.detectar_novidade(nivel, antiga)
    assert nov["novos"] == ["2026-05"] and nov["revisados"] == ["2026-02"]

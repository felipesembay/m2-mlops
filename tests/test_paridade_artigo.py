"""Reproduz a Tabela 11 do artigo (teste final jan-set/2026).

Garante que a refatoração do notebook para o pacote não alterou os modelos.

Observação: o SARIMA(3,0,0)(0,1,1,12) é estimado por máxima verossimilhança
numérica; com statsmodels/scipy locais as previsões diferem até ~0,07 p.p.
das do notebook (Colab), com o mesmo código. Por isso a tolerância do SARIMA
é maior. STL+ETS e Média Sazonal são determinísticos e batem em 1e-3.
"""

from pathlib import Path

import pandas as pd
import pytest

from m2forecast import backtest
from m2forecast.config import carregar

ALVOS = pd.period_range("2026-01", "2026-09", freq="M")
REF = Path(__file__).parent / "fixtures" / "previsoes_teste_2026.csv"

# RMSE no teste 2026 (Tabela 11)
RMSE_ARTIGO = {1: 0.266, 3: 0.402, 6: 0.389, 12: 0.345}
RMSE_MEDIA_SAZONAL = 0.449
TOL_RMSE = {"STL_ETS": 2e-3, "RF_Bayesian": 0.02, "SARIMA_300_011": 0.03}
TOL_PREV = {"STL_ETS": 1e-3, "RF_Bayesian": 0.05, "SARIMA_300_011": 0.08}


def test_serie_tem_297_observacoes(serie):
    assert len(serie) == 297
    assert str(serie.index.max()) == "2026-09"


@pytest.mark.parametrize("h", [1, 3, 6, 12])
def test_campeao_reproduz_tabela_11(serie, h):
    spec = carregar()["campeoes"][h]["spec"]
    df = backtest.walk_forward(serie, spec, h, ALVOS)
    m = backtest.metricas(df).iloc[0]
    assert m["N"] == 9
    assert m["RMSE"] == pytest.approx(RMSE_ARTIGO[h], abs=TOL_RMSE[spec])
    # e continua melhor que o benchmark, como no artigo
    assert m["RMSE"] < RMSE_MEDIA_SAZONAL


@pytest.mark.parametrize("h", [1, 3, 6, 12])
def test_previsoes_batem_com_notebook(serie, h):
    ref = pd.read_csv(REF)
    ref = ref[(ref["horizonte"] == h) & (ref["modelo"] != "Media_Sazonal")].sort_values("data_alvo")
    spec = carregar()["campeoes"][h]["spec"]
    df = backtest.walk_forward(serie, spec, h, ALVOS).sort_values("data_alvo")
    assert df["previsto"].to_numpy() == pytest.approx(ref["previsto"].to_numpy(), abs=TOL_PREV[spec])


@pytest.mark.parametrize("h", [1, 3, 6, 12])
def test_media_sazonal_benchmark(serie, h):
    df = backtest.walk_forward(serie, "Media_Sazonal", h, ALVOS, janela=None)
    assert backtest.metricas(df).iloc[0]["RMSE"] == pytest.approx(RMSE_MEDIA_SAZONAL, abs=2e-3)


# Validação 2021-2025 (Tabelas 6-9): origens a partir de 2021-02, janela 120
RMSE_VALIDACAO = {1: 0.610, 3: 0.588, 6: 0.543, 12: 0.495}
RMSE_VAL_BENCH = {1: 0.613, 3: 0.618, 6: 0.594, 12: 0.582}
N_VALIDACAO = {1: 58, 3: 56, 6: 53, 12: 47}


@pytest.mark.parametrize("h", [1, 3, 6, 12])
def test_validacao_reproduz_tabelas_6_a_9(serie, h):
    alvos = pd.period_range(pd.Period("2021-02", "M") + h, "2025-12", freq="M")
    spec = carregar()["campeoes"][h]["spec"]
    m = backtest.metricas(backtest.walk_forward(serie, spec, h, alvos)).iloc[0]
    b = backtest.metricas(backtest.walk_forward(serie, "Media_Sazonal", h, alvos, janela=120)).iloc[0]
    assert m["N"] == N_VALIDACAO[h]
    assert m["RMSE"] == pytest.approx(RMSE_VALIDACAO[h], abs=1e-3)
    assert b["RMSE"] == pytest.approx(RMSE_VAL_BENCH[h], abs=1e-3)

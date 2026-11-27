import pytest

from m2forecast import projection
from m2forecast.models import criar


@pytest.mark.parametrize("spec", ["SARIMA_300_011", "STL_ETS", "Media_Sazonal", "ETS_ANA"])
def test_trajetoria_igual_previsao_direta(serie, spec):
    """Um ajuste + trajetória == um ajuste por horizonte (modelos recursivos)."""
    origem = serie.index[-1]
    traj = projection._caminho(spec, {}, 120, serie, 6)
    for k in (1, 3, 6):
        direto = criar(spec, k, janela=120).fit(serie, origem).predict()
        assert traj[k - 1] == pytest.approx(direto, abs=1e-9)


def test_bucket():
    assert [projection.bucket(k) for k in (1, 2, 3, 4, 6, 7, 12, 18)] == [1, 3, 3, 6, 6, 12, 12, 12]


def test_projetar_melhor(serie, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path/'mlruns'}")
    nivel = 7_870_395_191.0
    out = projection.projetar(serie, nivel, "melhor", 8)
    p = out["projecao"]
    assert len(p) == 8 and p[0]["data_alvo"] == "2026-10"
    assert [x["h_modelo"] for x in p] == [1, 3, 3, 6, 6, 6, 12, 12]
    assert [x["modelo"] for x in p][:4] == ["STL_ETS", "RF_Bayesian", "RF_Bayesian", "SARIMA_300_011"]
    # passo 6 no modo "melhor" == campeão h=6 sozinho
    assert p[5]["previsto"] == pytest.approx(projection.projetar(serie, nivel, 6, 6)["projecao"][5]["previsto"])
    # nível projetado é consistente com o crescimento acumulado
    assert p[-1]["nivel_m2"] == pytest.approx(nivel * (1 + out["crescimento_acumulado_pct"] / 100))
    assert all(x["li"] < x["previsto"] < x["ls"] for x in p)


def test_limite_meses(serie):
    with pytest.raises(ValueError):
        projection.projetar(serie, 1.0, "benchmark", 25)

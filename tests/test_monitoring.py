import numpy as np
import pandas as pd

from m2forecast import monitoring

CFG = {
    "janela_meses": 12, "n_minimo": 6,
    "gatilho_a": {"rmse_relativo_max": 1.0, "meses_consecutivos": 3},
    "gatilho_b": {"fator_rmse_validacao": 1.25},
    "alertas": {"vies_abs_max": 0.30, "zscore_max": 3.0},
}


def _erros(e_champ, e_bench, h=1):
    alvos = [str(p) for p in pd.period_range("2024-01", periods=len(e_champ), freq="M")]
    real = np.zeros(len(e_champ))
    return pd.DataFrame({"horizonte": h, "data_alvo": alvos, "real": real,
                         "champion": -np.asarray(e_champ), "benchmark": -np.asarray(e_bench)})


def _rodar(e_c, e_b, rmse_val=0.6):
    mon = monitoring.metricas_moveis(_erros(e_c, e_b), 12, 6)
    return monitoring.aplicar_regras(mon, {1: rmse_val}, CFG)


def test_ok_quando_campeao_melhor():
    rng = np.random.default_rng(0)
    e_b = rng.normal(0, 0.6, 24)
    out = _rodar(e_b * 0.8, e_b)
    assert not out["gatilho_a"].any() and not out["gatilho_b"].any()
    assert out.iloc[-1]["status"] in {"OK", "ALERTA"}


def test_gatilho_a_exige_persistencia():
    e_b = np.full(24, 0.5, dtype=float)
    e_c = np.full(24, 0.45, dtype=float)
    e_c[-2:] = 2.0     # só 2 meses ruins: perde p/ benchmark mas sem persistência de 3
    out = _rodar(e_c, e_b, rmse_val=5)
    assert out.iloc[-1]["rmse_rel"] > 1
    assert not out.iloc[-1]["gatilho_a"]
    e_c[-3:] = 2.0     # 3 meses: dispara
    out = _rodar(e_c, e_b, rmse_val=5)
    assert out.iloc[-1]["gatilho_a"]
    assert out.iloc[-1]["status"] == "RETREINO"


def test_gatilho_b_degradacao_absoluta():
    e = np.full(12, 0.9)
    out = _rodar(e, e * 2, rmse_val=0.6)      # 0.9 > 1.25*0.6 = 0.75, mas melhor que bench
    assert out.iloc[-1]["gatilho_b"] and not out.iloc[-1]["gatilho_a"]


def test_n_minimo():
    out = monitoring.metricas_moveis(_erros(np.ones(5), np.ones(5)), 12, 6)
    assert out.empty


def test_outlier():
    s = pd.Series(np.r_[np.random.default_rng(1).normal(1, 0.5, 120), 6.0],
                  index=pd.period_range("2010-01", periods=121, freq="M"))
    assert monitoring.alerta_outlier(s)["alerta"]

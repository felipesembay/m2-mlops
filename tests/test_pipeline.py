"""Fluxo ponta a ponta com SQLite + MLflow em diretório temporário."""

import importlib
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures" / "bcb_27810_snapshot.json"


@pytest.fixture()
def ambiente(tmp_path, monkeypatch):
    monkeypatch.setenv("M2_DATABASE_URL", f"sqlite:///{tmp_path/'m2.db'}")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path/'mlruns'}")
    from m2forecast import db
    db.engine.cache_clear()
    import m2forecast.pipeline as p
    return importlib.reload(p)


def test_bootstrap_registra_e_monitora(ambiente):
    from m2forecast import db, tracking

    res = ambiente.bootstrap(FIX, inicio_origem="2024-09")
    assert len(res["previsoes_atuais"]) == 8
    for h in (1, 3, 6, 12):
        assert tracking.versao_alias(h) is not None
        assert tracking.versao_alias(h, "benchmark") is not None
        m = tracking.carregar_modelo(h)
        out = m.predict(None)
        assert out.loc[0, "origem"] == "2026-09"
    assert not db.ler_monitoramento().empty
    # sem novidade na 2ª ingestão
    assert ambiente.ingerir(FIX)["atualizar"] is False

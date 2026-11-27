"""API em modo só leitura (produção/Vercel) contra SQLite populado pelo pipeline."""

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIX = Path(__file__).parent / "fixtures" / "bcb_27810_snapshot.json"


@pytest.fixture(scope="module")
def cliente(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    mp = pytest.MonkeyPatch()
    mp.setenv("M2_DATABASE_URL", f"sqlite:///{tmp/'m2.db'}")
    mp.setenv("MLFLOW_TRACKING_URI", f"file://{tmp/'mlruns'}")
    mp.delenv("M2_SERVIR_MLFLOW", raising=False)
    from m2forecast import db
    db.engine.cache_clear()
    import m2forecast.pipeline as p
    p = importlib.reload(p)
    p.bootstrap(FIX, inicio_origem="2025-06")
    import api.main as m
    m = importlib.reload(m)
    yield TestClient(m.app), p
    db.engine.cache_clear()
    mp.undo()


def test_projecao_prefixo_igual_ao_calculo_direto(cliente):
    c, p = cliente
    from m2forecast import db, projection

    df = db.ler_serie()
    s = df["m2_growth"].dropna()
    for modo, meses in (("melhor", 7), ("6", 5), ("benchmark", 3)):
        r = c.get(f"/api/projecao?modelo={modo}&meses={meses}")
        assert r.status_code == 200
        api = r.json()
        m = modo if modo in {"melhor", "benchmark"} else int(modo)
        direto = projection.projetar(s, float(df["valor"].iloc[-1]), m, meses)
        assert api["meses"] == meses
        assert [x["previsto"] for x in api["projecao"]] == pytest.approx(
            [x["previsto"] for x in direto["projecao"]], abs=1e-9)
        assert api["crescimento_acumulado_pct"] == pytest.approx(direto["crescimento_acumulado_pct"])


def test_endpoints_sem_mlflow(cliente):
    c, _ = cliente
    assert c.get("/health").json() == {"db": True}
    assert len(c.get("/api/serie").json()) == 297
    mods = c.get("/api/modelos").json()
    assert {r["horizonte"] for r in mods["registro"]} == {1, 3, 6, 12}
    mon = c.get("/api/monitoramento").json()
    assert mon["atual"]["1"]["rmse_validacao"] > 0
    prev = c.get("/api/prever/6").json()
    assert prev["fonte"] == "banco" and prev["origem"] == "2026-09"
    assert c.get("/api/projecao?modelo=7").status_code == 422

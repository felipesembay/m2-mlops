"""API + front-end das previsões do crescimento mensal do M2."""

from __future__ import annotations

import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Só dependências leves no import: em produção (Vercel) a API apenas lê o banco,
# que o pipeline (GitHub Actions / Airflow) mantém com projeções pré-calculadas.
from m2forecast import HORIZONTES, db, monitoring
from m2forecast.config import carregar

log = logging.getLogger("m2api")
STATIC = Path(__file__).parent / "static"

MAX_MESES = 24
MODOS = {"melhor", "benchmark", *map(str, HORIZONTES)}
SERVIR_MLFLOW = os.environ.get("M2_SERVIR_MLFLOW") == "1"

app = FastAPI(
    title="M2 Brasil — Previsões",
    description="Previsão do crescimento mensal do M2 (BCB/SGS 27810) em 1, 3, 6 e 12 meses.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

NOMES = {
    "STL_ETS": "STL + ETS",
    "RF_Bayesian": "Random Forest (OB)",
    "SARIMA_300_011": "SARIMA(3,0,0)(0,1,1)12",
    "ETS_ANA": "ETS(A,N,A)",
    "Media_Sazonal": "Média sazonal",
}


def _limpo(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _registros(df: pd.DataFrame) -> list[dict]:
    return [{k: _limpo(v) for k, v in r.items()} for r in df.to_dict("records")]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    out = {"db": db.ping()}
    if SERVIR_MLFLOW:
        from m2forecast import tracking
        try:
            tracking.cliente().search_registered_models(max_results=1)
            out["mlflow"] = True
        except Exception:  # noqa: BLE001
            out["mlflow"] = False
    return out


@app.get("/api/serie")
def serie(desde: str | None = Query(None, description="YYYY-MM; vazio = série completa")):
    df = db.ler_serie()
    if df.empty:
        return []
    df = df.dropna(subset=["m2_growth"])  # 1º mês só tem nível
    if desde:
        df = df[df["data"] >= desde]
    return _registros(df[["data", "valor", "m2_growth"]])


@app.get("/api/previsoes")
def previsoes(h: int | None = Query(None, description="Horizonte (1, 3, 6, 12)")):
    """Última previsão por horizonte + histórico previsto x realizado."""
    prev = db.ler_previsoes(h)
    if prev.empty:
        return {"atuais": [], "historico": []}
    s = db.ler_serie()
    real = dict(zip(s["data"], s["m2_growth"])) if not s.empty else {}
    prev["real"] = prev["data_alvo"].map(real)
    ultima = prev["origem"].max()
    atuais = []
    for hh, g in prev[prev["origem"] == ultima].groupby("horizonte"):
        c = g[g["papel"] == "champion"].iloc[0]
        b = g[g["papel"] == "benchmark"]
        atuais.append({
            "horizonte": int(hh),
            "origem": c["origem"],
            "data_alvo": c["data_alvo"],
            "modelo": c["modelo"],
            "modelo_nome": NOMES.get(c["modelo"], c["modelo"]),
            "model_version": c["model_version"],
            "previsto": float(c["previsto"]),
            "benchmark": float(b.iloc[0]["previsto"]) if not b.empty else None,
        })
    hist = prev.pivot_table(index=["horizonte", "data_alvo", "origem"], columns="papel",
                            values="previsto", aggfunc="last").reset_index()
    hist["real"] = hist["data_alvo"].map(real)
    modelos = prev[prev["papel"] == "champion"].set_index(["horizonte", "data_alvo"])["modelo"]
    hist["modelo"] = [modelos.get((a, b)) for a, b in zip(hist["horizonte"], hist["data_alvo"])]
    return {"atuais": atuais, "historico": _registros(hist.sort_values(["horizonte", "data_alvo"]))}


def _registro() -> pd.DataFrame:
    """Retrato do registro do MLflow gravado pelo pipeline (tabela registro_modelos)."""
    return db.ler_registro()


@app.get("/api/modelos")
def modelos():
    cfg = carregar()
    reg = []
    for r in _registros(_registro()):
        if not r.get("versao"):
            continue
        extra = json.loads(r.pop("params_json") or "{}")
        reg.append({**r, "challenger": extra.get("challenger"),
                    "modelo_nome": NOMES.get(r["spec"], r["spec"]),
                    "atualizado_em": str(r.get("atualizado_em"))})
    artigo = [{"horizonte": h, "spec": c["spec"], "modelo_nome": NOMES.get(c["spec"], c["spec"]),
               "rmse_validacao": c["rmse_validacao"]} for h, c in cfg["campeoes"].items()]
    return {"registro": reg, "artigo": artigo}


@app.get("/api/projecao")
def projecao(
    modelo: str = Query("melhor", description='"melhor", "1", "3", "6", "12" ou "benchmark"'),
    meses: int = Query(12, ge=1, le=MAX_MESES),
):
    """Trajetória de 1..meses à frente da última observação (pré-calculada pelo pipeline).

    "melhor" usa, em cada passo, o campeão do horizonte correspondente do artigo.
    """
    if modelo not in MODOS:
        raise HTTPException(422, "modelo inválido")
    df = db.ler_projecoes(modelo)
    if df.empty:
        raise HTTPException(503, "Projeções ainda não publicadas: rode o pipeline.")
    df = df[df["origem"] == df["origem"].max()].sort_values("passo").head(meses)
    linhas = []
    for r in _registros(df):
        linhas.append({
            "passo": int(r["passo"]), "data_alvo": r["data_alvo"], "previsto": r["previsto"],
            "li": r["li"], "ls": r["ls"], "modelo": r["modelo"],
            "h_modelo": None if r["h_modelo"] is None else int(r["h_modelo"]),
            "fora_validacao": bool(r["fora_validacao"]), "nivel_m2": r["nivel_m2"],
            "acumulado_pct": r["acumulado_pct"], "modelo_nome": NOMES.get(r["modelo"], r["modelo"]),
        })
    return {
        "origem": df["origem"].iloc[0],
        "modo": modelo,
        "meses": len(linhas),
        "nivel_origem": float(df["nivel_origem"].iloc[0]),
        "crescimento_acumulado_pct": linhas[-1]["acumulado_pct"],
        "media_mensal_pct": float(df["previsto"].mean()),
        "projecao": linhas,
        "calculado_em": str(df["criado_em"].iloc[0]),
    }


@app.get("/api/monitoramento")
def monitoramento_api():
    mon = db.ler_monitoramento()
    cfg = carregar()["monitoramento"]
    atual = {}
    if not mon.empty:
        atual = {str(h): d for h, d in monitoring.decisao_atual(mon).items()}
        limite_rel = cfg["gatilho_a"]["rmse_relativo_max"]
        reg = _registro()
        rmse_vig = {int(r.horizonte): r.rmse_validacao for r in reg.itertuples()
                    if r.papel == "champion" and r.rmse_validacao == r.rmse_validacao and r.rmse_validacao}
        for h, d in atual.items():
            g = mon[mon["horizonte"] == int(h)].sort_values("data_ref")
            acima = 0
            for v in reversed(g["rmse_rel"].tolist()):
                if v > limite_rel:
                    acima += 1
                else:
                    break
            ult = g.iloc[-1]
            rmse_val = rmse_vig.get(int(h)) or carregar()["campeoes"][int(h)]["rmse_validacao"]
            d.update({
                "rmse_bench": float(ult["rmse_bench"]),
                "n": int(ult["n"]),
                "meses_acima": acima,
                "rmse_validacao": rmse_val,
                "limite_b": cfg["gatilho_b"]["fator_rmse_validacao"] * rmse_val,
            })
    eventos = db.ler_eventos()
    return {
        "regra": {
            "metrica": "RMSE_12m(campeão) / RMSE_12m(Média Sazonal)",
            "gatilho_a": cfg["gatilho_a"],
            "gatilho_b": cfg["gatilho_b"],
            "janela_meses": cfg["janela_meses"],
            "n_minimo": cfg["n_minimo"],
        },
        "atual": atual,
        "serie": _registros(mon.drop(columns=["calculado_em"], errors="ignore")),
        "eventos": _registros(eventos.astype({"criado_em": str})) if not eventos.empty else [],
    }


@lru_cache(maxsize=16)
def _modelo(h: int, versao: str):
    from m2forecast import tracking
    return tracking.carregar_modelo(h)


@app.get("/api/prever/{h}")
def prever(h: int):
    """Previsão vigente do campeão para o horizonte h.

    Com M2_SERVIR_MLFLOW=1 (stack Docker) carrega models:/m2_h{h}@champion e prevê na hora;
    senão devolve a última previsão gravada pelo pipeline (mesmo modelo, mesma origem).
    """
    if h not in HORIZONTES:
        raise HTTPException(404, f"Horizonte deve ser um de {HORIZONTES}")
    if SERVIR_MLFLOW:
        from m2forecast import tracking
        mv = tracking.versao_alias(h)
        if mv is None:
            raise HTTPException(503, "Nenhum modelo campeão registrado ainda.")
        out = _modelo(h, mv.version).predict(None).iloc[0].to_dict()
        return {**out, "versao": mv.version, "modelo": tracking.nome_modelo(h), "fonte": "mlflow"}
    prev = db.ler_previsoes(h)
    prev = prev[prev["papel"] == "champion"]
    if prev.empty:
        raise HTTPException(503, "Nenhuma previsão gravada ainda.")
    r = prev.sort_values("origem").iloc[-1]
    return {"origem": r["origem"], "data_alvo": r["data_alvo"], "horizonte": h, "spec": r["modelo"],
            "previsao": float(r["previsto"]), "versao": r["model_version"], "modelo": f"m2_h{h}",
            "fonte": "banco"}

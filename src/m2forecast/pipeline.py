"""Etapas do pipeline, chamadas pelos DAGs do Airflow, scripts e testes."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from . import HORIZONTES, backtest, data, db, monitoring, projection, reoptimize, tracking
from .config import carregar
from .models import criar

log = logging.getLogger("m2forecast")


# ------------------------------------------------------------------ ingestão
def ingerir(snapshot: str | Path | None = None) -> dict:
    """Busca a série no BCB (ou snapshot), valida e grava se houver novidade."""
    cfg = carregar()["serie"]
    nivel = data.carregar_snapshot(snapshot) if snapshot else data.fetch_bcb(cfg["codigo_sgs"])
    crescimento = data.para_crescimento(nivel)
    data.validar(crescimento)

    atual = db.ler_serie()
    antiga = atual["valor"] if not atual.empty else None
    novidade = data.detectar_novidade(nivel, antiga)
    novidade["atualizar"] = bool(novidade["novos"] or novidade["revisados"])
    if novidade["atualizar"]:
        db.salvar_serie(nivel, crescimento)
    novidade["outlier"] = monitoring.alerta_outlier(
        crescimento, cfg["janela"], carregar()["monitoramento"]["alertas"]["zscore_max"])
    log.info("Ingestão: %s", novidade)
    return novidade


def serie_crescimento() -> pd.Series:
    df = db.ler_serie()
    if df.empty:
        raise RuntimeError("Base vazia: rode a ingestão/bootstrap primeiro.")
    s = df["m2_growth"].dropna()
    s.name = "m2_growth"
    data.validar(s)
    return s


# ------------------------------------------------------- treino + previsão
def treinar_e_prever(horizontes=HORIZONTES, registrar: bool = True) -> list[dict]:
    """Reajusta campeão e benchmark na última origem e grava as previsões."""
    cfg = carregar()
    serie = serie_crescimento()
    origem = serie.index[-1]
    saida = []
    for h in horizontes:
        h = int(h)
        camp = tracking.campeao_atual(h)
        bench_cfg = cfg["benchmark"]
        modelos = {
            "champion": criar(camp["spec"], h, janela=cfg["serie"]["janela"], **camp["params"]),
            "benchmark": criar(bench_cfg["spec"], h, janela=bench_cfg["janela"]),
        }
        linhas = []
        for papel, f in modelos.items():
            f.fit(serie, origem)
            versao = None
            if registrar:
                rmse_val = cfg["campeoes"][h]["rmse_validacao"] if papel == "champion" else None
                versao = tracking.registrar(
                    f, serie, papel=papel, metricas={"rmse_validacao_artigo": rmse_val}, alias="champion")
            linhas.append({
                "origem": origem, "horizonte": h, "papel": papel, "data_alvo": f.data_alvo,
                "modelo": f.spec, "model_version": versao, "previsto": f.predict(), "fonte": "producao",
            })
        db.salvar_previsoes(linhas)
        saida.extend(linhas)
    log.info("Previsões geradas para origem %s", origem)
    return [{**ln, "origem": str(ln["origem"]), "data_alvo": str(ln["data_alvo"])} for ln in saida]


# -------------------------------------------------------------- monitoramento
def rmse_validacao_vigente(h: int) -> float:
    """RMSE de referência: o do artigo ou o da última re-seleção (tag no MLflow)."""
    mv = tracking.versao_alias(h)
    if mv is not None and "rmse_validacao" in mv.tags:
        return float(mv.tags["rmse_validacao"])
    return float(carregar()["campeoes"][h]["rmse_validacao"])


def monitorar() -> dict:
    cfg = carregar()["monitoramento"]
    serie = serie_crescimento()
    erros = monitoring.tabela_erros(db.ler_previsoes(), serie)
    mon = monitoring.metricas_moveis(erros, cfg["janela_meses"], cfg["n_minimo"])
    rmse_val = {h: rmse_validacao_vigente(h) for h in HORIZONTES}
    mon = monitoring.aplicar_regras(mon, rmse_val, cfg)
    db.salvar_monitoramento(mon)
    decisao = monitoring.decisao_atual(mon)
    eventos = db.ler_eventos()
    carencia = pd.Timedelta(days=30 * cfg.get("carencia_meses", 0))
    for h, d in decisao.items():
        recentes = eventos[(eventos["horizonte"] == h)
                           & (pd.to_datetime(eventos["criado_em"], utc=True) > db.agora() - carencia)]
        d["em_carencia"] = bool(d["retreinar"] and not recentes.empty)
        if d["em_carencia"]:
            d["retreinar"] = False
            d["motivo"] += " (em carência: re-otimizado recentemente)"
    decisao["horizontes_retreino"] = [h for h, d in decisao.items() if isinstance(h, int) and d["retreinar"]]
    log.info("Monitoramento: %s", decisao)
    return decisao


# ------------------------------------------------------------ re-otimização
def reotimizar(horizonte: int, motivo: str = "gatilho") -> dict:
    cfg = carregar()
    serie = serie_crescimento()
    h = int(horizonte)
    camp = tracking.campeao_atual(h)
    try:
        res = reoptimize.reselecionar(
            serie, h, cfg["candidatos_reotimizacao"], camp,
            cfg["reotimizacao"]["meses_validacao"], cfg["reotimizacao"]["bo_n_iter"],
            cfg["reotimizacao"].get("ganho_minimo_pct", 0.0))
    except Exception as erro:
        db.registrar_evento(horizonte=h, motivo=motivo, campeao_antigo=camp["spec"], status=f"falhou: {erro}"[:32])
        raise
    venc, vig = res["vencedor"], res["vigente"]
    origem = serie.index[-1]
    f = criar(venc["spec"], h, janela=cfg["serie"]["janela"], **venc["params"]).fit(serie, origem)
    alias = "champion" if res["promover"] else "challenger"
    versao = tracking.registrar(
        f, serie, papel="champion", alias=alias,
        metricas={"rmse_reselecao": venc["RMSE"], "rmse_vigente": vig["RMSE"]},
        tags_extra={"rmse_validacao": f"{venc['RMSE']:.5f}", "motivo": motivo[:250]})
    # Sem promoção: o vencedor fica só como "challenger" (auditoria) e o campeão segue.
    if res["promover"]:
        # substitui o histórico de previsões do campeão na janela de monitoramento
        # pelo backtest do novo campeão, evitando re-disparo imediato do gatilho
        jan = cfg["monitoramento"]["janela_meses"]
        df = venc["_df"].dropna(subset=["previsto"]).tail(jan)
        db.salvar_previsoes([{
            "origem": r.origem, "horizonte": h, "papel": "champion", "data_alvo": r.data_alvo,
            "modelo": venc["spec"], "model_version": versao, "previsto": r.previsto, "fonte": "reotimizacao",
        } for r in df.itertuples()])
        db.salvar_previsoes([{
            "origem": origem, "horizonte": h, "papel": "champion", "data_alvo": f.data_alvo,
            "modelo": f.spec, "model_version": versao, "previsto": f.predict(), "fonte": "producao",
        }])
    db.registrar_evento(
        horizonte=h, motivo=f"{motivo}; challenger={venc['spec']} ({res['ganho_pct']:+.1f}%)"[:512],
        campeao_antigo=camp["spec"], campeao_novo=venc["spec"] if res["promover"] else camp["spec"],
        rmse_antigo=vig["RMSE"], rmse_novo=venc["RMSE"], status="promovido" if res["promover"] else "mantido")
    return {"horizonte": h, "promovido": res["promover"], "ganho_pct": res["ganho_pct"],
            "versao": versao, "alias": alias, "ranking": res["ranking"]}


# ------------------------------------------------- publicação (API só leitura)
MODOS_PROJECAO = ("melhor", "1", "3", "6", "12", "benchmark")


def publicar() -> dict:
    """Pré-calcula as projeções (24 meses, todos os modos) e o retrato do registro.

    A API em produção (Vercel) só lê essas tabelas: não precisa de statsmodels,
    scikit-learn nem MLflow, e responde sem reajustar modelos.
    """
    df = db.ler_serie()
    serie = serie_crescimento()
    nivel = float(df["valor"].iloc[-1])
    origem = str(serie.index[-1])
    for modo in MODOS_PROJECAO:
        m = modo if modo in {"melhor", "benchmark"} else int(modo)
        out = projection.projetar(serie, nivel, m, projection.MAX_MESES)
        db.salvar_projecoes(origem, modo, out["projecao"], nivel)

    linhas = []
    try:
        resumo = tracking.resumo_registro(HORIZONTES)
    except Exception as erro:  # noqa: BLE001 - MLflow fora: publica só o artigo
        log.warning("MLflow indisponível ao publicar registro: %s", erro)
        resumo = []
    por_chave = {(r["horizonte"], r["papel"]): r for r in resumo}
    cfg = carregar()
    for h in HORIZONTES:
        camp = tracking.campeao_atual(h) if resumo else {
            "spec": cfg["campeoes"][h]["spec"], "params": {}, "versao": None}
        for papel in ("champion", "benchmark"):
            r = por_chave.get((h, papel), {})
            ch = r.get("challenger")
            linhas.append({
                "horizonte": h, "papel": papel,
                "modelo": r.get("modelo", tracking.nome_modelo(h, papel)),
                "spec": r.get("spec") or (camp["spec"] if papel == "champion" else cfg["benchmark"]["spec"]),
                "versao": str(r["versao"]) if r.get("versao") else None,
                "origem": r.get("origem"),
                "rmse_validacao": rmse_validacao_vigente(h) if papel == "champion" and resumo
                                  else (cfg["campeoes"][h]["rmse_validacao"] if papel == "champion" else None),
                "params_json": json.dumps({"challenger": list(ch) if ch else None})[:4000],
            })
    db.salvar_registro(linhas)
    log.info("Publicado: projeções de %s (%d modos) e registro (%d linhas)", origem, len(MODOS_PROJECAO), len(linhas))
    return {"origem": origem, "modos": len(MODOS_PROJECAO), "registro": len(linhas)}


# ------------------------------------------------------------------ bootstrap
def bootstrap(snapshot: str | Path | None = None, inicio_origem: str = "2021-02", registrar: bool = True) -> dict:
    """Carga inicial: série + backtest dos campeões/benchmark + v1 no MLflow.

    Mesmo protocolo da validação do artigo: origens a partir de 2021-02
    (fim do treino), logo alvos a partir de 2021-02 + h.
    """
    cfg = carregar()
    ingerir(snapshot)
    serie = serie_crescimento()
    total = 0
    for h in HORIZONTES:
        alvos = pd.period_range(pd.Period(inicio_origem, "M") + h, serie.index[-1], freq="M")
        camp = tracking.campeao_atual(h)
        for papel, spec, janela, params in (
            ("champion", camp["spec"], cfg["serie"]["janela"], camp["params"]),
            ("benchmark", cfg["benchmark"]["spec"], cfg["benchmark"]["janela"], {}),
        ):
            df = backtest.walk_forward(serie, spec, h, alvos, janela=janela, **params).dropna(subset=["previsto"])
            total += db.salvar_previsoes([{
                "origem": r.origem, "horizonte": h, "papel": papel, "data_alvo": r.data_alvo,
                "modelo": spec, "model_version": "backtest", "previsto": r.previsto, "fonte": "backtest",
            } for r in df.itertuples()])
    prev = treinar_e_prever(registrar=registrar)
    mon = monitorar()
    pub = publicar()
    return {"previsoes_backtest": total, "previsoes_atuais": prev, "monitoramento": mon, "publicado": pub}

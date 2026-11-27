"""Persistência (SQLAlchemy Core). Postgres no Docker, SQLite em dev/testes."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, delete, select, text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from .config import database_url

meta = MetaData()

serie_m2 = Table(
    "serie_m2", meta,
    Column("data", String(7), primary_key=True),          # YYYY-MM
    Column("valor", Float, nullable=False),               # nível do M2 (R$ mil)
    Column("m2_growth", Float),                           # crescimento % m/m
    Column("atualizado_em", DateTime(timezone=True)),
)

previsoes = Table(
    "previsoes", meta,
    Column("origem", String(7), primary_key=True),
    Column("horizonte", Integer, primary_key=True),
    Column("papel", String(16), primary_key=True),        # champion | benchmark
    Column("data_alvo", String(7), nullable=False, index=True),
    Column("modelo", String(64), nullable=False),
    Column("model_version", String(32)),
    Column("previsto", Float, nullable=False),
    Column("fonte", String(16), nullable=False),          # producao | backtest | reotimizacao
    Column("criado_em", DateTime(timezone=True)),
)

monitoramento = Table(
    "monitoramento", meta,
    Column("data_ref", String(7), primary_key=True),
    Column("horizonte", Integer, primary_key=True),
    Column("n", Integer),
    Column("rmse", Float),
    Column("rmse_bench", Float),
    Column("rmse_rel", Float),
    Column("bias", Float),
    Column("gatilho_a", Boolean),
    Column("gatilho_b", Boolean),
    Column("status", String(16)),
    Column("motivo", String(512)),
    Column("calculado_em", DateTime(timezone=True)),
)

eventos_retreino = Table(
    "eventos_retreino", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("criado_em", DateTime(timezone=True)),
    Column("horizonte", Integer),
    Column("motivo", String(512)),
    Column("campeao_antigo", String(64)),
    Column("campeao_novo", String(64)),
    Column("rmse_antigo", Float),
    Column("rmse_novo", Float),
    Column("status", String(32)),                         # promovido | mantido | falhou
)

# Projeções pré-calculadas (24 meses por modo) para a API em modo só leitura (Vercel).
projecoes = Table(
    "projecoes", meta,
    Column("origem", String(7), primary_key=True),
    Column("modo", String(16), primary_key=True),         # melhor | 1 | 3 | 6 | 12 | benchmark
    Column("passo", Integer, primary_key=True),
    Column("data_alvo", String(7), nullable=False),
    Column("previsto", Float, nullable=False),
    Column("li", Float),
    Column("ls", Float),
    Column("modelo", String(64)),
    Column("h_modelo", Integer),
    Column("fora_validacao", Boolean),
    Column("nivel_m2", Float),
    Column("acumulado_pct", Float),
    Column("nivel_origem", Float),
    Column("criado_em", DateTime(timezone=True)),
)

# Retrato do registro do MLflow (campeões vigentes), também para a API só leitura.
registro_modelos = Table(
    "registro_modelos", meta,
    Column("horizonte", Integer, primary_key=True),
    Column("papel", String(16), primary_key=True),
    Column("modelo", String(64)),
    Column("spec", String(64)),
    Column("versao", String(32)),
    Column("origem", String(7)),
    Column("rmse_validacao", Float),
    Column("params_json", String(4000)),
    Column("atualizado_em", DateTime(timezone=True)),
)


def agora() -> datetime:
    return datetime.now(timezone.utc)


@lru_cache(maxsize=4)
def engine(url: str | None = None) -> Engine:
    url = url or database_url()
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    # Serverless (Vercel): sem pool entre invocações.
    extra = {"poolclass": NullPool} if url.startswith("postgresql") else {}
    eng = create_engine(url, future=True, pool_pre_ping=True, **extra)
    meta.create_all(eng)
    return eng


def _upsert(tabela: Table, linhas: list[dict], chaves: list[str]) -> int:
    """Delete+insert por chave primária (portável entre SQLite e Postgres)."""
    if not linhas:
        return 0
    with engine().begin() as con:
        for ln in linhas:
            cond = [tabela.c[k] == ln[k] for k in chaves]
            con.execute(delete(tabela).where(*cond))
        con.execute(tabela.insert(), linhas)
    return len(linhas)


# ----------------------------------------------------------------------- série
def salvar_serie(nivel: pd.Series, crescimento: pd.Series) -> int:
    ts = agora()
    linhas = [
        {"data": str(p), "valor": float(v),
         "m2_growth": float(crescimento[p]) if p in crescimento.index else None,
         "atualizado_em": ts}
        for p, v in nivel.items()
    ]
    return _upsert(serie_m2, linhas, ["data"])


def _ler(q) -> pd.DataFrame:
    with engine().connect() as con:
        df = pd.read_sql(q, con)
    df.columns = [str(c) for c in df.columns]  # quoted_name -> str
    return df


def ler_serie() -> pd.DataFrame:
    df = _ler(select(serie_m2).order_by(serie_m2.c.data))
    if not df.empty:
        df.index = pd.PeriodIndex(df["data"].astype(str), freq="M")
    return df


# ------------------------------------------------------------------- previsões
def salvar_previsoes(linhas: list[dict]) -> int:
    ts = agora()
    for ln in linhas:
        ln.setdefault("criado_em", ts)
        ln["origem"], ln["data_alvo"] = str(ln["origem"]), str(ln["data_alvo"])
    return _upsert(previsoes, linhas, ["origem", "horizonte", "papel"])


def ler_previsoes(horizonte: int | None = None) -> pd.DataFrame:
    q = select(previsoes).order_by(previsoes.c.horizonte, previsoes.c.data_alvo)
    if horizonte is not None:
        q = q.where(previsoes.c.horizonte == horizonte)
    return _ler(q)


# --------------------------------------------------------------- monitoramento
def salvar_monitoramento(df: pd.DataFrame) -> int:
    ts = agora()
    linhas = [{**r, "calculado_em": ts} for r in df.to_dict("records")]
    return _upsert(monitoramento, linhas, ["data_ref", "horizonte"])


def ler_monitoramento() -> pd.DataFrame:
    return _ler(select(monitoramento).order_by(monitoramento.c.horizonte, monitoramento.c.data_ref))


def registrar_evento(**campos) -> None:
    with engine().begin() as con:
        con.execute(eventos_retreino.insert(), [{"criado_em": agora(), **campos}])


def ler_eventos() -> pd.DataFrame:
    return _ler(select(eventos_retreino).order_by(eventos_retreino.c.id.desc()))


# ------------------------------------------------------------------ projeções
def salvar_projecoes(origem: str, modo: str, linhas: list[dict], nivel_origem: float) -> int:
    ts = agora()
    with engine().begin() as con:
        con.execute(delete(projecoes).where(projecoes.c.modo == modo))
    regs = [{**{k: ln[k] for k in ("passo", "data_alvo", "previsto", "li", "ls", "modelo", "h_modelo",
                                    "fora_validacao", "nivel_m2", "acumulado_pct")},
             "origem": str(origem), "modo": modo, "nivel_origem": nivel_origem, "criado_em": ts}
            for ln in linhas]
    return _upsert(projecoes, regs, ["origem", "modo", "passo"])


def ler_projecoes(modo: str) -> pd.DataFrame:
    return _ler(select(projecoes).where(projecoes.c.modo == modo).order_by(projecoes.c.passo))


def salvar_registro(linhas: list[dict]) -> int:
    ts = agora()
    with engine().begin() as con:
        con.execute(delete(registro_modelos))
        if linhas:
            con.execute(registro_modelos.insert(), [{**ln, "atualizado_em": ts} for ln in linhas])
    return len(linhas)


def ler_registro() -> pd.DataFrame:
    return _ler(select(registro_modelos).order_by(registro_modelos.c.horizonte, registro_modelos.c.papel))


def ping() -> bool:
    with engine().connect() as con:
        con.execute(text("select 1"))
    return True

"""Leitura de configuração (YAML + variáveis de ambiente)."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

RAIZ = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return Path(os.environ.get("M2_CONFIG_DIR", RAIZ / "configs"))


@lru_cache(maxsize=1)
def carregar() -> dict[str, Any]:
    with open(config_dir() / "modelos.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["campeoes"] = {int(h): v for h, v in cfg["campeoes"].items()}
    return cfg


def params_rf(horizonte: int) -> dict[str, Any]:
    """Hiperparâmetros + features do RF otimizado (artigo) para o horizonte."""
    campeao = carregar()["campeoes"].get(horizonte, {})
    arquivo = campeao.get("params_file", f"rf_h{horizonte}.json")
    caminho = config_dir() / arquivo
    if not caminho.exists():
        # horizonte sem RF otimizado no artigo: usa o de h=3 como ponto de partida
        caminho = config_dir() / "rf_h3.json"
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    return {"melhores_parametros": dados["melhores_parametros"], "features": dados["features"]}


def database_url() -> str:
    padrao = f"sqlite:///{RAIZ / 'data' / 'm2.db'}"
    url = os.environ.get("M2_DATABASE_URL") or os.environ.get("DATABASE_URL") or padrao
    # Neon/Heroku usam "postgres://"; SQLAlchemy quer o driver explícito.
    for pref in ("postgres://", "postgresql://"):
        if url.startswith(pref):
            url = "postgresql+psycopg2://" + url.removeprefix(pref)
    return url


def mlflow_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"file://{RAIZ / 'mlruns'}")

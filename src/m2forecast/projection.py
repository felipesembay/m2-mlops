"""Projeção de N meses à frente a partir da última observação.

Modos:
  "melhor"     -> em cada passo k usa o campeão do horizonte correspondente
                  (k=1: h=1; k=2-3: h=3; k=4-6: h=6; k>=7: h=12), como no artigo.
  1, 3, 6, 12  -> usa o campeão daquele horizonte para todos os passos.
  "benchmark"  -> Média Sazonal.

Faixa indicativa: previsão ± 1,96 x RMSE de validação do horizonte do passo
(aproximação normal; os erros do M2 têm caudas pesadas, então é só indicativa).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import HORIZONTES, tracking
from .config import carregar
from .models import criar

MAX_MESES = 24


def bucket(k: int) -> int:
    """Horizonte do artigo usado para o passo k."""
    for h in HORIZONTES:
        if k <= h:
            return h
    return HORIZONTES[-1]


def _caminho(spec: str, params: dict, janela, serie: pd.Series, n: int) -> list[float]:
    origem = serie.index[-1]
    f = criar(spec, 1, janela=janela, **params).fit(serie, origem)
    traj = f.trajetoria(n)
    if traj is not None:
        return traj
    # modelo direto (RF): um ajuste por passo, com o alvo deslocado em k meses
    return [criar(spec, k, janela=janela, **params).fit(serie, origem).predict() for k in range(1, n + 1)]


def projetar(serie: pd.Series, nivel_final: float, modo: str | int = "melhor", meses: int = 12) -> dict:
    cfg = carregar()
    meses = int(meses)
    if not 1 <= meses <= MAX_MESES:
        raise ValueError(f"meses deve estar entre 1 e {MAX_MESES}")
    janela = cfg["serie"]["janela"]
    origem = serie.index[-1]
    rmse_val = {h: cfg["campeoes"][h]["rmse_validacao"] for h in HORIZONTES}

    if modo == "benchmark":
        b = cfg["benchmark"]
        prev = _caminho(b["spec"], {}, b["janela"], serie, meses)
        usados = [(b["spec"], None)] * meses
    elif modo == "melhor":
        prev, usados, cache = [], [], {}
        for k in range(1, meses + 1):
            h = bucket(k)
            if h not in cache:
                c = tracking.campeao_atual(h)
                n_h = min(meses, max(k2 for k2 in range(1, meses + 1) if bucket(k2) == h))
                cache[h] = (c, _caminho(c["spec"], c["params"], janela, serie, n_h))
            c, traj = cache[h]
            prev.append(traj[k - 1])
            usados.append((c["spec"], h))
    else:
        h = int(modo)
        if h not in HORIZONTES:
            raise ValueError(f"modo inválido: {modo}")
        c = tracking.campeao_atual(h)
        prev = _caminho(c["spec"], c["params"], janela, serie, meses)
        usados = [(c["spec"], h)] * meses

    linhas, nivel, acumulado = [], float(nivel_final), 1.0
    for k, (p, (spec, h)) in enumerate(zip(prev, usados), start=1):
        r = rmse_val[bucket(k)]
        nivel *= 1 + p / 100
        acumulado *= 1 + p / 100
        linhas.append({
            "passo": k,
            "data_alvo": str(origem + k),
            "previsto": float(p),
            "li": float(p - 1.96 * r),
            "ls": float(p + 1.96 * r),
            "modelo": spec,
            "h_modelo": h,
            "fora_validacao": k > 12,
            "nivel_m2": nivel,
            "acumulado_pct": (acumulado - 1) * 100,
        })
    return {
        "origem": str(origem),
        "modo": str(modo),
        "meses": meses,
        "nivel_origem": float(nivel_final),
        "crescimento_acumulado_pct": (acumulado - 1) * 100,
        "media_mensal_pct": float(np.mean(prev)),
        "projecao": linhas,
    }

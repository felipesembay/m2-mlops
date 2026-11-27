"""Monitoramento das previsões e regra de retreino.

Métrica principal (por horizonte h, últimos 12 meses-alvo com valor realizado):

    R_h = RMSE_12m(campeão) / RMSE_12m(Média Sazonal)

Gatilho A — perda persistente para o benchmark: R_h > 1,0 por 3 meses seguidos.
Gatilho B — degradação absoluta: RMSE_12m(campeão) > 1,25 x RMSE de validação.
A ou B => re-otimização e re-seleção do campeão daquele horizonte.

Por que não um teste formal (DM/MCS)? O artigo mostra que, com < 60 previsões
sobrepostas, os testes não distinguem diferenças de até 15% no RMSE. Uma razão
de RMSE com persistência temporal é mais estável e interpretável.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tabela_erros(prev: pd.DataFrame, serie: pd.Series) -> pd.DataFrame:
    """Uma linha por (horizonte, data_alvo) com previsto do campeão, benchmark e realizado."""
    if prev.empty:
        return pd.DataFrame(columns=["horizonte", "data_alvo", "champion", "benchmark", "real"])
    wide = (
        prev.pivot_table(index=["horizonte", "data_alvo"], columns="papel", values="previsto", aggfunc="last")
        .reset_index()
    )
    for col in ("champion", "benchmark"):
        if col not in wide:
            wide[col] = np.nan
    realizados = {str(p): float(v) for p, v in serie.items()}
    wide["real"] = wide["data_alvo"].map(realizados)
    return wide.dropna(subset=["real", "champion", "benchmark"]).sort_values(["horizonte", "data_alvo"])


def metricas_moveis(erros: pd.DataFrame, janela: int = 12, n_minimo: int = 6) -> pd.DataFrame:
    """RMSE móvel do campeão e do benchmark, avaliado a cada mês com realizado."""
    linhas = []
    for h, g in erros.groupby("horizonte"):
        g = g.sort_values("data_alvo").reset_index(drop=True)
        e_c = (g["real"] - g["champion"]).to_numpy()
        e_b = (g["real"] - g["benchmark"]).to_numpy()
        for i in range(len(g)):
            ini = max(0, i - janela + 1)
            n = i - ini + 1
            if n < n_minimo:
                continue
            rmse = float(np.sqrt(np.mean(e_c[ini:i + 1] ** 2)))
            rmse_b = float(np.sqrt(np.mean(e_b[ini:i + 1] ** 2)))
            linhas.append({
                "data_ref": g.loc[i, "data_alvo"],
                "horizonte": int(h),
                "n": n,
                "rmse": rmse,
                "rmse_bench": rmse_b,
                "rmse_rel": rmse / rmse_b if rmse_b > 0 else np.nan,
                "bias": float(np.mean(e_c[ini:i + 1])),
            })
    return pd.DataFrame(linhas)


def aplicar_regras(mon: pd.DataFrame, rmse_validacao: dict[int, float], cfg: dict) -> pd.DataFrame:
    """Marca gatilhos A/B, alertas e status em cada linha (data_ref, horizonte)."""
    if mon.empty:
        return mon.assign(gatilho_a=[], gatilho_b=[], status=[], motivo=[])
    a_cfg, b_cfg, al = cfg["gatilho_a"], cfg["gatilho_b"], cfg["alertas"]
    saida = []
    for h, g in mon.groupby("horizonte"):
        g = g.sort_values("data_ref").copy()
        perde = g["rmse_rel"] > a_cfg["rmse_relativo_max"]
        consec = perde.astype(int).groupby((~perde).cumsum()).cumsum()
        g["gatilho_a"] = consec >= a_cfg["meses_consecutivos"]
        limite_b = b_cfg["fator_rmse_validacao"] * rmse_validacao[int(h)]
        g["gatilho_b"] = g["rmse"] > limite_b

        status, motivos = [], []
        for _, r in g.iterrows():
            m = []
            if r["gatilho_a"]:
                m.append(f"A: RMSE rel {r['rmse_rel']:.2f} > {a_cfg['rmse_relativo_max']} "
                         f"por {a_cfg['meses_consecutivos']}+ meses")
            if r["gatilho_b"]:
                m.append(f"B: RMSE {r['rmse']:.3f} > {limite_b:.3f}")
            if m:
                status.append("RETREINO")
            else:
                if r["rmse_rel"] > a_cfg["rmse_relativo_max"]:
                    m.append(f"RMSE rel {r['rmse_rel']:.2f} acima do benchmark")
                if abs(r["bias"]) > al["vies_abs_max"]:
                    m.append(f"viés {r['bias']:+.2f} p.p.")
                status.append("ALERTA" if m else "OK")
            motivos.append("; ".join(m))
        g["status"], g["motivo"] = status, motivos
        saida.append(g)
    return pd.concat(saida, ignore_index=True)


def alerta_outlier(serie: pd.Series, janela: int = 120, z_max: float = 3.0) -> dict:
    """z-score da última observação vs janela anterior (sinal de quebra estrutural)."""
    if len(serie) < janela + 1:
        return {"z": None, "alerta": False}
    ref = serie.iloc[-janela - 1:-1]
    z = float((serie.iloc[-1] - ref.mean()) / ref.std())
    return {"data": str(serie.index[-1]), "valor": float(serie.iloc[-1]), "z": z, "alerta": abs(z) > z_max}


def decisao_atual(mon: pd.DataFrame) -> dict[int, dict]:
    """Último status por horizonte."""
    out = {}
    for h, g in mon.groupby("horizonte"):
        r = g.sort_values("data_ref").iloc[-1]
        out[int(h)] = {
            "data_ref": r["data_ref"], "status": r["status"], "motivo": r["motivo"],
            "rmse": float(r["rmse"]), "rmse_rel": float(r["rmse_rel"]),
            "retreinar": bool(r["gatilho_a"] or r["gatilho_b"]),
        }
    return out

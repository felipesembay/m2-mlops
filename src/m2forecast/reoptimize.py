"""Re-otimização e re-seleção do campeão de um horizonte.

1. Otimização bayesiana do Random Forest (BayesSearchCV, espaço e splits do
   notebook, BO-01/BO-02), usando só dados ANTERIORES à janela de avaliação.
2. Walk-forward dos candidatos nos últimos `meses_validacao` meses-alvo.
3. Melhor RMSE vira "challenger"; é promovido a "champion" se superar o
   campeão vigente na mesma janela por pelo menos `ganho_minimo_pct`.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import backtest
from .features import criar_features_boost

ESPACO_RF = {
    "n_estimators": [100, 150, 200, 300],
    "max_depth": [3, 5, 8, 12, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [2, 4, 6, 8],
    "max_features": [0.5, 0.7, 1.0],
}


def criar_splits_rolling(n_amostras, janela_treino=120, tamanho_validacao=12, gap=1, n_splits=3):
    """Divisões cronológicas com janela fixa e gap (igual ao notebook)."""
    splits = []
    for k in range(n_splits, 0, -1):
        fim_val = n_amostras - (k - 1) * tamanho_validacao
        ini_val = fim_val - tamanho_validacao
        fim_tr = ini_val - gap
        ini_tr = fim_tr - janela_treino
        if ini_tr < 0:
            raise ValueError("Histórico insuficiente para os splits.")
        splits.append((np.arange(ini_tr, fim_tr), np.arange(ini_val, fim_val)))
    return splits


def otimizar_rf(serie: pd.Series, horizonte: int, ate: pd.Period, n_iter: int = 12) -> dict:
    from skopt import BayesSearchCV
    from skopt.space import Categorical

    inicio = time.perf_counter()
    dados = criar_features_boost(serie, horizonte=horizonte).dropna()
    dados = dados.loc[dados.index <= ate - horizonte]
    features = dados.columns.drop("target").tolist()
    busca = BayesSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=1, bootstrap=True),
        search_spaces={k: Categorical(v) for k, v in ESPACO_RF.items()},
        n_iter=n_iter,
        n_points=1,
        scoring="neg_root_mean_squared_error",
        cv=criar_splits_rolling(len(dados), 120, 12, horizonte, 3),
        n_jobs=1,
        random_state=42,
        refit=False,
        error_score="raise",
    )
    busca.fit(dados[features], dados["target"])
    melhores = {k: (v.item() if hasattr(v, "item") else v) for k, v in dict(busca.best_params_).items()}
    return {
        "melhores_parametros": melhores,
        "features": features,
        "rmse_cv": float(-busca.best_score_),
        "tempo_s": time.perf_counter() - inicio,
    }


def reselecionar(
    serie: pd.Series,
    horizonte: int,
    candidatos: list[str],
    campeao: dict,
    meses_validacao: int = 58,
    n_iter: int = 12,
    ganho_minimo_pct: float = 0.0,
) -> dict:
    """Retorna ranking dos candidatos e o vencedor (com params)."""
    alvos = serie.index[-meses_validacao:]
    corte = alvos[0] - 1  # último mês disponível antes da janela de avaliação
    params = {}
    if "RF_Bayesian" in candidatos:
        bo = otimizar_rf(serie.loc[:corte], horizonte, corte, n_iter=n_iter)
        params["RF_Bayesian"] = {"melhores_parametros": bo["melhores_parametros"], "features": bo["features"]}

    resultados = []
    for spec in candidatos:
        df = backtest.walk_forward(serie, spec, horizonte, alvos, **params.get(spec, {}))
        resultados.append((spec, params.get(spec, {}), df, "candidato"))
    # campeão vigente com os parâmetros atuais (pode diferir do RF re-otimizado)
    df_c = backtest.walk_forward(serie, campeao["spec"], horizonte, alvos, **campeao.get("params", {}))
    resultados.append((campeao["spec"], campeao.get("params", {}), df_c, "vigente"))

    ranking = []
    for spec, p, df, tipo in resultados:
        ok = df.dropna(subset=["previsto"])
        if ok.empty:
            continue
        m = backtest.metricas(ok).iloc[0]
        ranking.append({"spec": spec, "tipo": tipo, "params": p, "RMSE": float(m["RMSE"]),
                        "MAE": float(m["MAE"]), "Bias": float(m["Bias"]), "N": int(m["N"]),
                        "falhas": int(df["previsto"].isna().sum()), "_df": df})
    ranking.sort(key=lambda r: r["RMSE"])
    vigente = next(r for r in ranking if r["tipo"] == "vigente")
    vencedor = ranking[0]
    return {
        "ranking": [{k: v for k, v in r.items() if k != "_df"} for r in ranking],
        "vencedor": vencedor,
        "vigente": vigente,
        "ganho_pct": (1 - vencedor["RMSE"] / vigente["RMSE"]) * 100,
        "promover": (vencedor["tipo"] != "vigente"
                     and vencedor["RMSE"] < vigente["RMSE"] * (1 - ganho_minimo_pct / 100)),
    }

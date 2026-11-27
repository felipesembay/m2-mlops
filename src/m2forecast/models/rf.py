"""Random Forest com hiperparâmetros da otimização bayesiana (FINAL-02).

Previsão direta: o exemplo com origem t-h tem alvo em t, logo o treino
usa apenas exemplos com índice <= origem - h (sem vazamento).
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..features import criar_features_boost
from .base import Forecaster


class RFBayesian(Forecaster):
    spec = "RF_Bayesian"
    familia = "AM"

    def __init__(self, horizonte, janela=120, melhores_parametros=None, features=None, **kw):
        if melhores_parametros is None or features is None:
            from ..config import params_rf

            cfg = params_rf(horizonte)
            melhores_parametros = melhores_parametros or cfg["melhores_parametros"]
            features = features or cfg["features"]
        super().__init__(
            horizonte, janela, melhores_parametros=dict(melhores_parametros), features=list(features), **kw
        )

    def _estimador(self):
        return RandomForestRegressor(
            **self.params["melhores_parametros"], random_state=42, n_jobs=1, bootstrap=True
        )

    def fit(self, serie, origem):
        origem = pd.Period(origem, freq="M")
        serie = serie.loc[serie.index <= origem].dropna().astype(float)
        self._historico(serie, origem)  # valida origem e registra metadados
        features = self.params["features"]
        dados = criar_features_boost(serie, horizonte=self.horizonte)
        limite = origem - self.horizonte
        treino = dados.loc[dados.index <= limite].dropna(subset=features + ["target"]).tail(self.janela)
        if len(treino) < 60:
            raise ValueError(f"Treinamento insuficiente: {len(treino)}")
        self._X = dados.loc[[origem], features]
        if self._X.isna().any().any():
            raise ValueError(f"Features indisponíveis na origem {origem}")
        self._modelo = self._estimador().fit(treino[features], treino["target"])
        return self

    def predict(self) -> float:
        return float(self._modelo.predict(self._X)[0])

    def importancias(self) -> dict[str, float]:
        return dict(zip(self.params["features"], map(float, self._modelo.feature_importances_)))

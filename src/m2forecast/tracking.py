"""Registro dos modelos no MLflow: um registered model por horizonte.

    m2_h1, m2_h3, m2_h6, m2_h12                 -> campeões (alias "champion")
    m2_benchmark_h1 ... m2_benchmark_h12        -> Média Sazonal (alias "champion")
    alias "challenger"                          -> melhor candidato da re-otimização

Cada versão é o modelo reajustado nos últimos 120 meses de uma origem
(protocolo do artigo). A spec e os hiperparâmetros ficam em tags da versão,
então o treino mensal sempre reajusta a spec do campeão vigente.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from .config import carregar, mlflow_uri, params_rf
from .models import Forecaster

EXPERIMENTO = "m2_forecast"
REQUISITOS = ["numpy", "pandas", "statsmodels", "scikit-learn", "m2forecast"]


def nome_modelo(horizonte: int, papel: str = "champion") -> str:
    return f"m2_h{horizonte}" if papel == "champion" else f"m2_benchmark_h{horizonte}"


def cliente() -> MlflowClient:
    mlflow.set_tracking_uri(mlflow_uri())
    return MlflowClient()


class M2PyFunc(mlflow.pyfunc.PythonModel):
    """Envelope pyfunc: o artefato é o Forecaster já ajustado na origem."""

    def load_context(self, context):
        with open(context.artifacts["forecaster"], "rb") as f:
            self.forecaster: Forecaster = pickle.load(f)

    def predict(self, context, model_input=None, params=None):
        f = self.forecaster
        return pd.DataFrame([{
            "spec": f.spec,
            "origem": str(f.origem),
            "horizonte": f.horizonte,
            "data_alvo": str(f.data_alvo),
            "previsto": f.predict(),
        }])


def hash_serie(serie: pd.Series) -> str:
    bruto = pd.util.hash_pandas_object(serie.astype(float), index=False).values.tobytes()
    return hashlib.sha256(bruto + str(serie.index.max()).encode()).hexdigest()[:16]


def registrar(
    forecaster: Forecaster,
    serie: pd.Series,
    papel: str = "champion",
    metricas: dict[str, float] | None = None,
    alias: str | None = "champion",
    tags_extra: dict[str, str] | None = None,
) -> str:
    """Loga run + registra versão; devolve número da versão."""
    cli = cliente()
    mlflow.set_experiment(EXPERIMENTO)
    h = forecaster.horizonte
    nome = nome_modelo(h, papel)
    desc = forecaster.descricao()
    tags = {
        "spec": forecaster.spec,
        "familia": forecaster.familia,
        "horizonte": str(h),
        "papel": papel,
        "origem": str(forecaster.origem),
        "data_alvo": str(forecaster.data_alvo),
        "janela": str(forecaster.janela),
        "dados_ate": str(forecaster.dados_ate),
        "hash_dados": hash_serie(serie.loc[: forecaster.origem]),
        "params": json.dumps(forecaster.params, default=str),
        **(tags_extra or {}),
    }
    with mlflow.start_run(run_name=f"{nome}_{forecaster.origem}") as run, tempfile.TemporaryDirectory() as tmp:
        mlflow.set_tags(tags)
        mlflow.log_params({"spec": forecaster.spec, "horizonte": h, "janela": forecaster.janela,
                           **{f"p_{k}": v for k, v in _achatar(desc["params"]).items()}})
        mlflow.log_metric("previsao", forecaster.predict())
        if metricas:
            mlflow.log_metrics({k: float(v) for k, v in metricas.items() if v is not None})
        caminho = Path(tmp) / "forecaster.pkl"
        with open(caminho, "wb") as f:
            pickle.dump(forecaster, f)
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=M2PyFunc(),
            artifacts={"forecaster": str(caminho)},
            pip_requirements=REQUISITOS,
            registered_model_name=nome,
        )
    versao = str(info.registered_model_version)
    for k, v in tags.items():
        cli.set_model_version_tag(nome, versao, k, v)
    if alias:
        cli.set_registered_model_alias(nome, alias, versao)
    return versao


def _achatar(d: dict, pref: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_achatar(v, f"{pref}{k}_"))
        elif isinstance(v, (list, tuple)):
            out[f"{pref}{k}"] = len(v)
        else:
            out[f"{pref}{k}"] = v
    return out


def versao_alias(horizonte: int, papel: str = "champion", alias: str = "champion"):
    try:
        return cliente().get_model_version_by_alias(nome_modelo(horizonte, papel), alias)
    except MlflowException:
        return None


def campeao_atual(horizonte: int) -> dict[str, Any]:
    """Spec + params do campeão vigente (MLflow) ou do artigo (config)."""
    mv = versao_alias(horizonte)
    if mv is not None and "spec" in mv.tags:
        params = json.loads(mv.tags.get("params", "{}"))
        return {"spec": mv.tags["spec"], "params": params, "versao": mv.version, "fonte": "mlflow"}
    cfg = carregar()["campeoes"][horizonte]
    params = params_rf(horizonte) if cfg["spec"] == "RF_Bayesian" else {}
    return {"spec": cfg["spec"], "params": params, "versao": None, "fonte": "config"}


def carregar_modelo(horizonte: int, papel: str = "champion", alias: str = "champion"):
    mlflow.set_tracking_uri(mlflow_uri())
    return mlflow.pyfunc.load_model(f"models:/{nome_modelo(horizonte, papel)}@{alias}")


def resumo_registro(horizontes) -> list[dict]:
    linhas = []
    for h in horizontes:
        for papel in ("champion", "benchmark"):
            mv = versao_alias(h, papel)
            if mv is None:
                continue
            ch = versao_alias(h, papel, "challenger") if papel == "champion" else None
            linhas.append({
                "horizonte": h,
                "papel": papel,
                "modelo": nome_modelo(h, papel),
                "versao": mv.version,
                "spec": mv.tags.get("spec"),
                "origem": mv.tags.get("origem"),
                "dados_ate": mv.tags.get("dados_ate"),
                "challenger": (ch.version, ch.tags.get("spec")) if ch else None,
                "criado_em": pd.to_datetime(mv.creation_timestamp, unit="ms").isoformat(),
            })
    return linhas

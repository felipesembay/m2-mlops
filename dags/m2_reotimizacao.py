"""DAG 4 — Re-otimização (BO do RF) + re-seleção do campeão de um horizonte.

Disparada pelo monitoramento com conf={"horizonte": h, "motivo": "..."}.
Também pode ser disparada manualmente pela UI do Airflow.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from airflow.models.param import Param


@dag(
    dag_id="m2_reotimizacao",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Sao_Paulo"),
    catchup=False,
    max_active_runs=1,
    params={
        "horizonte": Param(6, type="integer", enum=[1, 3, 6, 12]),
        "motivo": Param("manual", type="string"),
    },
    default_args={"owner": "m2", "retries": 0, "execution_timeout": pendulum.duration(hours=2)},
    tags=["m2", "retreino", "mlflow"],
)
def m2_reotimizacao():
    @task
    def reotimizar(params=None) -> dict:
        from m2forecast import pipeline

        res = pipeline.reotimizar(int(params["horizonte"]), params.get("motivo", "manual"))
        for r in res["ranking"]:
            print(f"{r['spec']:<16} {r['tipo']:<10} RMSE={r['RMSE']:.4f} N={r['N']}")
        print("Promovido:", res["promovido"], "| versão:", res["versao"])
        return {"horizonte": res["horizonte"], "promovido": res["promovido"], "versao": res["versao"]}

    @task
    def atualizar_monitoramento() -> None:
        from m2forecast import pipeline

        pipeline.monitorar()
        pipeline.publicar()

    reotimizar() >> atualizar_monitoramento()


m2_reotimizacao()

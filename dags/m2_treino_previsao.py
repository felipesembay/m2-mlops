"""DAG 2 — Reajusta os campeões (h=1,3,6,12) + benchmark na última origem,
registra novas versões no MLflow (alias "champion") e grava as previsões."""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


@dag(
    dag_id="m2_treino_previsao",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Sao_Paulo"),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "m2", "retries": 1},
    tags=["m2", "treino", "mlflow"],
)
def m2_treino_previsao():
    @task
    def horizontes() -> list[int]:
        from m2forecast import HORIZONTES

        return list(HORIZONTES)

    @task(max_active_tis_per_dag=2)
    def treinar_horizonte(h: int) -> list[dict]:
        from m2forecast import pipeline

        return pipeline.treinar_e_prever([h])

    @task
    def publicar() -> dict:
        """Pré-calcula projeções e o retrato do registro para a API só leitura."""
        from m2forecast import pipeline

        return pipeline.publicar()

    disparar_monitoramento = TriggerDagRunOperator(
        task_id="disparar_monitoramento",
        trigger_dag_id="m2_monitoramento",
        wait_for_completion=False,
    )

    treinar_horizonte.expand(h=horizontes()) >> publicar() >> disparar_monitoramento


m2_treino_previsao()

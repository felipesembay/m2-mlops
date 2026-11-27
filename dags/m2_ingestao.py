"""DAG 1 — Atualiza a base do M2 (BCB/SGS 27810).

Roda todo dia às 06:00 (o BCB publica o M2 com ~1 mês de defasagem, em data
variável). Só segue para o treino quando há mês novo ou revisão de valores.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

PADRAO = {"owner": "m2", "retries": 3, "retry_delay": pendulum.duration(minutes=30)}


@dag(
    dag_id="m2_ingestao",
    schedule="0 6 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Sao_Paulo"),
    catchup=False,
    max_active_runs=1,
    default_args=PADRAO,
    tags=["m2", "dados"],
)
def m2_ingestao():
    @task
    def ingerir() -> dict:
        from m2forecast import pipeline

        return pipeline.ingerir()

    @task.short_circuit
    def houve_novidade(resultado: dict) -> bool:
        print("Novos:", resultado["novos"][-5:], "Revisados:", resultado["revisados"])
        if resultado["outlier"].get("alerta"):
            print("ALERTA: nova observação com |z| alto:", resultado["outlier"])
        return resultado["atualizar"]

    disparar_treino = TriggerDagRunOperator(
        task_id="disparar_treino",
        trigger_dag_id="m2_treino_previsao",
        conf={"origem": "{{ ti.xcom_pull(task_ids='ingerir')['ultimo'] }}"},
        wait_for_completion=False,
    )

    houve_novidade(ingerir()) >> disparar_treino


m2_ingestao()

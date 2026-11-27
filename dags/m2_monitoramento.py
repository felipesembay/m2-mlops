"""DAG 3 — Monitoramento e regra de retreino.

Por horizonte: R_h = RMSE_12m(campeão) / RMSE_12m(Média Sazonal).
Gatilho A: R_h > 1 por 3 meses seguidos. Gatilho B: RMSE_12m > 1,25 x RMSE de validação.
Se A ou B, dispara m2_reotimizacao para o horizonte.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


@dag(
    dag_id="m2_monitoramento",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Sao_Paulo"),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "m2", "retries": 1},
    tags=["m2", "monitoramento"],
)
def m2_monitoramento():
    @task
    def avaliar() -> list[dict]:
        from m2forecast import pipeline

        decisao = pipeline.monitorar()
        for h, d in decisao.items():
            if isinstance(h, int):
                print(f"h={h}: {d['status']} | RMSE={d['rmse']:.3f} | R={d['rmse_rel']:.2f} | {d['motivo']}")
        return [
            {"horizonte": h, "motivo": decisao[h]["motivo"]}
            for h in decisao["horizontes_retreino"]
        ]

    TriggerDagRunOperator.partial(
        task_id="disparar_reotimizacao",
        trigger_dag_id="m2_reotimizacao",
        wait_for_completion=False,
    ).expand(conf=avaliar())


m2_monitoramento()

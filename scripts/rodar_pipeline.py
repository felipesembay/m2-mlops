"""Pipeline completo fora do Airflow (usado pelo GitHub Actions em produção).

Mesmo encadeamento das DAGs:
    m2_ingestao -> m2_treino_previsao -> m2_monitoramento -> m2_reotimizacao (se gatilho)
e, no fim, publica projeções + registro para a API só leitura (Vercel).

    python scripts/rodar_pipeline.py              # execução diária (sai cedo sem novidade)
    python scripts/rodar_pipeline.py --bootstrap  # carga inicial (backtest desde 2021-02)
    python scripts/rodar_pipeline.py --forcar     # treina/publica mesmo sem mês novo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from m2forecast import pipeline

log = logging.getLogger("m2.rodar")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--inicio", default="2021-02", help="1ª origem do backtest no bootstrap")
    ap.add_argument("--forcar", action="store_true")
    ap.add_argument("--snapshot", default=None, help="JSON do SGS em vez da API do BCB (testes)")
    a = ap.parse_args(argv)

    if a.bootstrap:
        res = pipeline.bootstrap(a.snapshot, inicio_origem=a.inicio)
        print(json.dumps({"monitoramento": res["monitoramento"], "publicado": res["publicado"]},
                         indent=2, default=str, ensure_ascii=False))
        return 0

    nov = pipeline.ingerir(a.snapshot)                                  # DAG 1
    if not nov["atualizar"] and not a.forcar:
        log.info("Sem mês novo nem revisão: nada a fazer.")
        return 0
    pipeline.treinar_e_prever()                                         # DAG 2
    decisao = pipeline.monitorar()                                      # DAG 3
    for h in decisao["horizontes_retreino"]:                            # DAG 4
        motivo = decisao[h]["motivo"]
        log.info("Gatilho em h=%s: %s", h, motivo)
        res = pipeline.reotimizar(h, motivo)
        log.info("Re-otimização h=%s: promovido=%s ganho=%.1f%%", h, res["promovido"], res["ganho_pct"])
    if decisao["horizontes_retreino"]:
        decisao = pipeline.monitorar()
    pub = pipeline.publicar()
    print(json.dumps({"ingestao": nov, "monitoramento": decisao, "publicado": pub},
                     indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())

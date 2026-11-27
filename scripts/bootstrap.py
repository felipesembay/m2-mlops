"""Carga inicial do sistema.

1. Ingere a série do BCB (ou de um snapshot JSON).
2. Backtest walk-forward dos campeões do artigo + Média Sazonal desde
   a origem --inicio (padrão 2021-02, fim do treino no artigo), para o painel de
   monitoramento já nascer com histórico.
3. Registra v1 de cada modelo no MLflow (alias "champion") e gera as
   previsões da última origem.

Uso:
    python scripts/bootstrap.py                      # baixa do BCB
    python scripts/bootstrap.py --snapshot tests/fixtures/bcb_27810_snapshot.json
"""

import argparse
import json
import logging

from m2forecast import pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None, help="JSON do SGS salvo localmente")
    ap.add_argument("--inicio", default="2021-02", help="primeira origem do backtest")
    args = ap.parse_args()
    res = pipeline.bootstrap(args.snapshot, args.inicio)
    print(json.dumps({
        "previsoes_backtest": res["previsoes_backtest"],
        "previsoes_atuais": res["previsoes_atuais"],
        "monitoramento": {str(k): v for k, v in res["monitoramento"].items()},
    }, indent=2, ensure_ascii=False, default=str))

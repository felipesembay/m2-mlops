"""Dispara manualmente a re-otimização de um horizonte (fora do Airflow)."""

import argparse
import json
import logging

from m2forecast import pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("horizonte", type=int, choices=[1, 3, 6, 12])
    ap.add_argument("--motivo", default="manual")
    a = ap.parse_args()
    res = pipeline.reotimizar(a.horizonte, a.motivo)
    print(json.dumps(res, indent=2, default=str, ensure_ascii=False))

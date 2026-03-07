#!/usr/bin/env python3
"""Run PII detection on gold-set transcripts and evaluate against gold entities."""
from __future__ import annotations

import argparse

from src.pii.eval_gold import run_pii_eval_gold
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PII pipeline on gold set")
    parser.add_argument("--config", default="configs/pii_eval_gold.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = run_pii_eval_gold(cfg)
    print(f"Gold-set evaluation complete. Output: {out_dir}")


if __name__ == "__main__":
    main()

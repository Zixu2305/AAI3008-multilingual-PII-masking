import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii.eval import run_pii_eval
from src.utils.io import load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pii_eval_wikiann.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out = run_pii_eval(cfg)
    print(f"PII eval output: {out}")

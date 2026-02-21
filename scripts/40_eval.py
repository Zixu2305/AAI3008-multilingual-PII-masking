import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.run import run_eval
from src.utils.io import load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out = run_eval(cfg)
    print(f"Eval output: {out}")

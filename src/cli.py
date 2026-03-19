from __future__ import annotations

import argparse

from src.asr.eval import run_asr_eval
from src.asr.run import run_asr
from src.eval.run import run_eval
from src.mask.run import run_mask
from src.pii.eval import run_pii_eval
from src.pii.eval_gold import run_pii_eval_gold
from src.pii.run import run_pii
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="AAI3008 multilingual PII pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_asr = sub.add_parser("asr", help="Run ASR stage")
    p_asr.add_argument("--config", default="configs/asr.yaml")

    p_asr_eval = sub.add_parser("asr-eval", help="Run ASR evaluation stage")
    p_asr_eval.add_argument("--config", default="configs/asr_eval.yaml")

    p_pii = sub.add_parser("pii", help="Run PII detection stage")
    p_pii.add_argument("--config", default="configs/pii.yaml")

    p_pii_eval = sub.add_parser("pii-eval", help="Run WikiAnn PII/NER evaluation stage")
    p_pii_eval.add_argument("--config", default="configs/pii_eval_wikiann.yaml")

    p_pii_eval_gold = sub.add_parser("pii-eval-gold", help="Run gold-set PII evaluation")
    p_pii_eval_gold.add_argument("--config", default="configs/pii_eval_gold.yaml")

    p_mask = sub.add_parser("mask", help="Run audio masking stage")
    p_mask.add_argument("--config", default="configs/pii.yaml")

    p_eval = sub.add_parser("eval", help="Run evaluation stage")
    p_eval.add_argument("--config", default="configs/eval.yaml")

    args = parser.parse_args()
    cfg = load_yaml(args.config)

    if args.cmd == "asr":
        out = run_asr(cfg)
    elif args.cmd == "asr-eval":
        out = run_asr_eval(cfg)
    elif args.cmd == "pii":
        out = run_pii(cfg)
    elif args.cmd == "pii-eval":
        out = run_pii_eval(cfg)
    elif args.cmd == "pii-eval-gold":
        out = run_pii_eval_gold(cfg)
    elif args.cmd == "mask":
        out = run_mask(cfg)
    elif args.cmd == "eval":
        out = run_eval(cfg)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")

    print(out)


if __name__ == "__main__":
    main()

"""Command-line interface for reproducible pipeline stages."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from .config import load_config
from .data import prepare_dataset
from .evaluate import evaluate_model
from .train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecg-denoise")
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "pipeline"])
    parser.add_argument("--config", default="params.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    if args.command in {"prepare", "pipeline"}:
        path = prepare_dataset(config, "data/processed/dataset.npz")
        logging.info("Prepared dataset at %s", path)
    if args.command in {"train", "pipeline"}:
        summary = train_model(config)
        logging.info("Training summary: %s", json.dumps(summary, sort_keys=True))
    if args.command in {"evaluate", "pipeline"}:
        metrics = evaluate_model(config)
        logging.info("Evaluation metrics: %s", json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

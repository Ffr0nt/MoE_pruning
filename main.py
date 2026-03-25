#!/usr/bin/env python3
"""main.py — Stage-ориентированный CLI для pruning-проекта.

Использование::

    python -m pruning.main --stage collect
    python -m pruning.main --stage cluster
    python -m pruning.main --stage prune
"""

from __future__ import annotations

import argparse
import logging

from pruning.src.config import load_project_config
from pruning.src.workflow_steps import run_stage


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Парсит CLI-аргументы."""
    parser = argparse.ArgumentParser(description="Stage runner for pruning pipeline")
    parser.add_argument(
        "--config",
        default=None,
        help="Путь к YAML-конфигу (по умолчанию pruning/config.yaml)",
    )
    parser.add_argument(
        "--stage",
        choices=["collect", "cluster", "prune"],
        default=None,
        help="Какой шаг запустить (если не задан, берётся runtime.stage из YAML)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    # Загружаем конфиг, явно передавая stage если задан
    config = load_project_config(args.config, stage=args.stage)

    stage = args.stage or config.runtime.stage
    logger.info("[workflow] Running stage: %s", stage)
    run_stage(config, stage)


if __name__ == "__main__":
    main()

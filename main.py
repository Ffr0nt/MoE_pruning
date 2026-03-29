#!/usr/bin/env python3
"""main.py — Stage-ориентированный CLI для pruning-проекта.

Использование::

    python -m pruning.main --stage collect
    python -m pruning.main --stage profile
    python -m pruning.main --stage cluster
    python -m pruning.main --stage pruning_choice
"""

from __future__ import annotations

import argparse
import logging

from pruning.src.config import load_project_config
from pruning.src.workflow_steps import run_stage


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Парсит CLI-аргументы."""
    parser = argparse.ArgumentParser(
        description="Pipeline для обработки Mixture-of-Experts моделей с SAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры использования:\n"
            "  python -m pruning.main --stage collect          # Сбор SAE-статистик по экспертам\n"
            "  python -m pruning.main --stage profile          # Этап создания профиля датасета\n"
            "  python -m pruning.main --stage cluster          # Кластеризация экспертов\n"
            "  python -m pruning.main --stage pruning_choice   # Выбор экспертов для удаления\n"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Путь к папке config/ (по умолчанию pruning/config/)",
    )
    parser.add_argument(
        "--stage",
        choices=["collect", "profile", "cluster", "pruning_choice"],
        default=None,
        help=(
            "Выбрать этап pipeline: "
            "collect (сбор статистик), "
            "profile (создание профиля датасета), "
            "cluster (кластеризация), "
            "pruning_choice (выбор экспертов). "
            "Если не задан, используется runtime.stage из конфига."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    config = load_project_config(args.config, stage=args.stage)

    stage = args.stage or config.runtime.stage
    logger.info("=" * 80)
    logger.info("[pipeline] Running stage: %s", stage)
    logger.info("=" * 80)
    
    if stage == "profile":
        logger.info("[pipeline] Dataset profile inputs:")
        if config.profile.input_json_paths:
            for path in config.profile.input_json_paths:
                logger.info("  - %s", path)
        if config.profile.input_json_path:
            logger.info("  - %s", config.profile.input_json_path)
    
    run_stage(config, stage)


if __name__ == "__main__":
    main()

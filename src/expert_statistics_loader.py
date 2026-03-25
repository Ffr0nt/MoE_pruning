from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pruning.src.config import ProjectConfig


logger = logging.getLogger(__name__)


@dataclass
class ExpertStatistics:
    """Объединённые статистики по всем экспертам."""

    mean_activations: np.ndarray
    sum_squared_activations: np.ndarray
    count_per_expert: np.ndarray
    num_experts: int
    num_latents: int


@dataclass
class StatisticsLoadConfig:
    """Конфигурация загрузки per-expert статистик с диска."""

    data_dir: str
    num_experts: int | None = None
    suffix: str = ""


def load_expert_statistics(
    data_dir: str,
    num_experts: Optional[int] = None,
    suffix: str = "",
) -> ExpertStatistics:
    """Загружает статистики активаций с диска и объединяет в матрицы."""
    data_dir = os.path.expanduser(data_dir)
    logger.info("Loading expert statistics from: %s", data_dir)

    if num_experts is None:
        stats_path = os.path.join(data_dir, "collection_stats.npy")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"num_experts не задан, а файл метаданных не найден: {stats_path}"
            )
        meta = np.load(stats_path, allow_pickle=True).item()
        num_experts = int(meta["num_experts"])
        logger.info("num_experts inferred from metadata: %s", num_experts)
    else:
        logger.info("num_experts provided explicitly: %s", num_experts)

    means: list[np.ndarray] = []
    sum_sqs: list[np.ndarray] = []
    counts: list[int] = []

    for expert_id in range(num_experts):
        mean_path = os.path.join(data_dir, f"expert_{expert_id}_mean{suffix}.npy")
        sum_sq_path = os.path.join(data_dir, f"expert_{expert_id}_sum_squared{suffix}.npy")
        sum_path = os.path.join(data_dir, f"expert_{expert_id}_sum{suffix}.npy")

        if not os.path.exists(mean_path):
            raise FileNotFoundError(
                f"Файл статистик не найден для эксперта {expert_id}: {mean_path}"
            )
        if not os.path.exists(sum_sq_path):
            raise FileNotFoundError(
                f"Файл sum_squared не найден для эксперта {expert_id}: {sum_sq_path}"
            )

        mean = np.load(mean_path).astype(np.float64)
        sum_sq = np.load(sum_sq_path).astype(np.float64)

        if os.path.exists(sum_path):
            raw_sum = np.load(sum_path).astype(np.float64)
            nonzero = mean != 0
            if nonzero.any():
                count_estimate = int(round(float(raw_sum[nonzero].sum() / mean[nonzero].sum())))
            else:
                count_estimate = 0
        else:
            count_estimate = 0

        means.append(mean)
        sum_sqs.append(sum_sq)
        counts.append(count_estimate)

    stats_path = os.path.join(data_dir, "collection_stats.npy")
    if os.path.exists(stats_path):
        meta = np.load(stats_path, allow_pickle=True).item()
        meta_counts = meta.get("counts_per_expert", {})
        if meta_counts:
            for expert_id in range(num_experts):
                c = meta_counts.get(expert_id, meta_counts.get(str(expert_id), None))
                if c is not None:
                    counts[expert_id] = int(c)

    shapes = [m.shape for m in means]
    if len(set(shapes)) != 1:
        raise ValueError(f"Файлы экспертов имеют разные формы: {set(shapes)}")

    mean_matrix = np.stack(means, axis=0)
    sum_sq_matrix = np.stack(sum_sqs, axis=0)
    count_array = np.array(counts, dtype=np.float64)
    num_latents = mean_matrix.shape[1]

    logger.info(
        "Statistics merged: means=%s, sum_squared=%s, counts=%s",
        mean_matrix.shape,
        sum_sq_matrix.shape,
        count_array.shape,
    )

    return ExpertStatistics(
        mean_activations=mean_matrix,
        sum_squared_activations=sum_sq_matrix,
        count_per_expert=count_array,
        num_experts=num_experts,
        num_latents=num_latents,
    )


def load_expert_statistics_from_config(config: StatisticsLoadConfig) -> ExpertStatistics:
    """Загружает статистики из конфигурационного объекта."""
    return load_expert_statistics(
        data_dir=config.data_dir,
        num_experts=config.num_experts,
        suffix=config.suffix,
    )


def load_expert_statistics_from_project_config(config: ProjectConfig) -> ExpertStatistics:
    """Загружает статистики из общего ProjectConfig."""
    return load_expert_statistics(
        data_dir=config.paths.collection_dir,
        num_experts=config.pipeline.num_experts,
        suffix=config.pipeline.collection_suffix,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Any

import numpy as np

from pruning.src.config import ExpertChoiceConfig
from pruning.src.expert_statistics_loader import ExpertStatistics


ScoreOrder = Literal["ascending", "descending"]


@dataclass
class PruningDecision:
    """Решение прунинга для одного слоя."""

    experts_to_remove: list[int]
    diagnostics: dict[str, Any]


def _compute_keep_count(total: int, min_count: int, keep_ratio: float) -> int:
    """Возвращает количество экспертов, которые нужно сохранить."""
    if total <= 0:
        return 0

    k = max(int(min_count), int(np.floor(total * float(keep_ratio))))
    return min(total, k)


def _rank_and_keep(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    keep_count: int,
    order: ScoreOrder,
) -> np.ndarray:
    """Выбирает индексы сохраняемых экспертов детерминированно (с tie-break по id)."""
    if keep_count <= 0 or candidate_indices.size == 0:
        return np.array([], dtype=np.int64)

    local_scores = scores[candidate_indices]
    if order == "descending":
        ranking = np.lexsort((candidate_indices, -local_scores))
    else:
        ranking = np.lexsort((candidate_indices, local_scores))

    return candidate_indices[ranking[:keep_count]].astype(np.int64)


def _variance_per_expert(stats: ExpertStatistics) -> np.ndarray:
    """Агрегирует variance по латентам в один score на эксперта."""
    counts = stats.count_per_expert.astype(np.float64)
    means = stats.mean_activations.astype(np.float64)
    sum_sq = stats.sum_squared_activations.astype(np.float64)

    ex2 = np.zeros_like(sum_sq)
    safe = counts > 0
    if safe.any():
        ex2[safe, :] = sum_sq[safe, :] / counts[safe, None]

    var = ex2 - np.square(means)
    var = np.maximum(var, 0.0)
    score = var.mean(axis=1)

    # Эксперты без наблюдений считаем максимально ненадёжными.
    score[~safe] = np.inf
    return score


def _distance_to_nearest_centroid(labels: np.ndarray, reduced_data: np.ndarray) -> np.ndarray:
    """Считает расстояние до ближайшего центроида кластеризованных экспертов."""
    num_experts = labels.shape[0]
    centroids: list[np.ndarray] = []

    for cluster_id in sorted(int(x) for x in np.unique(labels) if int(x) >= 0):
        members = np.where(labels == cluster_id)[0]
        if members.size == 0:
            continue
        centroids.append(reduced_data[members].mean(axis=0))

    if not centroids:
        return np.full(shape=(num_experts,), fill_value=-np.inf, dtype=np.float64)

    centroid_matrix = np.stack(centroids, axis=0)
    deltas = reduced_data[:, None, :] - centroid_matrix[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    return distances.min(axis=1)


def _criterion_scores(
    criterion: str,
    stats: ExpertStatistics,
    labels: np.ndarray,
    reduced_data: np.ndarray | None,
) -> np.ndarray:
    if criterion == "load":
        return stats.count_per_expert.astype(np.float64)
    if criterion == "variance":
        return _variance_per_expert(stats)
    if criterion == "distance":
        if reduced_data is None:
            raise ValueError("criterion='distance' requires reduced_data from clustering artifact")
        return _distance_to_nearest_centroid(labels=labels, reduced_data=reduced_data)
    raise ValueError(f"Unsupported pruning criterion: {criterion}")


def _criterion_order(criterion: str) -> ScoreOrder:
    if criterion == "variance":
        # По согласованию: сохраняем экспертов с меньшей variance.
        return "ascending"
    return "descending"


def build_pruning_decision(
    *,
    labels: np.ndarray,
    stats: ExpertStatistics,
    expert_choice: ExpertChoiceConfig,
    reduced_data: np.ndarray | None = None,
) -> PruningDecision:
    """Строит список индексов экспертов для удаления в одном слое."""
    labels = np.asarray(labels).astype(int).ravel()

    if labels.shape[0] != stats.num_experts:
        raise ValueError(
            f"labels size mismatch: got {labels.shape[0]}, expected {stats.num_experts}"
        )

    if reduced_data is not None and reduced_data.shape[0] != stats.num_experts:
        raise ValueError(
            "reduced_data size mismatch: "
            f"got {reduced_data.shape[0]}, expected {stats.num_experts}"
        )

    clustered_scores = _criterion_scores(
        criterion=expert_choice.clustered.criterion,
        stats=stats,
        labels=labels,
        reduced_data=reduced_data,
    )

    unclustered_scores = _criterion_scores(
        criterion=expert_choice.unclustered.criterion,
        stats=stats,
        labels=labels,
        reduced_data=reduced_data,
    )

    keep_mask = np.zeros(stats.num_experts, dtype=bool)
    cluster_debug: dict[str, object] = {}

    clustered_ids = [int(x) for x in np.unique(labels) if int(x) >= 0]
    for cluster_id in clustered_ids:
        members = np.where(labels == cluster_id)[0]
        if not expert_choice.clustered.use:
            keep_mask[members] = True
            cluster_debug[str(cluster_id)] = {
                "cluster_size": int(members.size),
                "kept": int(members.size),
                "removed": 0,
                "reason": "clustered.use=false",
            }
            continue

        keep_count = _compute_keep_count(
            total=int(members.size),
            min_count=int(expert_choice.clustered.min_per_cluster),
            keep_ratio=float(expert_choice.clustered.keep_ratio),
        )
        kept = _rank_and_keep(
            candidate_indices=members,
            scores=clustered_scores,
            keep_count=keep_count,
            order=_criterion_order(expert_choice.clustered.criterion),
        )
        keep_mask[kept] = True
        cluster_debug[str(cluster_id)] = {
            "cluster_size": int(members.size),
            "k": int(keep_count),
            "kept": int(kept.size),
            "removed": int(members.size - kept.size),
        }

    unclustered_members = np.where(labels == -1)[0]
    unclustered_debug: dict[str, object] = {
        "total": int(unclustered_members.size),
    }

    if unclustered_members.size > 0:
        if not expert_choice.unclustered.use:
            keep_mask[unclustered_members] = True
            unclustered_debug.update(
                {
                    "k": int(unclustered_members.size),
                    "kept": int(unclustered_members.size),
                    "removed": 0,
                    "reason": "unclustered.use=false",
                }
            )
        else:
            keep_count = _compute_keep_count(
                total=int(unclustered_members.size),
                min_count=int(expert_choice.unclustered.min_experts),
                keep_ratio=float(expert_choice.unclustered.keep_ratio),
            )
            kept = _rank_and_keep(
                candidate_indices=unclustered_members,
                scores=unclustered_scores,
                keep_count=keep_count,
                order=_criterion_order(expert_choice.unclustered.criterion),
            )
            keep_mask[kept] = True
            unclustered_debug.update(
                {
                    "k": int(keep_count),
                    "kept": int(kept.size),
                    "removed": int(unclustered_members.size - kept.size),
                }
            )
    else:
        unclustered_debug.update({"k": 0, "kept": 0, "removed": 0})

    experts_to_remove = np.where(~keep_mask)[0].astype(np.int64).tolist()
    diagnostics = {
        "clustered": {
            "enabled": bool(expert_choice.clustered.use),
            "criterion": expert_choice.clustered.criterion,
            "clusters": cluster_debug,
        },
        "unclustered": {
            "enabled": bool(expert_choice.unclustered.use),
            "criterion": expert_choice.unclustered.criterion,
            **unclustered_debug,
        },
        "totals": {
            "num_experts": int(stats.num_experts),
            "kept": int(keep_mask.sum()),
            "removed": int((~keep_mask).sum()),
            "removed_ratio": float((~keep_mask).sum() / max(1, stats.num_experts)),
        },
    }

    return PruningDecision(
        experts_to_remove=experts_to_remove,
        diagnostics=diagnostics,
    )

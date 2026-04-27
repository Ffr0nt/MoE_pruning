from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from pruning.src.config import PruningChoiceConfig
from pruning.src.expert_statistics_loader import ExpertStatistics


logger = logging.getLogger(__name__)


@dataclass
class PruningChoiceDecision:
    """Решение выбора экспертов для одного слоя."""

    experts_to_remove: list[int]
    diagnostics: dict[str, Any]


def _rank_indices_by_score(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    *,
    descending: bool,
) -> np.ndarray:
    if candidate_indices.size == 0:
        return np.array([], dtype=np.int64)

    local_scores = scores[candidate_indices]
    primary = -local_scores if descending else local_scores
    ranking = np.lexsort((candidate_indices, primary))
    return candidate_indices[ranking].astype(np.int64)


def _select_by_score(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    top_n: int,
    *,
    descending: bool,
) -> np.ndarray:
    if top_n <= 0 or candidate_indices.size == 0:
        return np.array([], dtype=np.int64)

    ranked = _rank_indices_by_score(
        candidate_indices=candidate_indices,
        scores=scores,
        descending=descending,
    )
    return ranked[: min(int(top_n), int(ranked.size))]


def _variance_per_expert(stats: ExpertStatistics) -> np.ndarray:
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

    # Эксперты без наблюдений не должны доминировать в max-variance выборе.
    score[~safe] = -np.inf
    return score


def _cosine_similarity_to_anchor(vectors: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape={vectors.shape}")

    anchor = np.asarray(anchor, dtype=np.float64).ravel()
    if vectors.shape[1] != anchor.shape[0]:
        raise ValueError(
            "anchor size mismatch: "
            f"vectors have {vectors.shape[1]} dims, anchor has {anchor.shape[0]}"
        )

    anchor_norm = float(np.linalg.norm(anchor))
    if anchor_norm == 0.0:
        raise ValueError("anchor_vector has zero norm; cosine similarity is undefined")

    vectors = np.asarray(vectors, dtype=np.float64)
    vec_norms = np.linalg.norm(vectors, axis=1)
    denom = vec_norms * anchor_norm

    similarities = np.full(shape=(vectors.shape[0],), fill_value=-1.0, dtype=np.float64)
    safe = denom > 0
    if safe.any():
        similarities[safe] = (vectors[safe] @ anchor) / denom[safe]
    return similarities


def _cluster_similarity_to_anchor(
    labels: np.ndarray,
    expert_vectors: np.ndarray,
    anchor_vector: np.ndarray,
) -> dict[int, float]:
    result: dict[int, float] = {}
    cluster_ids = [int(x) for x in np.unique(labels) if int(x) >= 0]

    for cluster_id in cluster_ids:
        members = np.where(labels == cluster_id)[0]
        if members.size == 0:
            result[cluster_id] = -1.0
            continue
        centroid = expert_vectors[members].mean(axis=0, dtype=np.float64)
        similarity = float(_cosine_similarity_to_anchor(centroid[None, :], anchor_vector)[0])
        result[cluster_id] = similarity

    return result


def _compute_keep_top_ratio_count(total: int, ratio: float) -> int:
    if total <= 0:
        return 0
    k = int(np.floor(float(total) * float(ratio)))
    if ratio > 0.0 and k == 0:
        k = 1
    return min(total, k)


def _resolve_count_based_removal_percent(
    pruning_choice: PruningChoiceConfig,
    hook_layer: int,
) -> float:
    if pruning_choice.percent_mode == "global":
        return float(pruning_choice.global_removal_percent)

    if pruning_choice.percent_mode == "per_layer":
        per_layer = pruning_choice.per_layer_removal_percent or {}
        if hook_layer in per_layer:
            return float(per_layer[hook_layer])
        logger.warning(
            "[pruning_choice][layer=%s] percent_mode=per_layer, but layer is missing in "
            "per_layer_removal_percent. Using 0%% removal for this layer.",
            hook_layer,
        )
        return 0.0

    raise ValueError(f"Unknown percent_mode: {pruning_choice.percent_mode}")


def build_pruning_choice_decision(
    *,
    labels: np.ndarray | None,
    stats: ExpertStatistics,
    pruning_choice: PruningChoiceConfig,
    anchor_vector: np.ndarray | None,
    hook_layer: int,
    reduced_data: np.ndarray | None = None,
) -> PruningChoiceDecision:
    """Строит список индексов экспертов для удаления в одном слое."""
    if pruning_choice.strategy == "count_based":
        counts = np.asarray(stats.count_per_expert, dtype=np.float64).ravel()
        if counts.shape[0] != stats.num_experts:
            raise ValueError(
                "count_per_expert size mismatch: "
                f"got {counts.shape[0]}, expected {stats.num_experts}"
            )

        percent = _resolve_count_based_removal_percent(pruning_choice, hook_layer)
        remove_count = int(np.floor(float(stats.num_experts) * percent / 100.0))

        all_indices = np.arange(stats.num_experts, dtype=np.int64)
        ranked_for_removal = _rank_indices_by_score(
            candidate_indices=all_indices,
            scores=counts,
            descending=False,
        )

        to_remove = ranked_for_removal[:remove_count]
        keep_mask = np.ones(stats.num_experts, dtype=bool)
        keep_mask[to_remove] = False

        experts_to_remove = to_remove.astype(np.int64).tolist()
        diagnostics = {
            "count_based": {
                "enabled": True,
                "criterion": "bottom_x_percent_by_count",
                "percent_mode": pruning_choice.percent_mode,
                "used_percent": float(percent),
                "global_removal_percent": float(pruning_choice.global_removal_percent),
                "remove_count": int(remove_count),
                "min_count": float(np.min(counts)) if counts.size else 0.0,
                "max_count": float(np.max(counts)) if counts.size else 0.0,
                "threshold_count": float(counts[to_remove[-1]]) if to_remove.size else None,
            },
            "clustered": {
                "enabled": False,
                "criterion": "skipped_for_count_based",
            },
            "unclustered": {
                "enabled": False,
                "criterion": "skipped_for_count_based",
                "total": 0,
                "k": 0,
                "kept": 0,
                "removed": 0,
                "locked": 0,
            },
            "totals": {
                "num_experts": int(stats.num_experts),
                "kept": int(keep_mask.sum()),
                "removed": int((~keep_mask).sum()),
                "removed_ratio": float((~keep_mask).sum() / max(1, stats.num_experts)),
            },
        }

        return PruningChoiceDecision(
            experts_to_remove=experts_to_remove,
            diagnostics=diagnostics,
        )

    if labels is None:
        raise ValueError("labels must be provided for cosine_anchor strategy")
    if anchor_vector is None:
        raise ValueError("anchor_vector must be provided for cosine_anchor strategy")

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

    expert_vectors = np.asarray(stats.mean_activations, dtype=np.float64)
    variance_scores = _variance_per_expert(stats)
    cosine_scores = _cosine_similarity_to_anchor(expert_vectors, anchor_vector)

    keep_mask = np.zeros(stats.num_experts, dtype=bool)

    # Clustered: сохраняем top-R% кластеров полностью, в остальных по 1 эксперту с max variance.
    cluster_debug: dict[str, object] = {}
    clustered_ids = [int(x) for x in np.unique(labels) if int(x) >= 0]

    if clustered_ids and pruning_choice.clustered.use:
        cluster_similarity = _cluster_similarity_to_anchor(
            labels=labels,
            expert_vectors=expert_vectors,
            anchor_vector=anchor_vector,
        )
        cluster_id_array = np.array(clustered_ids, dtype=np.int64)
        cluster_score_array = np.array(
            [cluster_similarity[cid] for cid in clustered_ids],
            dtype=np.float64,
        )

        ranked_clusters = cluster_id_array[
            np.lexsort((cluster_id_array, -cluster_score_array))
        ]
        keep_clusters_count = _compute_keep_top_ratio_count(
            total=len(clustered_ids),
            ratio=float(pruning_choice.clustered.keep_top_cluster_ratio or 0.0),
        )
        full_keep_clusters = set(ranked_clusters[:keep_clusters_count].tolist())

        for cluster_id in clustered_ids:
            members = np.where(labels == cluster_id)[0]
            if cluster_id in full_keep_clusters:
                keep_mask[members] = True
                cluster_debug[str(cluster_id)] = {
                    "cluster_size": int(members.size),
                    "kept": int(members.size),
                    "removed": 0,
                    "mode": "full_keep",
                    "cluster_similarity": float(cluster_similarity[cluster_id]),
                }
                continue

            kept = _select_by_score(
                candidate_indices=members,
                scores=variance_scores,
                top_n=1,
                descending=True,
            )
            keep_mask[kept] = True
            cluster_debug[str(cluster_id)] = {
                "cluster_size": int(members.size),
                "kept": int(kept.size),
                "removed": int(members.size - kept.size),
                "mode": "single_representative",
                "cluster_similarity": float(cluster_similarity[cluster_id]),
                "representative": int(kept[0]) if kept.size else None,
            }
    else:
        for cluster_id in clustered_ids:
            members = np.where(labels == cluster_id)[0]
            keep_mask[members] = True
            cluster_debug[str(cluster_id)] = {
                "cluster_size": int(members.size),
                "kept": int(members.size),
                "removed": 0,
                "mode": "clustered.use=false",
            }

    # Unclustered: сначала lock top-N variance, затем удаляем bottom X% по cosine.
    unclustered_members = np.where(labels == -1)[0]
    unclustered_debug: dict[str, object] = {"total": int(unclustered_members.size)}

    if unclustered_members.size > 0:
        if not pruning_choice.unclustered.use:
            keep_mask[unclustered_members] = True
            unclustered_debug.update(
                {
                    "k": int(unclustered_members.size),
                    "kept": int(unclustered_members.size),
                    "removed": 0,
                    "locked": 0,
                    "reason": "unclustered.use=false",
                }
            )
        else:
            lock_n = int(pruning_choice.unclustered.variance_lock_top_n or 0)
            locked = _select_by_score(
                candidate_indices=unclustered_members,
                scores=variance_scores,
                top_n=lock_n,
                descending=True,
            )
            remaining = np.setdiff1d(unclustered_members, locked, assume_unique=False)

            bottom_x = float(pruning_choice.unclustered.bottom_x_percent or 0.0)
            remove_count = int(np.floor(float(remaining.size) * bottom_x / 100.0))
            to_remove = _select_by_score(
                candidate_indices=remaining,
                scores=cosine_scores,
                top_n=remove_count,
                descending=False,
            )

            kept = np.setdiff1d(unclustered_members, to_remove, assume_unique=False)
            keep_mask[kept] = True

            unclustered_debug.update(
                {
                    "k": int(kept.size),
                    "kept": int(kept.size),
                    "removed": int(unclustered_members.size - kept.size),
                    "locked": int(locked.size),
                    "removed_by_bottom_x": int(to_remove.size),
                    "bottom_x_percent": float(bottom_x),
                }
            )
    else:
        unclustered_debug.update({"k": 0, "kept": 0, "removed": 0, "locked": 0})

    experts_to_remove = np.where(~keep_mask)[0].astype(np.int64).tolist()
    diagnostics = {
        "clustered": {
            "enabled": bool(pruning_choice.clustered.use),
            "criterion": "cosine_anchor_cluster_similarity+max_variance_representative",
            "keep_top_cluster_ratio": float(pruning_choice.clustered.keep_top_cluster_ratio or 0.0),
            "clusters": cluster_debug,
        },
        "unclustered": {
            "enabled": bool(pruning_choice.unclustered.use),
            "criterion": "variance_lock_then_bottom_x_cosine",
            "variance_lock_top_n": int(pruning_choice.unclustered.variance_lock_top_n or 0),
            **unclustered_debug,
        },
        "totals": {
            "num_experts": int(stats.num_experts),
            "kept": int(keep_mask.sum()),
            "removed": int((~keep_mask).sum()),
            "removed_ratio": float((~keep_mask).sum() / max(1, stats.num_experts)),
        },
    }

    return PruningChoiceDecision(
        experts_to_remove=experts_to_remove,
        diagnostics=diagnostics,
    )

# Feature specificity metrics library

from typing import Dict, Optional, Tuple

import numpy as np


def calculate_fold_change(
    mean_activations: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Calculate fold change for SAE features per expert."""
    num_experts, _ = mean_activations.shape
    total_sum = mean_activations.sum(axis=0)
    fold_change = np.zeros_like(mean_activations)

    for e in range(num_experts):
        mu_e = mean_activations[e]
        sum_others = total_sum - mu_e
        mean_others = sum_others / (num_experts - 1)
        fold_change[e] = mu_e / (mean_others + epsilon)

    return fold_change


def calculate_normalized_mean_difference(
    mean_activations: np.ndarray,
    sum_squared_activations: np.ndarray,
    count_per_expert: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Calculate normalized mean difference (Cohen's d) for SAE features per expert."""
    total_count = count_per_expert.sum()
    weighted_sum = (mean_activations * count_per_expert[:, np.newaxis]).sum(axis=0)
    global_mean = weighted_sum / total_count
    total_sum_sq = sum_squared_activations.sum(axis=0)
    global_variance = (total_sum_sq / total_count) - (global_mean ** 2)
    global_std = np.sqrt(np.maximum(global_variance, 0.0))
    normalized_diff = (mean_activations - global_mean) / (global_std + epsilon)
    return normalized_diff


def get_specific_features(
    specificity_scores: np.ndarray,
    expert_ids: Optional[np.ndarray] = None,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    descending: bool = True,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Extract features with high specificity scores for given experts."""
    if top_k is None and threshold is None:
        raise ValueError("At least one of 'top_k' or 'threshold' must be specified")

    num_experts, num_latents = specificity_scores.shape

    if expert_ids is None:
        expert_ids_array = np.arange(num_experts)
    else:
        expert_ids_array = expert_ids

    results = {}

    for expert_id in expert_ids_array:
        scores = specificity_scores[expert_id]

        if threshold is not None:
            if descending:
                mask = scores >= threshold
            else:
                mask = scores <= threshold
            valid_indices = np.where(mask)[0]
            valid_scores = scores[mask]
        else:
            valid_indices = np.arange(num_latents)
            valid_scores = scores

        sorted_idx = np.argsort(valid_scores)[::-1] if descending else np.argsort(valid_scores)
        feature_indices = valid_indices[sorted_idx]
        feature_scores = valid_scores[sorted_idx]

        if top_k is not None:
            feature_indices = feature_indices[:top_k]
            feature_scores = feature_scores[:top_k]

        results[int(expert_id)] = (feature_indices, feature_scores)

    return results


def compute_specificity_stats(specificity_scores: np.ndarray) -> dict[str, object]:
    """Вычисляет базовые статистики по матрице специфичности."""
    return {
        "min": float(specificity_scores.min()),
        "max": float(specificity_scores.max()),
        "mean": float(specificity_scores.mean()),
        "median": float(np.median(specificity_scores)),
        "features_gt_1_per_expert": (specificity_scores > 1).sum(axis=1),
        "features_gt_10_per_expert": (specificity_scores > 10).sum(axis=1),
    }

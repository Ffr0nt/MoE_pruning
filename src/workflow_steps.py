from __future__ import annotations

import logging
import os

import numpy as np

from pruning.src.cluster_experts import cluster_experts
from pruning.src.collect_expert_statistics import collect_statistics
from pruning.src.config import (
    ProjectConfig,
    get_layer_clustering_dir,
    get_layer_collection_dir,
    get_layer_pruning_dir,
)
from pruning.src.dataset_profile import get_dataset_profile
from pruning.src.expert_pruning import build_pruning_decision
from pruning.src.expert_statistics_loader import load_expert_statistics
from pruning.src.feature_selection import (
    calculate_normalized_mean_difference,
    get_specific_features,
)
from pruning.src.pipeline_artifact_store import (
    save_clustering_artifact,
    save_pruning_plan,
    save_pruning_plan_placeholder,
)


logger = logging.getLogger(__name__)


def validate_collection_artifacts(config: ProjectConfig, hook_layer: int) -> str:
    """Проверяет наличие артефактов шага collect перед cluster."""
    collection_dir = get_layer_collection_dir(config, hook_layer)
    if not os.path.isdir(collection_dir):
        raise FileNotFoundError(
            "Не найден каталог статистик collection: "
            f"{collection_dir}. Сначала запустите: python -m pruning.main --stage collect"
        )

    stats_path = os.path.join(collection_dir, "collection_stats.npy")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            "Не найден файл метаданных collection_stats.npy в "
            f"{collection_dir}. Сначала запустите: python -m pruning.main --stage collect"
        )

    return collection_dir


def validate_clustering_artifacts(config: ProjectConfig, hook_layer: int) -> str:
    """Проверяет наличие артефактов шага cluster перед expert_choice/prune."""
    clustering_dir = get_layer_clustering_dir(config, hook_layer)
    if not os.path.isdir(clustering_dir):
        raise FileNotFoundError(
            "Не найден каталог кластеризации: "
            f"{clustering_dir}. Сначала запустите: python -m pruning.main --stage cluster"
        )

    required_files = [
        "labels.npy",
        "reduced_data.npy",
        "selected_columns.npy",
        "top_indices.npy",
        "clustering_summary.json",
    ]
    missing = [
        filename
        for filename in required_files
        if not os.path.exists(os.path.join(clustering_dir, filename))
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            "Не найдены обязательные артефакты кластеризации "
            f"в {clustering_dir}: {missing_str}. "
            "Сначала запустите: python -m pruning.main --stage cluster"
        )

    return clustering_dir


def collect_top_indices(nmd: np.ndarray, top_k: int) -> np.ndarray:
    """Возвращает объединение top-k признаков по всем экспертам."""
    per_expert = get_specific_features(nmd, top_k=top_k)

    all_indices: list[int] = []
    for indices, _ in per_expert.values():
        all_indices.extend(indices.tolist())

    return np.unique(all_indices)


def run_collect_step(config: ProjectConfig) -> None:
    """Шаг 1: сбор и сохранение статистик по экспертам."""
    logger.info(
        "[collect] Запуск сбора статистик для слоёв %s в корень: %s",
        config.collection.hook_layers,
        config.paths.collection_dir,
    )
    os.makedirs(config.paths.collection_dir, exist_ok=True)
    collect_statistics(config)


def run_profile_step(config: ProjectConfig) -> None:
    """Шаг создания профиля датасета: расчёт dataset-level SAE профиля."""
    logger.info(
        "[создание профиля датасета] Запуск для слоёв %s в корень: %s",
        config.collection.hook_layers,
        config.paths.profile_dir,
    )
    os.makedirs(config.paths.profile_dir, exist_ok=True)
    get_dataset_profile(config)


def run_cluster_step(config: ProjectConfig, hook_layer: int):
    """Шаг 2: кластеризация экспертов и сохранение артефакта."""
    collection_dir = validate_collection_artifacts(config, hook_layer)
    clustering_dir = get_layer_clustering_dir(config, hook_layer)

    logger.info("[cluster][layer=%s] Загрузка статистик из: %s", hook_layer, collection_dir)
    stats = load_expert_statistics(
        collection_dir,
        num_experts=config.pipeline.num_experts,
        suffix=config.pipeline.collection_suffix,
    )

    logger.info(
        "      num_experts=%s, num_latents=%s, total_tokens=%s",
        stats.num_experts,
        stats.num_latents,
        int(stats.count_per_expert.sum()),
    )

    logger.info(
        "[cluster][layer=%s] Расчёт NMD и отбор топ-%s признаков на эксперта...",
        hook_layer,
        config.pipeline.top_k,
    )
    nmd = calculate_normalized_mean_difference(
        mean_activations=stats.mean_activations,
        sum_squared_activations=stats.sum_squared_activations,
        count_per_expert=stats.count_per_expert,
    )

    top_indices = collect_top_indices(nmd, top_k=config.pipeline.top_k)

    logger.info(
        "[cluster][layer=%s] Уникальных признаков после объединения: %s",
        hook_layer,
        len(top_indices),
    )

    logger.info("[cluster][layer=%s] Кластеризация экспертов (PCA + HDBSCAN)...", hook_layer)
    result = cluster_experts(
        expert_profiles=stats.mean_activations,
        top_indices=top_indices,
        config=config.cluster,
    )

    os.makedirs(clustering_dir, exist_ok=True)
    save_clustering_artifact(
        output_dir=clustering_dir,
        result=result,
        top_indices=top_indices,
        hook_layer=hook_layer,
        stage_config={
            "hook_layers": config.collection.hook_layers,
            "top_k": config.pipeline.top_k,
            "collection_suffix": config.pipeline.collection_suffix,
            "cluster": {
                "pca_n_components": config.cluster.pca_n_components,
                "pca_whiten": config.cluster.pca_whiten,
                "pca_random_state": config.cluster.pca_random_state,
                "normalize_before_pca": config.cluster.normalize_before_pca,
                "hdbscan_min_cluster_size": config.cluster.hdbscan_min_cluster_size,
                "hdbscan_min_samples": config.cluster.hdbscan_min_samples,
                "hdbscan_metric": config.cluster.hdbscan_metric,
                "hdbscan_cluster_selection": config.cluster.hdbscan_cluster_selection,
            },
        },
    )
    logger.info("[cluster][layer=%s] Артефакт сохранён в: %s", hook_layer, clustering_dir)

    logger.info("=== Результат слоя %s ===", hook_layer)
    logger.info("  Кластеров: %s", result.n_clusters)
    logger.info("  Шумовых точек: %.1f%%", result.noise_ratio * 100)
    logger.info("  PCA-компонент: %s", result.pca_n_components)
    logger.info("  Использованных признаков: %s", len(result.selected_columns))
    logger.info("  Метки кластеров: %s", result.labels)

    return result


def run_expert_choice_step(config: ProjectConfig, hook_layer: int) -> None:
    """Шаг 3: построение expert_choice-плана удаления экспертов для слоя."""
    collection_dir = validate_collection_artifacts(config, hook_layer)
    clustering_dir = validate_clustering_artifacts(config, hook_layer)
    pruning_dir = get_layer_pruning_dir(config, hook_layer)

    labels = np.load(os.path.join(clustering_dir, "labels.npy"))
    reduced_data = np.load(os.path.join(clustering_dir, "reduced_data.npy"))

    stats = load_expert_statistics(
        collection_dir,
        num_experts=config.pipeline.num_experts,
        suffix=config.pipeline.collection_suffix,
    )

    decision = build_pruning_decision(
        labels=labels,
        reduced_data=reduced_data,
        stats=stats,
        expert_choice=config.expert_choice,
    )

    target_layer = config.expert_choice.target_layer
    if target_layer is None:
        target_layer = hook_layer

    plan_path = save_pruning_plan(
        output_dir=pruning_dir,
        target_layer=target_layer,
        hook_layer=hook_layer,
        strategy=config.expert_choice.strategy,
        experts_to_remove=decision.experts_to_remove,
        diagnostics=decision.diagnostics,
    )

    removed = decision.diagnostics["totals"]["removed"]
    removed_ratio = decision.diagnostics["totals"]["removed_ratio"]
    logger.info(
        "[expert_choice][layer=%s] План сохранён: %s (removed=%s, removed_ratio=%.3f)",
        hook_layer,
        plan_path,
        removed,
        removed_ratio,
    )


def run_prune_step(config: ProjectConfig, hook_layer: int) -> None:
    """Отдельный этап prune: пока заглушка для будущего удаления весов."""
    validate_clustering_artifacts(config, hook_layer)
    pruning_dir = get_layer_pruning_dir(config, hook_layer)
    target_layer = config.pruning.target_layer
    if target_layer is None:
        target_layer = hook_layer

    plan_path = save_pruning_plan_placeholder(
        output_dir=pruning_dir,
        target_layer=target_layer,
        hook_layer=hook_layer,
        strategy=config.pruning.strategy,
    )
    logger.warning(
        "[prune][layer=%s] Этап prune пока не реализован. Заглушка сохранена: %s",
        hook_layer,
        plan_path,
    )


def run_stage(config: ProjectConfig, stage: str) -> None:
    """Запускает один выбранный шаг workflow."""
    if stage == "collect":
        run_collect_step(config)
        return
    if stage == "profile":
        run_profile_step(config)
        return
    if stage == "cluster":
        for hook_layer in config.collection.hook_layers:
            run_cluster_step(config, hook_layer)
        return
    if stage == "expert_choice":
        for hook_layer in config.collection.hook_layers:
            run_expert_choice_step(config, hook_layer)
        return
    if stage == "prune":
        for hook_layer in config.collection.hook_layers:
            run_prune_step(config, hook_layer)
        return
    raise ValueError(f"Unknown stage: {stage}")

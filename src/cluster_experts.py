from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Конфигурация пайплайна кластеризации.

    Attributes:
        pca_n_components:           Число PCA-компонент после отбора признаков.
        pca_whiten:                 Включить whitening в PCA.
        pca_random_state:           Seed для PCA.
        normalize_before_pca:       Применить L2-нормализацию перед PCA.
        hdbscan_min_cluster_size:   Минимальный размер кластера для HDBSCAN.
        hdbscan_min_samples:        Минимальное число соседей для core-точки.
        hdbscan_metric:             Метрика расстояния для HDBSCAN.
        hdbscan_cluster_selection:  Метод выбора кластеров ('eom' или 'leaf').
    """

    pca_n_components: int = 10
    pca_whiten: bool = True
    pca_random_state: int = 21
    normalize_before_pca: bool = False
    hdbscan_min_cluster_size: int = 2
    hdbscan_min_samples: int = 1
    hdbscan_metric: Literal["euclidean", "cosine", "manhattan"] = "euclidean"
    hdbscan_cluster_selection: Literal["eom", "leaf"] = "eom"


@dataclass
class ClusterResult:
    """Результат кластеризации.

    Attributes:
        labels:             Метки кластеров (N,). -1 означает шумовую точку.
        n_clusters:         Число найденных кластеров (без учёта шума).
        noise_ratio:        Доля шумовых точек.
        pca_n_components:   Реально использованное число PCA-компонент.
        reduced_data:       Данные после PCA формы (N, pca_n_components).
        selected_columns:   Индексы столбцов в `expert_profiles`, использованных при кластеризации.
    """

    labels: np.ndarray
    n_clusters: int
    noise_ratio: float
    pca_n_components: int
    reduced_data: np.ndarray
    selected_columns: np.ndarray


def cluster_experts(
    expert_profiles: np.ndarray,
    top_indices: np.ndarray,
    config: ClusterConfig | None = None,
) -> ClusterResult:
    """Кластеризует профили экспертов по заданному подмножеству признаков SAE."""
    try:
        import hdbscan as _hdbscan
    except ImportError as exc:
        raise ImportError(
            "Для кластеризации требуется пакет hdbscan. "
            "Установите его: pip install hdbscan"
        ) from exc

    if config is None:
        config = ClusterConfig()

    expert_profiles = np.asarray(expert_profiles, dtype=np.float64)
    top_indices = np.asarray(top_indices, dtype=int).ravel()

    logger.info(
        "Start clustering: expert_profiles=%s, raw_top_indices=%s",
        expert_profiles.shape,
        top_indices.shape,
    )

    if expert_profiles.ndim != 2:
        raise ValueError(
            f"expert_profiles должен быть 2-D массивом (N, D), "
            f"получено shape={expert_profiles.shape}"
        )
    if top_indices.ndim != 1:
        raise ValueError("top_indices должен быть 1-D массивом индексов")

    max_idx = top_indices.max()
    if max_idx >= expert_profiles.shape[1]:
        raise ValueError(
            f"top_indices содержит индекс {max_idx}, "
            f"но expert_profiles имеет только {expert_profiles.shape[1]} столбцов"
        )

    selected_columns = np.unique(top_indices)
    X = expert_profiles[:, selected_columns]

    if config.normalize_before_pca:
        X = normalize(X, norm="l2")

    n_samples, n_features = X.shape
    actual_pca = min(config.pca_n_components, n_features, max(1, n_samples - 1))
    pca = PCA(
        n_components=actual_pca,
        whiten=config.pca_whiten,
        random_state=config.pca_random_state,
    )
    X_reduced = pca.fit_transform(X)

    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric=config.hdbscan_metric,
        cluster_selection_method=config.hdbscan_cluster_selection,
    )
    labels: np.ndarray = clusterer.fit_predict(X_reduced)

    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    noise_ratio = float((labels == -1).sum()) / len(labels)

    logger.info(
        "Clustering finished: n_clusters=%s, noise_ratio=%.3f, pca_components=%s, selected_features=%s",
        n_clusters,
        noise_ratio,
        actual_pca,
        len(selected_columns),
    )

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        noise_ratio=noise_ratio,
        pca_n_components=actual_pca,
        reduced_data=X_reduced,
        selected_columns=selected_columns,
    )

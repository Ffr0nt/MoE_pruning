from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from pruning.src.cluster_experts import ClusterResult


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_clustering_artifact(
    output_dir: str,
    result: ClusterResult,
    top_indices: np.ndarray,
    hook_layer: int,
    stage_config: dict[str, Any],
) -> None:
    """Сохраняет артефакт шага кластеризации в output_dir."""
    _ensure_dir(output_dir)

    np.save(os.path.join(output_dir, "labels.npy"), result.labels)
    np.save(os.path.join(output_dir, "selected_columns.npy"), result.selected_columns)
    np.save(os.path.join(output_dir, "top_indices.npy"), np.asarray(top_indices, dtype=np.int64))
    np.save(os.path.join(output_dir, "reduced_data.npy"), result.reduced_data)

    payload = {
        "created_at_utc": _utc_iso(),
        "hook_layer": hook_layer,
        "n_clusters": int(result.n_clusters),
        "noise_ratio": float(result.noise_ratio),
        "pca_n_components": int(result.pca_n_components),
        "num_selected_columns": int(len(result.selected_columns)),
        "num_top_indices": int(len(top_indices)),
        "config": stage_config,
    }

    with open(os.path.join(output_dir, "clustering_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_pruning_plan(
    output_dir: str,
    target_layer: int,
    hook_layer: int,
    strategy: str,
    experts_to_remove: list[int],
    diagnostics: dict[str, Any],
) -> str:
    """Сохраняет итоговый план удаления экспертов для одного слоя."""
    _ensure_dir(output_dir)
    deduped_sorted = sorted({int(idx) for idx in experts_to_remove})

    payload = {
        "created_at_utc": _utc_iso(),
        "status": "ok",
        "hook_layer": int(hook_layer),
        "target_layer": int(target_layer),
        "strategy": strategy,
        "criteria_summary": diagnostics,
        "experts_to_remove_by_layer": {
            str(int(target_layer)): deduped_sorted,
        },
    }

    path = os.path.join(output_dir, "pruning_plan.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

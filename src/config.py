from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pruning.src.cluster_experts import ClusterConfig


logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    """Общие runtime-настройки (из base.yaml)."""

    stage: str = "cluster"
    device: str | None = None
    hf_token: str | None = None
    hf_token_env_var: str = "HF_TOKEN"
    environment: dict[str, str] = field(default_factory=dict)


@dataclass
class PathsConfig:
    """Пути проекта (из .env и строятся на их основе)."""

    output_root: str = ""
    collection_dir: str = ""
    clustering_dir: str = ""
    pruning_dir: str = ""
    profile_dir: str = ""
    latent_indices_path: str = ""


@dataclass
class CollectionConfig:
    """Конфиг сбора SAE-активаций (из base.yaml + collect.yaml)."""

    mode: str = ""
    batch_size: int = 0
    max_batches: int = 0
    max_activations_per_expert: int = 0
    save_interval: int = 0
    dataset_name: str = ""
    dataset_split: str = ""
    max_chars_per_text: int | None = None
    model_id: str = ""
    hook_layers: list[int] = field(default_factory=list)
    sae_repo_id: str = ""


@dataclass
class PipelineConfig:
    """Конфиг pipeline (из base.yaml + stage-specific .yaml)."""

    num_experts: int | None = None
    top_k: int = 0
    collection_suffix: str = ""


@dataclass
class ExpertChoiceConfig:
    """Конфиг expert_choice-этапа (из expert_choice.yaml)."""

    target_layer: int | None = None
    strategy: str = "clustered_unclustered"
    clustered: "ClusteredExpertChoiceConfig" = field(
        default_factory=lambda: ClusteredExpertChoiceConfig()
    )
    unclustered: "UnclusteredExpertChoiceConfig" = field(
        default_factory=lambda: UnclusteredExpertChoiceConfig()
    )


@dataclass
class ClusteredExpertChoiceConfig:
    """Параметры прунинга экспертов с label >= 0."""

    use: bool = True
    min_per_cluster: int = 1
    keep_ratio: float = 0.2
    criterion: Literal["load", "variance"] = "load"


@dataclass
class UnclusteredExpertChoiceConfig:
    """Параметры prунинга экспертов с label == -1."""

    use: bool = True
    min_experts: int = 5
    keep_ratio: float = 0.2
    criterion: Literal["load", "variance", "distance"] = "load"


@dataclass
class ProfileConfig:
    """Конфиг этапа создания профиля датасета (из profile.yaml)."""

    input_json_path: str = ""
    input_json_paths: list[str] = field(default_factory=list)
    batch_size: int = 1
    max_batches: int = 0
    max_texts: int | None = None
    save_interval: int = 0
    max_chars_per_text: int | None = None
    file_suffix: str = ""


@dataclass
class ProjectConfig:
    """Общий конфиг всего пайплайна pruning."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    expert_choice: ExpertChoiceConfig = field(default_factory=ExpertChoiceConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


def normalize_paths(paths: PathsConfig) -> PathsConfig:
    """Нормализует пути (раскрывает ~ и переменные окружения)."""
    # Все пути должны быть явно установлены в .env
    # Здесь просто раскрываем пути и переменные окружения
    paths.output_root = os.path.expanduser(paths.output_root)
    paths.collection_dir = os.path.expanduser(paths.collection_dir)
    paths.clustering_dir = os.path.expanduser(paths.clustering_dir)
    paths.pruning_dir = os.path.expanduser(paths.pruning_dir)
    paths.profile_dir = os.path.expanduser(paths.profile_dir)
    paths.latent_indices_path = os.path.expanduser(paths.latent_indices_path)
    return paths


def _load_yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Для загрузки config.yaml требуется пакет PyYAML. "
            "Установите его: pip install pyyaml"
        ) from exc
    return yaml


def _merge_dataclass(instance: Any, updates: dict[str, Any] | None) -> Any:
    if not updates:
        return instance

    for key, value in updates.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def normalize_hook_layers(raw_hook_layers: Any) -> list[int]:
    """Нормализует и валидирует список слоёв для сбора."""
    if raw_hook_layers is None:
        raise ValueError("model.hook_layers must be set in base.yaml")

    if not isinstance(raw_hook_layers, list):
        raise TypeError("model.hook_layers must be a list of integers")

    if not raw_hook_layers:
        raise ValueError("model.hook_layers must contain at least one layer")

    normalized: list[int] = []
    for layer in raw_hook_layers:
        if not isinstance(layer, int):
            raise TypeError("Each value in model.hook_layers must be an integer")
        if layer < 0:
            raise ValueError("Layer indices in model.hook_layers must be non-negative")
        normalized.append(layer)

    unique_layers = list(dict.fromkeys(normalized))
    if len(unique_layers) != len(normalized):
        raise ValueError("model.hook_layers must not contain duplicate values")

    return unique_layers


def get_layer_output_dir(base_dir: str, layer: int) -> str:
    """Возвращает подпапку артефактов для конкретного слоя."""
    return os.path.join(os.path.expanduser(base_dir), f"layer_{layer}")


def get_layer_collection_dir(config: "ProjectConfig", layer: int) -> str:
    """Возвращает каталог collection для конкретного слоя."""
    return get_layer_output_dir(config.paths.collection_dir, layer)


def get_layer_clustering_dir(config: "ProjectConfig", layer: int) -> str:
    """Возвращает каталог clustering для конкретного слоя."""
    return get_layer_output_dir(config.paths.clustering_dir, layer)


def get_layer_pruning_dir(config: "ProjectConfig", layer: int) -> str:
    """Возвращает каталог pruning для конкретного слоя."""
    return get_layer_output_dir(config.paths.pruning_dir, layer)


def get_layer_profile_dir(config: "ProjectConfig", layer: int) -> str:
    """Возвращает каталог profile для конкретного слоя."""
    return get_layer_output_dir(config.paths.profile_dir, layer)


def _validate_keep_ratio(value: float, field_name: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {value}")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if int(value) < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value}")


def validate_expert_choice_config(expert_choice: ExpertChoiceConfig) -> None:
    """Валидирует конфиг выбора экспертов для удаления."""
    _validate_non_negative_int(
        expert_choice.clustered.min_per_cluster,
        "expert_choice.clustered.min_per_cluster",
    )
    _validate_keep_ratio(
        expert_choice.clustered.keep_ratio,
        "expert_choice.clustered.keep_ratio",
    )

    _validate_non_negative_int(
        expert_choice.unclustered.min_experts,
        "expert_choice.unclustered.min_experts",
    )
    _validate_keep_ratio(
        expert_choice.unclustered.keep_ratio,
        "expert_choice.unclustered.keep_ratio",
    )


def resolve_default_config_path() -> str:
    """Возвращает путь к папке config/ в корне pruning."""
    return str(Path(__file__).resolve().parent.parent / "config")


def resolve_default_env_path() -> str:
    """Возвращает путь к .env в корне pruning."""
    return str(Path(__file__).resolve().parent.parent / ".env")


def load_dotenv_file(env_path: str | None = None, override: bool = False) -> None:
    """Загружает переменные окружения из .env без внешних зависимостей."""
    if env_path is None:
        env_path = resolve_default_env_path()

    env_path = os.path.expanduser(env_path)
    if not os.path.exists(env_path):
        logger.debug(".env file not found: %s", env_path)
        return

    logger.info("Loading environment variables from: %s", env_path)

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            if (
                len(value) >= 2
                and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'"))
            ):
                value = value[1:-1]

            if override or key not in os.environ:
                os.environ[key] = value


def load_project_config(config_path: str | None = None, stage: str | None = None) -> ProjectConfig:
    """Загружает конфиг проекта из папки config/, разбив на base + stage-specific.
    
    Args:
        config_path: Путь к папке config/ (по умолчанию pruning/config/).
        stage: Явно заданный stage (collect/profile/cluster/expert_choice).
               Если None, берётся из base.yaml.
    """
    load_dotenv_file()
    yaml = _load_yaml_module()

    if config_path is None:
        config_path = resolve_default_config_path()

    config_path = os.path.expanduser(config_path)

    if not os.path.isdir(config_path):
        raise FileNotFoundError(f"Config directory not found: {config_path}")

    # Загружаем base.yaml
    base_path = os.path.join(config_path, "base.yaml")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"base.yaml not found in {config_path}")

    logger.info("Loading base config from: %s", base_path)
    with open(base_path, "r", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f) or {}

    runtime = _merge_dataclass(RuntimeConfig(), base_raw.get("runtime"))
    
    # Загружаем пути из окружения (все пути из .env, machine-specific)
    paths_from_env = {
        "output_root": os.getenv("PRUNING_OUTPUT_ROOT"),
        "collection_dir": os.getenv("PRUNING_COLLECTION_DIR"),
        "clustering_dir": os.getenv("PRUNING_CLUSTERING_DIR"),
        "pruning_dir": os.getenv("PRUNING_PRUNING_DIR"),
        "profile_dir": os.getenv("PRUNING_PROFILE_DIR"),
        "latent_indices_path": os.getenv("PRUNING_LATENT_INDICES_PATH"),
    }
    # Все пути должны быть установлены в .env
    missing_paths = [k for k, v in paths_from_env.items() if v is None]
    if missing_paths:
        logger.warning("Missing environment variables for paths: %s", missing_paths)
    paths_from_env = {k: v for k, v in paths_from_env.items() if v is not None}
    if "profile_dir" not in paths_from_env and "output_root" in paths_from_env:
        paths_from_env["profile_dir"] = os.path.join(paths_from_env["output_root"], "profile")
    paths = normalize_paths(_merge_dataclass(PathsConfig(), paths_from_env if paths_from_env else None))

    # Определяем stage из аргумента или из runtime
    resolved_stage = stage or runtime.stage
    logger.info("Using stage: %s", resolved_stage)

    # Загружаем stage-specific конфиг
    stage_file = f"{resolved_stage}.yaml"
    stage_path = os.path.join(config_path, stage_file)
    if not os.path.exists(stage_path):
        raise FileNotFoundError(f"Stage config not found: {stage_path}")

    logger.info("Loading stage config from: %s", stage_path)
    with open(stage_path, "r", encoding="utf-8") as f:
        stage_raw = yaml.safe_load(f) or {}

    # Сливаем параметры: base (model + pipeline) + stage-specific (collection/pipeline/cluster/expert_choice/profile)
    # Для collection: мержим base model-параметры + stage collection-параметры
    collection_base = base_raw.get("model", {})
    collection_stage = stage_raw.get("collection", {})
    collection_merged = {**collection_base, **collection_stage}
    collection = _merge_dataclass(CollectionConfig(), collection_merged if collection_merged else None)
    collection.hook_layers = normalize_hook_layers(collection.hook_layers)
    
    # Для pipeline: мержим base pipeline-параметры + stage pipeline-параметры
    pipeline_base = base_raw.get("pipeline", {})
    pipeline_stage = stage_raw.get("pipeline", {})
    pipeline_merged = {**pipeline_base, **pipeline_stage}
    pipeline = _merge_dataclass(PipelineConfig(), pipeline_merged if pipeline_merged else None)
    
    # expert_choice-конфиг поддерживает как секцию expert_choice, так и плоский формат.
    expert_choice_raw = stage_raw.get("expert_choice")
    if expert_choice_raw is None and (
        "clustered" in stage_raw or "unclustered" in stage_raw or "target_layer" in stage_raw
    ):
        expert_choice_raw = stage_raw
    expert_choice_raw = expert_choice_raw or {}

    expert_choice = ExpertChoiceConfig()
    expert_choice.target_layer = expert_choice_raw.get("target_layer", expert_choice.target_layer)
    expert_choice.strategy = expert_choice_raw.get("strategy", expert_choice.strategy)
    expert_choice.clustered = _merge_dataclass(
        ClusteredExpertChoiceConfig(),
        expert_choice_raw.get("clustered"),
    )
    expert_choice.unclustered = _merge_dataclass(
        UnclusteredExpertChoiceConfig(),
        expert_choice_raw.get("unclustered"),
    )
    validate_expert_choice_config(expert_choice)

    profile = _merge_dataclass(ProfileConfig(), stage_raw.get("profile"))
    cluster = _merge_dataclass(ClusterConfig(), stage_raw.get("cluster"))

    config = ProjectConfig(
        runtime=runtime,
        paths=paths,
        collection=collection,
        pipeline=pipeline,
        expert_choice=expert_choice,
        profile=profile,
        cluster=cluster,
    )
    logger.info(
        "Config loaded: stage=%s, output_root=%s, hook_layers=%s",
        config.runtime.stage,
        config.paths.output_root,
        config.collection.hook_layers,
    )
    return config


def resolve_device(runtime_config: RuntimeConfig) -> str:
    """Возвращает устройство вычисления (из env PRUNING_DEVICE или auto-detect)."""

    # Auto-detect
    try:
        import torch
    except ImportError:
        logger.warning("torch is not installed, falling back to CPU")
        return "cpu"

    resolved = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Auto-resolved device: %s", resolved)
    return resolved

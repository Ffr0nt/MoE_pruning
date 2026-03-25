#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Сбор SAE-активаций или статистик по экспертам через общий YAML-конфиг."""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import login
from sparsify import Sae
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pruning.src.config import (
    ProjectConfig,
    get_layer_collection_dir,
    resolve_device,
)


logger = logging.getLogger(__name__)


@dataclass
class CollectionRuntime:
    """Общие рантайм-объекты шага сбора."""

    device: str
    dataset: Any
    tokenizer: Any
    model: Any
    requested_latent_indices: list[int]


@dataclass
class LayerCollectionRuntime:
    """Рантайм-объекты шага сбора для конкретного слоя."""

    hook_layer: int
    sae: Any
    num_latents: int
    num_experts: int
    output_dir: str
    latent_indices: list[int]


@dataclass
class LayerCollectionState:
    """Состояние накопления статистик для конкретного слоя."""

    runtime: LayerCollectionRuntime
    storage: dict[str, Any]


def configure_environment(config: ProjectConfig) -> None:
    """Применяет runtime-переменные окружения из YAML."""
    for key, value in config.runtime.environment.items():
        os.environ[key] = value


def load_latent_indices(path: str) -> list[int]:
    """Загружает список latent indices из JSON."""
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        return list(json.load(f))


def login_huggingface_if_needed(config: ProjectConfig) -> None:
    """Логинится в Hugging Face, если токен задан в конфиге или env."""
    token = config.runtime.hf_token
    if token is None and config.runtime.hf_token_env_var:
        token = os.getenv(config.runtime.hf_token_env_var)

    if token:
        login(token=token)


def build_runtime(config: ProjectConfig) -> CollectionRuntime:
    """Загружает датасет, модель и общие параметры сбора."""
    device = resolve_device(config.runtime)

    logger.info("Using device: %s", device)
    logger.info("Collection root directory: %s", config.paths.collection_dir)
    logger.info("Collection mode: %s", config.collection.mode)
    logger.info("Hook layers: %s", config.collection.hook_layers)

    login_huggingface_if_needed(config)

    logger.info("Loading dataset %s...", config.collection.dataset_name)
    dataset = load_dataset(
        config.collection.dataset_name,
        split=config.collection.dataset_split,
    )
    logger.info("Dataset loaded: %s", dataset)

    model_config = AutoConfig.from_pretrained(config.collection.model_id)
    model_config.output_hidden_states = True

    tokenizer = AutoTokenizer.from_pretrained(config.collection.model_id)
    model = cast(Any, AutoModelForCausalLM.from_pretrained(
        config.collection.model_id,
        config=model_config,
    ))
    model = model.to(device)
    model.eval()

    requested_latent_indices = load_latent_indices(config.paths.latent_indices_path)

    return CollectionRuntime(
        device=device,
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        requested_latent_indices=requested_latent_indices,
    )


def resolve_latent_indices(
    requested_latent_indices: list[int],
    num_latents: int,
    hook_layer: int,
) -> list[int]:
    """Валидирует и нормализует список latent indices для слоя."""
    if not requested_latent_indices:
        return list(range(num_latents))

    out_of_range = [idx for idx in requested_latent_indices if idx < 0 or idx >= num_latents]
    if out_of_range:
        preview = out_of_range[:5]
        raise ValueError(
            "Некоторые latent indices выходят за границы SAE слоя "
            f"{hook_layer}: {preview}"
        )

    return requested_latent_indices


def build_layer_runtime(
    config: ProjectConfig,
    runtime: CollectionRuntime,
    hook_layer: int,
) -> LayerCollectionRuntime:
    """Загружает SAE и вычисляет размеры для конкретного слоя."""
    sae = cast(Any, Sae.load_from_hub(
        config.collection.sae_repo_id,
        hookpoint=f"layers.{hook_layer}",
    ))
    sae = sae.to(runtime.device)

    if sae.W_dec is None:
        raise ValueError(f"У загруженного SAE слоя {hook_layer} отсутствует матрица W_dec")

    num_latents = int(sae.W_dec.shape[0])
    num_experts = int(runtime.model.model.layers[hook_layer].mlp.gate.out_features)
    latent_indices = resolve_latent_indices(
        runtime.requested_latent_indices,
        num_latents,
        hook_layer,
    )
    output_dir = get_layer_collection_dir(config, hook_layer)

    logger.info(
        "Layer %s: num_latents=%s | num_experts=%s | selected_latents=%s",
        hook_layer,
        num_latents,
        num_experts,
        len(latent_indices),
    )

    return LayerCollectionRuntime(
        hook_layer=hook_layer,
        sae=sae,
        num_latents=num_latents,
        num_experts=num_experts,
        output_dir=output_dir,
        latent_indices=latent_indices,
    )


def initialize_storage(config: ProjectConfig, layer_runtime: LayerCollectionRuntime) -> dict[str, Any]:
    """Инициализирует структуры хранения результатов."""
    os.makedirs(layer_runtime.output_dir, exist_ok=True)

    if config.collection.mode == "activations":
        return {
            "activations_per_expert": {i: [] for i in range(layer_runtime.num_experts)},
            "count_per_expert": {i: 0 for i in range(layer_runtime.num_experts)},
        }

    if config.collection.mode == "statistics":
        num_selected_latents = len(layer_runtime.latent_indices)
        return {
            "sum_activations": {
                i: np.zeros(num_selected_latents, dtype=np.float64)
                for i in range(layer_runtime.num_experts)
            },
            "sum_squared_activations": {
                i: np.zeros(num_selected_latents, dtype=np.float64)
                for i in range(layer_runtime.num_experts)
            },
            "count_per_expert": {i: 0 for i in range(layer_runtime.num_experts)},
        }

    raise ValueError(
        f"Unknown collection mode: {config.collection.mode}. Use 'activations' or 'statistics'."
    )


def prepare_batch_texts(batch: dict[str, list[str]], max_chars_per_text: int | None) -> list[str]:
    """Подготавливает тексты батча."""
    texts = list(batch["text"])
    if max_chars_per_text is None:
        return texts
    return [text[:max_chars_per_text] for text in texts]


def encode_batch(
    texts: list[str],
    runtime: CollectionRuntime,
    layer_states: dict[int, LayerCollectionState],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Получает SAE-активации и expert routing для всех слоёв батча."""
    inputs = runtime.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(runtime.device)

    with torch.no_grad():
        outputs = runtime.model(**inputs)

    encoded_by_layer: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for hook_layer, layer_state in layer_states.items():
        hidden_states = outputs.hidden_states[hook_layer]
        batch_size, seq_len, hidden_dim = hidden_states.size()
        hidden_flat = hidden_states.view(batch_size * seq_len, hidden_dim)

        sae_latents = layer_state.runtime.sae.encode(hidden_flat)
        gate_logits = runtime.model.model.layers[hook_layer].mlp.gate(hidden_flat)
        expert_idx = torch.argmax(gate_logits, dim=-1).cpu().numpy()

        sae_activations = sae_latents.pre_acts.detach().cpu().numpy()
        sae_activations_filtered = sae_activations[:, layer_state.runtime.latent_indices]
        encoded_by_layer[hook_layer] = (sae_activations_filtered, expert_idx)

    return encoded_by_layer


def process_activations_for_layer(
    activations: np.ndarray,
    expert_idx: np.ndarray,
    storage: dict[str, Any],
    layer_runtime: LayerCollectionRuntime,
    config: ProjectConfig,
) -> None:
    """Сохраняет сами активации для конкретного слоя."""
    activations_dict = storage["activations_per_expert"]
    count_dict = storage["count_per_expert"]

    for idx in range(layer_runtime.num_experts):
        if count_dict[idx] >= config.collection.max_activations_per_expert:
            continue

        mask = expert_idx == idx
        if not mask.any():
            continue

        selected = activations[mask]
        remaining_space = config.collection.max_activations_per_expert - count_dict[idx]
        if remaining_space <= 0:
            continue

        to_add = selected[:remaining_space]
        activations_dict[idx].append(to_add)
        count_dict[idx] += len(to_add)


def process_statistics_for_layer(
    activations: np.ndarray,
    expert_idx: np.ndarray,
    storage: dict[str, Any],
    layer_runtime: LayerCollectionRuntime,
) -> None:
    """Накапливает sum / sum_squared по экспертам для конкретного слоя."""
    sum_dict = storage["sum_activations"]
    sum_squared_dict = storage["sum_squared_activations"]
    count_dict = storage["count_per_expert"]

    for idx in range(layer_runtime.num_experts):
        mask = expert_idx == idx
        if not mask.any():
            continue

        selected = activations[mask]
        sum_dict[idx] += selected.sum(axis=0)
        sum_squared_dict[idx] += (selected ** 2).sum(axis=0)
        count_dict[idx] += len(selected)


def save_activations(
    activations_dict: dict[int, list[np.ndarray]],
    output_dir: str,
    suffix: str = "",
) -> None:
    """Сохраняет собранные активации в отдельные файлы по экспертам."""
    for expert_id, activations_list in activations_dict.items():
        if len(activations_list) == 0:
            continue

        all_activations = np.vstack(activations_list)
        filename = f"expert_{expert_id}_activations{suffix}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, all_activations)
        logger.info(
            "Saved %s activations for expert %s to %s",
            all_activations.shape[0],
            expert_id,
            filename,
        )


def save_statistics(
    sum_dict: dict[int, np.ndarray],
    sum_squared_dict: dict[int, np.ndarray],
    count_dict: dict[int, int],
    output_dir: str,
    num_experts: int,
    suffix: str = "",
) -> None:
    """Сохраняет накопленные статистики по экспертам."""
    for expert_id in range(num_experts):
        if count_dict[expert_id] <= 0:
            continue

        sum_filename = f"expert_{expert_id}_sum{suffix}.npy"
        sum_sq_filename = f"expert_{expert_id}_sum_squared{suffix}.npy"
        mean_filename = f"expert_{expert_id}_mean{suffix}.npy"
        std_filename = f"expert_{expert_id}_std{suffix}.npy"

        np.save(os.path.join(output_dir, sum_filename), sum_dict[expert_id])
        np.save(os.path.join(output_dir, sum_sq_filename), sum_squared_dict[expert_id])

        mean = sum_dict[expert_id] / count_dict[expert_id]
        variance = (sum_squared_dict[expert_id] / count_dict[expert_id]) - (mean ** 2)
        std = np.sqrt(np.maximum(variance, 0.0))

        np.save(os.path.join(output_dir, mean_filename), mean)
        np.save(os.path.join(output_dir, std_filename), std)
        logger.info(
            "Saved statistics for expert %s: %s samples",
            expert_id,
            count_dict[expert_id],
        )


def save_collection_stats(
    config: ProjectConfig,
    runtime: LayerCollectionRuntime,
    count_per_expert: dict[int, int],
    source_hook_layers: list[int],
) -> None:
    """Сохраняет метаданные сбора."""
    stats = {
        "collection_mode": config.collection.mode,
        "hook_layer": runtime.hook_layer,
        "source_hook_layers": source_hook_layers,
        "num_experts": runtime.num_experts,
        "num_latents": runtime.num_latents,
        "latent_indices": runtime.latent_indices,
        "num_saved_latents": len(runtime.latent_indices),
        "counts_per_expert": count_per_expert,
        "total_samples": sum(count_per_expert.values()),
    }

    stats_file = os.path.join(runtime.output_dir, "collection_stats.npy")
    np.save(stats_file, np.array(stats, dtype=object))


def maybe_save_intermediate(
    step: int,
    layer_states: dict[int, LayerCollectionState],
    config: ProjectConfig,
) -> None:
    """Делает промежуточное сохранение."""
    if step % config.collection.save_interval != 0 or step == 0:
        return

    logger.info("Saving intermediate results at batch %s", step)
    for hook_layer, layer_state in layer_states.items():
        count_per_expert = layer_state.storage["count_per_expert"]
        logger.info(
            "Layer %s current counts per expert: %s",
            hook_layer,
            count_per_expert,
        )

        if config.collection.mode == "activations":
            save_activations(
                layer_state.storage["activations_per_expert"],
                layer_state.runtime.output_dir,
                suffix=f"_batch_{step}",
            )
            continue

        save_statistics(
            layer_state.storage["sum_activations"],
            layer_state.storage["sum_squared_activations"],
            count_per_expert,
            layer_state.runtime.output_dir,
            layer_state.runtime.num_experts,
            suffix=f"_batch_{step}",
        )


def reached_activation_limit(
    layer_states: dict[int, LayerCollectionState],
    config: ProjectConfig,
) -> bool:
    """Проверяет условие остановки для режима activations."""
    if config.collection.mode != "activations":
        return False

    for layer_state in layer_states.values():
        min_count = min(layer_state.storage["count_per_expert"].values())
        if min_count < config.collection.max_activations_per_expert:
            return False

    return True


def finalize_collection(
    layer_states: dict[int, LayerCollectionState],
    config: ProjectConfig,
) -> None:
    """Сохраняет финальные результаты сбора."""
    logger.info("=== Saving final results ===")

    for hook_layer, layer_state in layer_states.items():
        if config.collection.mode == "activations":
            save_activations(
                layer_state.storage["activations_per_expert"],
                layer_state.runtime.output_dir,
            )
        else:
            save_statistics(
                layer_state.storage["sum_activations"],
                layer_state.storage["sum_squared_activations"],
                layer_state.storage["count_per_expert"],
                layer_state.runtime.output_dir,
                layer_state.runtime.num_experts,
            )

        save_collection_stats(
            config,
            layer_state.runtime,
            layer_state.storage["count_per_expert"],
            source_hook_layers=config.collection.hook_layers,
        )

        logger.info(
            "Layer %s done. SAE %s saved to '%s'",
            hook_layer,
            config.collection.mode,
            layer_state.runtime.output_dir,
        )
        logger.info(
            "Layer %s final counts per expert: %s",
            hook_layer,
            layer_state.storage["count_per_expert"],
        )
        logger.info(
            "Layer %s total samples processed: %s",
            hook_layer,
            sum(layer_state.storage["count_per_expert"].values()),
        )


def initialize_layer_states(
    config: ProjectConfig,
    runtime: CollectionRuntime,
) -> dict[int, LayerCollectionState]:
    """Создаёт runtime и storage для всех требуемых слоёв."""
    layer_states: dict[int, LayerCollectionState] = {}
    for hook_layer in config.collection.hook_layers:
        layer_runtime = build_layer_runtime(config, runtime, hook_layer)
        layer_states[hook_layer] = LayerCollectionState(
            runtime=layer_runtime,
            storage=initialize_storage(config, layer_runtime),
        )

    return layer_states


def collect_statistics(config: ProjectConfig) -> None:
    """Запускает сбор SAE-активаций / статистик."""
    configure_environment(config)
    runtime = build_runtime(config)
    layer_states = initialize_layer_states(config, runtime)

    logger.info("=== Collecting SAE %s ===", config.collection.mode)
    for i in tqdm(range(0, config.collection.max_batches, config.collection.batch_size), desc="Batches"):
        batch = runtime.dataset[i : i + config.collection.batch_size]
        texts_batch = prepare_batch_texts(batch, config.collection.max_chars_per_text)

        try:
            encoded_by_layer = encode_batch(texts_batch, runtime, layer_states)
            for hook_layer, layer_state in layer_states.items():
                activations, expert_idx = encoded_by_layer[hook_layer]
                if config.collection.mode == "activations":
                    process_activations_for_layer(
                        activations,
                        expert_idx,
                        layer_state.storage,
                        layer_state.runtime,
                        config,
                    )
                else:
                    process_statistics_for_layer(
                        activations,
                        expert_idx,
                        layer_state.storage,
                        layer_state.runtime,
                    )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("OOM on batch %s, skipping...", i)
                torch.cuda.empty_cache()
                gc.collect()
                continue
            raise

        del batch, texts_batch
        torch.cuda.empty_cache()
        gc.collect()

        if reached_activation_limit(layer_states, config):
            logger.info(
                "Reached %s activations for all experts across all layers",
                config.collection.max_activations_per_expert,
            )
            break

        maybe_save_intermediate(i, layer_states, config)

    finalize_collection(layer_states, config)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Этап создания профиля датасета: расчёт усреднённого SAE-профиля по слоям."""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Generator, cast

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pruning.src.collect_expert_statistics import (
    CollectionRuntime,
    LayerCollectionRuntime,
    build_layer_runtime,
    configure_environment,
    load_latent_indices,
    login_huggingface_if_needed,
)
from pruning.src.config import ProjectConfig, get_layer_profile_dir, resolve_device


logger = logging.getLogger(__name__)


@dataclass
class LayerProfileState:
    """Состояние накопления профиля датасета для конкретного слоя."""

    runtime: LayerCollectionRuntime
    sum_activations: np.ndarray
    sum_squared_activations: np.ndarray
    token_count: int = 0


def build_profile_runtime(config: ProjectConfig) -> CollectionRuntime:
    """Загружает только то, что нужно для этапа создания профиля датасета."""
    device = resolve_device(config.runtime)
    logger.info("Using device: %s", device)
    logger.info("Dataset profile root directory: %s", config.paths.profile_dir)
    logger.info("Hook layers: %s", config.collection.hook_layers)

    login_huggingface_if_needed(config)

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
        dataset=None,
        tokenizer=tokenizer,
        model=model,
        requested_latent_indices=requested_latent_indices,
    )


def load_input_texts(json_path: str) -> list[str]:
    """Загружает и валидирует входные тексты из JSON-файла."""
    resolved_path = os.path.expanduser(json_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Input JSON file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError("Input JSON must contain a list of strings")

    cleaned: list[str] = []
    for idx, item in enumerate(data):
        if not isinstance(item, str):
            raise TypeError(
                f"Input JSON item at index {idx} is not a string: {type(item).__name__}"
            )
        text = item.strip()
        if text:
            cleaned.append(text)

    if not cleaned:
        raise ValueError("Input JSON contains no non-empty text items")

    return cleaned


def iter_text_batches(
    texts: list[str],
    batch_size: int,
    max_batches: int,
    max_texts: int | None,
) -> Generator[list[str], None, None]:
    """Итерирует тексты батчами с ограничением по max_texts/max_batches."""
    if batch_size <= 0:
        raise ValueError("profile.batch_size must be > 0")
    if max_batches < 0:
        raise ValueError("profile.max_batches must be >= 0")
    if max_texts is not None and max_texts <= 0:
        raise ValueError("profile.max_texts must be > 0 when set")

    limited = texts if max_texts is None else texts[:max_texts]
    batch_counter = 0
    for start in range(0, len(limited), batch_size):
        if max_batches > 0 and batch_counter >= max_batches:
            break
        yield limited[start : start + batch_size]
        batch_counter += 1


def resolve_input_dataset_files(config: ProjectConfig) -> list[str]:
    """Возвращает список JSON-файлов датасетов для этапа создания профиля."""
    paths: list[str] = []
    if config.profile.input_json_paths:
        paths.extend(config.profile.input_json_paths)
    if config.profile.input_json_path:
        paths.append(config.profile.input_json_path)

    if not paths:
        raise ValueError(
            "Для этапа создания профиля датасета задайте profile.input_json_paths "
            "или profile.input_json_path"
        )

    return [os.path.expanduser(path) for path in paths]


def to_dataset_tag(path: str) -> str:
    """Формирует безопасный dataset-тег из имени JSON-файла."""
    base_name = os.path.splitext(os.path.basename(path))[0].strip().lower()
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", base_name).strip("_")
    return normalized or "dataset"


def build_dataset_inputs(config: ProjectConfig) -> list[tuple[str, str, list[str]]]:
    """Собирает входы вида (dataset_path, dataset_tag, texts)."""
    dataset_paths = resolve_input_dataset_files(config)

    inputs: list[tuple[str, str, list[str]]] = []
    used_tags: dict[str, int] = {}
    for path in dataset_paths:
        texts = load_input_texts(path)
        base_tag = to_dataset_tag(path)
        counter = used_tags.get(base_tag, 0)
        used_tags[base_tag] = counter + 1
        dataset_tag = base_tag if counter == 0 else f"{base_tag}_{counter + 1}"
        inputs.append((path, dataset_tag, texts))

    return inputs


def initialize_layer_profile_states(
    config: ProjectConfig,
    runtime: CollectionRuntime,
) -> dict[int, LayerProfileState]:
    """Создаёт storage для profile по всем слоям из hook_layers."""
    states: dict[int, LayerProfileState] = {}
    for hook_layer in config.collection.hook_layers:
        layer_runtime = build_layer_runtime(config, runtime, hook_layer)
        layer_runtime.output_dir = get_layer_profile_dir(config, hook_layer)
        os.makedirs(layer_runtime.output_dir, exist_ok=True)

        num_selected_latents = len(layer_runtime.latent_indices)
        states[hook_layer] = LayerProfileState(
            runtime=layer_runtime,
            sum_activations=np.zeros(num_selected_latents, dtype=np.float64),
            sum_squared_activations=np.zeros(num_selected_latents, dtype=np.float64),
            token_count=0,
        )
    return states


def prepare_batch_texts(texts_batch: list[str], max_chars_per_text: int | None) -> list[str]:
    """Ограничивает длину текстов батча при необходимости."""
    if max_chars_per_text is None:
        return texts_batch
    return [text[:max_chars_per_text] for text in texts_batch]


def encode_profile_batch(
    texts: list[str],
    runtime: CollectionRuntime,
    layer_states: dict[int, LayerProfileState],
) -> dict[int, np.ndarray]:
    """Получает SAE pre-acts для всех токенов и всех слоёв батча."""
    inputs = runtime.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(runtime.device)

    with torch.no_grad():
        outputs = runtime.model(**inputs)

    encoded_by_layer: dict[int, np.ndarray] = {}
    for hook_layer, state in layer_states.items():
        hidden_states = outputs.hidden_states[hook_layer]
        batch_size, seq_len, hidden_dim = hidden_states.size()
        hidden_flat = hidden_states.view(batch_size * seq_len, hidden_dim)

        sae_latents = state.runtime.sae.encode(hidden_flat)
        sae_activations = sae_latents.pre_acts.detach().cpu().numpy()
        sae_activations_filtered = sae_activations[:, state.runtime.latent_indices]
        encoded_by_layer[hook_layer] = sae_activations_filtered

    return encoded_by_layer


def aggregate_layer_profile(activations: np.ndarray, state: LayerProfileState) -> None:
    """Накапливает суммарные статистики profile для одного слоя."""
    state.sum_activations += activations.sum(axis=0)
    state.sum_squared_activations += (activations ** 2).sum(axis=0)
    state.token_count += int(activations.shape[0])


def save_layer_profile(
    config: ProjectConfig,
    state: LayerProfileState,
    dataset_path: str,
    dataset_tag: str,
    suffix: str = "",
) -> None:
    """Сохраняет профиль датасета одного слоя в отдельные .npy файлы."""
    if state.token_count <= 0:
        logger.warning("Layer %s has no tokens to save", state.runtime.hook_layer)
        return

    output_dir = state.runtime.output_dir
    sum_filename = f"dataset_profile_{dataset_tag}_sum{suffix}.npy"
    sum_sq_filename = f"dataset_profile_{dataset_tag}_sum_squared{suffix}.npy"
    mean_filename = f"dataset_profile_{dataset_tag}_mean{suffix}.npy"
    std_filename = f"dataset_profile_{dataset_tag}_std{suffix}.npy"
    count_filename = f"dataset_profile_{dataset_tag}_count{suffix}.npy"

    mean = state.sum_activations / state.token_count
    variance = (state.sum_squared_activations / state.token_count) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 0.0))

    np.save(os.path.join(output_dir, sum_filename), state.sum_activations)
    np.save(os.path.join(output_dir, sum_sq_filename), state.sum_squared_activations)
    np.save(os.path.join(output_dir, mean_filename), mean)
    np.save(os.path.join(output_dir, std_filename), std)
    np.save(os.path.join(output_dir, count_filename), np.array(state.token_count, dtype=np.int64))

    stats = {
        "hook_layer": state.runtime.hook_layer,
        "source_hook_layers": config.collection.hook_layers,
        "num_latents": state.runtime.num_latents,
        "latent_indices": state.runtime.latent_indices,
        "num_saved_latents": len(state.runtime.latent_indices),
        "token_count": state.token_count,
        "dataset_path": dataset_path,
        "dataset_tag": dataset_tag,
        "file_suffix": suffix,
    }
    np.save(
        os.path.join(output_dir, f"dataset_profile_{dataset_tag}_stats{suffix}.npy"),
        np.array(stats, dtype=object),
    )


def maybe_save_intermediate(
    step: int,
    config: ProjectConfig,
    layer_states: dict[int, LayerProfileState],
    dataset_path: str,
    dataset_tag: str,
) -> None:
    """Промежуточно сохраняет профиль датасета по слоям."""
    if config.profile.save_interval <= 0:
        return
    if step <= 0 or step % config.profile.save_interval != 0:
        return

    logger.info(
        "[создание профиля датасета] Промежуточное сохранение на batch %s для %s",
        step,
        dataset_tag,
    )
    for state in layer_states.values():
        save_layer_profile(
            config,
            state,
            dataset_path=dataset_path,
            dataset_tag=dataset_tag,
            suffix=f"_batch_{step}",
        )


def get_dataset_profile(config: ProjectConfig) -> None:
    """Считает и сохраняет усреднённый SAE-профиль для каждого входного датасета."""
    configure_environment(config)
    runtime = build_profile_runtime(config)

    datasets = build_dataset_inputs(config)

    logger.info(
        "[создание профиля датасета] Старт: datasets=%s, layers=%s, batch_size=%s",
        len(datasets),
        config.collection.hook_layers,
        config.profile.batch_size,
    )

    for dataset_path, dataset_tag, texts in datasets:
        logger.info(
            "[создание профиля датасета] Обработка dataset=%s path=%s texts=%s",
            dataset_tag,
            dataset_path,
            len(texts),
        )
        layer_states = initialize_layer_profile_states(config, runtime)

        for step, texts_batch in enumerate(
            tqdm(
                iter_text_batches(
                    texts,
                    batch_size=config.profile.batch_size,
                    max_batches=config.profile.max_batches,
                    max_texts=config.profile.max_texts,
                ),
                desc=f"Dataset profile batches ({dataset_tag})",
            ),
            start=1,
        ):
            prepared_batch = prepare_batch_texts(texts_batch, config.profile.max_chars_per_text)

            try:
                encoded_by_layer = encode_profile_batch(prepared_batch, runtime, layer_states)
                for hook_layer, activations in encoded_by_layer.items():
                    aggregate_layer_profile(activations, layer_states[hook_layer])
            except RuntimeError as exc:
                if "CUDA out of memory" in str(exc):
                    logger.warning(
                        "[создание профиля датасета] OOM на batch %s для %s, skip",
                        step,
                        dataset_tag,
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise

            del prepared_batch
            torch.cuda.empty_cache()
            gc.collect()

            maybe_save_intermediate(
                step,
                config,
                layer_states,
                dataset_path=dataset_path,
                dataset_tag=dataset_tag,
            )

        logger.info(
            "[создание профиля датасета] Финальное сохранение для dataset=%s",
            dataset_tag,
        )
        for state in layer_states.values():
            save_layer_profile(
                config,
                state,
                dataset_path=dataset_path,
                dataset_tag=dataset_tag,
                suffix=config.profile.file_suffix,
            )

    logger.info("[создание профиля датасета] Готово")

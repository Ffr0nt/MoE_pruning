# Pruning Pipeline (MoE + SAE)

Этот документ описывает текущий пайплайн в папке pruning, его этапы, форматы данных между этапами и структуру артефактов на диске.

## 1. Карта модулей в src

- src/config.py - загрузка и валидация конфигов (base.yaml + stage-specific), нормализация путей, резолв директорий layer_{hook_layer}.
- src/workflow_steps.py - оркестратор этапов (collect, profile, cluster, expert_choice) и проверки обязательных upstream-артефактов.
- src/collect_expert_statistics.py - сбор SAE-активаций по токенам и агрегация per-expert статистик.
- src/dataset_profile.py - dataset-level профиль SAE по входным JSON-файлам.
- src/expert_statistics_loader.py - загрузка per-expert .npy и объединение в матрицы num_experts x num_latents.
- src/feature_selection.py - вычисление NMD и выбор top-k признаков.
- src/cluster_experts.py - PCA + HDBSCAN кластеризация экспертов.
- src/expert_pruning.py - логика выбора экспертов для удаления.
- src/pipeline_artifact_store.py - сериализация артефактов кластеризации и плана удаления экспертов.

## 2. Этапы пайплайна

Порядок выполнения через CLI:

1. collect
2. profile
3. cluster
4. expert_choice

Команды запуска:

```bash
python -m pruning.main --stage collect
python -m pruning.main --stage profile
python -m pruning.main --stage cluster
python -m pruning.main --stage expert_choice
```

### Stage collect

Что делает:
- Прогоняет тексты через MoE-модель и SAE.
- Для каждого эксперта собирает статистики по латентам (или сырые активации в mode=activations).

Выходные артефакты (${PRUNING_COLLECTION_DIR}/layer_{hook_layer}/):
- expert_{id}_sum{suffix}.npy
- expert_{id}_sum_squared{suffix}.npy
- expert_{id}_mean{suffix}.npy
- expert_{id}_std{suffix}.npy
- collection_stats.npy

### Stage profile

Что делает:
- Строит dataset-level SAE-профиль, независимо от кластеризации.

Выходные артефакты (${PRUNING_PROFILE_DIR}/layer_{hook_layer}/):
- dataset_profile_{dataset_tag}_sum{suffix}.npy
- dataset_profile_{dataset_tag}_sum_squared{suffix}.npy
- dataset_profile_{dataset_tag}_mean{suffix}.npy
- dataset_profile_{dataset_tag}_std{suffix}.npy
- dataset_profile_{dataset_tag}_count{suffix}.npy
- dataset_profile_{dataset_tag}_stats{suffix}.npy

### Stage cluster

Что делает:
- Загружает per-expert статистики из collect.
- Считает NMD и выбирает top-k признаки.
- Кластеризует экспертов (PCA -> HDBSCAN).

Выходные артефакты (${PRUNING_CLUSTERING_DIR}/layer_{hook_layer}/):
- labels.npy
- selected_columns.npy
- top_indices.npy
- reduced_data.npy
- clustering_summary.json

### Stage expert_choice

Что делает:
- Берет метки кластеров и статистики экспертов.
- По критериям (load, variance, distance) и keep_ratio выбирает экспертов для удаления.

Выходные артефакты (${PRUNING_PRUNING_DIR}/layer_{hook_layer}/):
- pruning_plan.json со status: ok и experts_to_remove_by_layer.

## 3. Формат данных между этапами

| Producer stage | Consumer stage | Что передается | Формат на диске | Формат после загрузки |
|---|---|---|---|---|
| collect | cluster | per-expert статистики | .npy по экспертам + collection_stats.npy | ExpertStatistics |
| collect | expert_choice | per-expert статистики | .npy | ExpertStatistics |
| cluster | expert_choice | метки кластеров и PCA-представление | labels.npy, reduced_data.npy | np.ndarray |
| profile | аналитика/сравнение | dataset-level профиль | .npy | np.ndarray/np.int64/dict |

## 4. Минимальные проверки после запуска

После collect:
- Есть collection_stats.npy и per-expert mean/sum_squared/sum/std.

После profile:
- На каждый dataset_tag сохранены sum/sum_squared/mean/std/count/stats.

После cluster:
- Есть labels.npy, reduced_data.npy, selected_columns.npy, top_indices.npy, clustering_summary.json.

После expert_choice:
- pruning_plan.json имеет status=ok и валидный criteria_summary.

## 5. Важные замечания

- Папки артефактов задаются через .env:
  - PRUNING_COLLECTION_DIR
  - PRUNING_PROFILE_DIR
  - PRUNING_CLUSTERING_DIR
  - PRUNING_PRUNING_DIR
  - PRUNING_LATENT_INDICES_PATH
- Stage profile концептуально независим от collect/cluster/expert_choice, но использует те же SAE/модельные настройки.

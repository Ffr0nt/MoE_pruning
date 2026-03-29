# Pruning Pipeline (MoE + SAE)

Этот документ описывает пайплайн в папке `pruning`, его этапы, форматы данных между этапами и структуру артефактов на диске.

## 1. Карта модулей в `src`

- `src/config.py` — загрузка и валидация конфигов (`base.yaml` + stage-specific), нормализация путей, резолв директорий `layer_{hook_layer}`.
- `src/workflow_steps.py` — оркестратор этапов (`collect`, `profile`, `cluster`, `expert_choice`, `prune`) и проверки обязательных upstream-артефактов.
- `src/collect_expert_statistics.py` — сбор SAE-активаций по токенам, агрегация per-expert статистик (`sum`, `sum_squared`, `mean`, `std`, `count`).
- `src/dataset_profile.py` — независимый от роутинга экспертов dataset-level профиль SAE по входным JSON-файлам с текстами.
- `src/expert_statistics_loader.py` — загрузка per-expert `.npy` с диска и объединение в матрицы `num_experts x num_latents`.
- `src/feature_selection.py` — вычисление NMD (Cohen's d) и выбор специфичных top-k признаков.
- `src/cluster_experts.py` — PCA + HDBSCAN кластеризация экспертов по выбранным SAE-признакам.
- `src/expert_pruning.py` — построение решения, каких экспертов удалять (`clustered/unclustered`, критерии ранжирования).
- `src/pipeline_artifact_store.py` — сериализация артефактов кластеризации и итогового pruning-плана.
- `src/__init__.py` — package marker.

## 2. Этапы пайплайна

Порядок выполнения через CLI:

1. `collect`
2. `profile`
3. `cluster`
4. `expert_choice`
5. `prune`

Команды запуска:

```bash
python -m pruning.main --stage collect
python -m pruning.main --stage profile
python -m pruning.main --stage cluster
python -m pruning.main --stage expert_choice
python -m pruning.main --stage prune
```

### Stage `collect`

Что делает концептуально:
- Прогоняет тексты через MoE-модель и SAE.
- Для каждого эксперта собирает статистики по латентам (в режиме `statistics`) или сырые активации (в режиме `activations`).

Основные входы:
- `config/base.yaml` + `config/collect.yaml`.
- `model_id`, `sae_repo_id`, `dataset_name`, `dataset_split`, `hook_layers`.
- `latent_indices_path` (через переменную окружения `PRUNING_LATENT_INDICES_PATH`).

Что загружается после шага (формат в памяти):
- `load_expert_statistics(...)` из `src/expert_statistics_loader.py` читает:
  - `mean_activations: np.ndarray[float64]` формы `(num_experts, num_latents)`.
  - `sum_squared_activations: np.ndarray[float64]` формы `(num_experts, num_latents)`.
  - `count_per_expert: np.ndarray[float64]` формы `(num_experts,)`.

Выходные артефакты на диске (`${PRUNING_COLLECTION_DIR}/layer_{hook_layer}/`):
- `expert_{id}_sum{suffix}.npy` — `np.ndarray`, форма `(num_latents,)`.
- `expert_{id}_sum_squared{suffix}.npy` — `np.ndarray`, форма `(num_latents,)`.
- `expert_{id}_mean{suffix}.npy` — `np.ndarray`, форма `(num_latents,)`.
- `expert_{id}_std{suffix}.npy` — `np.ndarray`, форма `(num_latents,)`.
- `collection_stats.npy` — `np.ndarray(dtype=object)` с dict-метаданными:
  - `collection_mode`, `hook_layer`, `source_hook_layers`, `num_experts`, `num_latents`,
  - `latent_indices`, `num_saved_latents`, `counts_per_expert`, `total_samples`.

Примечание:
- При `mode=activations` также сохраняется `expert_{id}_activations{suffix}.npy` формы `(num_tokens_for_expert, num_latents)`.

### Stage `profile`

Что делает концептуально:
- Строит dataset-level SAE-профиль (среднее/дисперсия по токенам), независимый от кластеризации и выбора экспертов.

Основные входы:
- `config/base.yaml` + `config/profile.yaml`.
- Один или несколько JSON-файлов (`input_json_path` / `input_json_paths`) с массивом строк-текстов.

Что загружается после шага (формат в памяти):
- Обычно как `np.ndarray` через `np.load(...)`:
  - dataset-level `sum`, `sum_squared`, `mean`, `std` формы `(num_latents,)`.
  - `count` как `np.int64` (скаляр).

Выходные артефакты (`${PRUNING_PROFILE_DIR}/layer_{hook_layer}/`):
- `dataset_profile_{dataset_tag}_sum{suffix}.npy`
- `dataset_profile_{dataset_tag}_sum_squared{suffix}.npy`
- `dataset_profile_{dataset_tag}_mean{suffix}.npy`
- `dataset_profile_{dataset_tag}_std{suffix}.npy`
- `dataset_profile_{dataset_tag}_count{suffix}.npy` (`np.int64`)
- `dataset_profile_{dataset_tag}_stats{suffix}.npy` — `np.ndarray(dtype=object)` с dict-метаданными.

### Stage `cluster`

Что делает концептуально:
- Загружает per-expert статистики из `collect`.
- Считает NMD (Cohen's d) по латентам.
- Выбирает top-k признаки на эксперта, объединяет индексы.
- Кластеризует экспертов: PCA -> HDBSCAN.

Основные входы:
- `config/base.yaml` + `config/cluster.yaml`.
- Артефакты `collect` в `collection_dir/layer_{hook_layer}`.

Что загружается после шага (формат в памяти):
- `labels: np.ndarray[int]` формы `(num_experts,)`.
- `reduced_data: np.ndarray[float64]` формы `(num_experts, pca_n_components)`.
- `selected_columns: np.ndarray[int64]` формы `(n_selected_features,)`.
- `top_indices: np.ndarray[int64]` формы `(n_top_indices,)`.
- summary JSON как `dict`.

Выходные артефакты (`${PRUNING_CLUSTERING_DIR}/layer_{hook_layer}/`):
- `labels.npy`
- `selected_columns.npy`
- `top_indices.npy`
- `reduced_data.npy`
- `clustering_summary.json` со служебными полями:
  - `created_at_utc`, `hook_layer`, `n_clusters`, `noise_ratio`, `pca_n_components`,
  - `num_selected_columns`, `num_top_indices`, `config`.

### Stage `expert_choice`

Что делает концептуально:
- Берет метки кластеров + статистики экспертов.
- Отдельно обрабатывает экспертов внутри кластеров (`label >= 0`) и шум (`label == -1`).
- По критерию (`load`, `variance`, `distance`) и `keep_ratio` решает, кого оставить и кого удалить.

Основные входы:
- `config/base.yaml` + `config/expert_choice.yaml`.
- `labels.npy`, `reduced_data.npy` из `cluster`.
- Статистики `collect` через `load_expert_statistics(...)`.

Что загружается после шага (формат в памяти):
- pruning plan JSON как dict:
  - `experts_to_remove_by_layer: dict[str, list[int]]`.
  - `criteria_summary: dict` с clustered/unclustered/totals.

Выходные артефакты (`${PRUNING_PRUNING_DIR}/layer_{hook_layer}/`):
- `pruning_plan.json`:

```json
{
  "status": "ok",
  "hook_layer": 15,
  "target_layer": 15,
  "strategy": "clustered_unclustered",
  "criteria_summary": {
    "clustered": {"enabled": true, "criterion": "load", "clusters": {}},
    "unclustered": {"enabled": true, "criterion": "load", "total": 0, "k": 0, "kept": 0, "removed": 0},
    "totals": {"num_experts": 60, "kept": 18, "removed": 42, "removed_ratio": 0.7}
  },
  "experts_to_remove_by_layer": {
    "15": [1, 3, 5]
  }
}
```

### Stage `prune`

Что делает концептуально:
- Пока не удаляет веса модели.
- Сохраняет placeholder-план для последующей реализации шага физического прунинга.

Основные входы:
- `config/base.yaml` + `config/prune.yaml`.
- Проверенные артефакты `cluster`.

Что загружается после шага (формат в памяти):
- placeholder JSON как dict.

Выходные артефакты (`${PRUNING_PRUNING_DIR}/layer_{hook_layer}/`):
- `pruning_plan.json` со статусом:

```json
{
  "status": "not_implemented",
  "message": "Pruning step is not implemented yet.",
  "experts_to_remove_by_layer": {}
}
```

## 3. Формат данных между этапами

| Producer stage | Consumer stage | Что передается | Формат на диске | Формат после загрузки |
|---|---|---|---|---|
| `collect` | `cluster` | per-expert статистики | `.npy` по экспертам + `collection_stats.npy` | `ExpertStatistics` (3 массива NumPy + мета размеры) |
| `collect` | `expert_choice` | те же per-expert статистики | `.npy` | `ExpertStatistics` |
| `cluster` | `expert_choice` | метки кластеров и PCA-представление | `labels.npy`, `reduced_data.npy` | `np.ndarray` |
| `expert_choice` | `prune` (будущий) | итоговое решение удаления | `pruning_plan.json` | `dict` |
| `profile` | аналитика/сравнение | dataset-level профиль | `.npy` | `np.ndarray`/`np.int64`/`dict` |

## 4. Структура артефактов по этапам

### 4.1 `collect` (statistics mode)

```text
${PRUNING_COLLECTION_DIR}/
  layer_{hook_layer}/
    expert_0_sum.npy
    expert_0_sum_squared.npy
    expert_0_mean.npy
    expert_0_std.npy
    ...
    expert_59_sum.npy
    expert_59_sum_squared.npy
    expert_59_mean.npy
    expert_59_std.npy
    collection_stats.npy
```

Как устроено:
- Один набор файлов на каждого эксперта (`id` от `0` до `num_experts-1`).
- Все векторы одного эксперта имеют одинаковую длину: `num_saved_latents`.
- `collection_stats.npy` хранит метаданные и `counts_per_expert`, которые используются при реконструкции `count_per_expert` в loader-е.

### 4.2 `profile`

```text
${PRUNING_PROFILE_DIR}/
  layer_{hook_layer}/
    dataset_profile_{dataset_tag}_sum.npy
    dataset_profile_{dataset_tag}_sum_squared.npy
    dataset_profile_{dataset_tag}_mean.npy
    dataset_profile_{dataset_tag}_std.npy
    dataset_profile_{dataset_tag}_count.npy
    dataset_profile_{dataset_tag}_stats.npy
```

Как устроено:
- Отдельный набор файлов на каждый входной JSON датасет (`dataset_tag`).
- Файлы совместимы с обычным `np.load`, stats-файл — объектный dict.

### 4.3 `cluster`

```text
${PRUNING_CLUSTERING_DIR}/
  layer_{hook_layer}/
    labels.npy
    selected_columns.npy
    top_indices.npy
    reduced_data.npy
    clustering_summary.json
```

Как устроено:
- `labels.npy` индексируется по expert_id.
- `selected_columns.npy`/`top_indices.npy` описывают, какие SAE-латенты использованы для кластеризации.
- `reduced_data.npy` — то же множество экспертов в PCA-пространстве.
- `clustering_summary.json` фиксирует итог и параметры запуска.

### 4.4 `expert_choice` / `prune`

```text
${PRUNING_PRUNING_DIR}/
  layer_{hook_layer}/
    pruning_plan.json
```

Как устроено:
- Единый JSON-контракт для downstream-prune.
- В рабочем режиме (`expert_choice`) содержит `status=ok`, список удаляемых экспертов по целевому слою и диагностику критериев.
- В заглушке (`prune`) содержит `status=not_implemented` и пустой `experts_to_remove_by_layer`.

## 5. Минимальные проверки после запуска

После `collect`:
- Есть `collection_stats.npy` и per-expert `mean/sum_squared/sum/std`.
- Количество экспертов на диске совпадает с `pipeline.num_experts`.

После `profile`:
- На каждый `dataset_tag` сохранены `sum/sum_squared/mean/std/count/stats`.

После `cluster`:
- Есть пять обязательных файлов: `labels.npy`, `reduced_data.npy`, `selected_columns.npy`, `top_indices.npy`, `clustering_summary.json`.

После `expert_choice`:
- `pruning_plan.json` имеет `status=ok` и непустой/валидный `criteria_summary`.

После `prune`:
- `pruning_plan.json` существует, но имеет `status=not_implemented` (текущая ожидаемая логика).

## 6. Важные замечания

- Папки артефактов задаются через `.env`:
  - `PRUNING_COLLECTION_DIR`
  - `PRUNING_PROFILE_DIR`
  - `PRUNING_CLUSTERING_DIR`
  - `PRUNING_PRUNING_DIR`
  - `PRUNING_LATENT_INDICES_PATH`
- Stage `profile` концептуально независим от `collect/cluster/expert_choice`, но использует те же SAE/модельные настройки.
- Stage `prune` пока не реализует физическое удаление весов модели; это только формализация контракта артефакта.

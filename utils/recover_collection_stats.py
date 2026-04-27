#!/usr/bin/env python3
"""Validate and recover collection artifacts for pruning pipeline.

The script is designed for interrupted "collect" runs where only intermediate
checkpoint files (e.g. *_batch_1000.npy) are present.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


MEAN_RE = re.compile(r"^expert_(\d+)_mean(?P<suffix>.*)\.npy$")


@dataclass
class RecoveryReport:
    suffix: str
    num_experts: int
    num_saved_latents: int
    full_num_latents: int
    missing_experts: list[int]
    counts_per_expert: dict[int, int]
    repaired_files: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate/recover per-expert statistics and collection_stats.npy"
    )
    parser.add_argument(
        "--collection-dir",
        required=True,
        help="Path to layer directory with expert_* files (e.g. .../collection/layer_15)",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Suffix to use (example: _batch_50000). If omitted, auto-select latest.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Expected number of experts. If omitted, inferred from files.",
    )
    parser.add_argument(
        "--hook-layer",
        type=int,
        default=None,
        help="Hook layer id for metadata. If omitted, inferred from folder name layer_<id>.",
    )
    parser.add_argument(
        "--source-hook-layers",
        default=None,
        help="Comma-separated source hook layers for metadata (default: hook_layer only).",
    )
    parser.add_argument(
        "--latent-indices-path",
        default=None,
        help="Optional path to latent indices json. If omitted, uses range(num_saved_latents).",
    )
    parser.add_argument(
        "--full-num-latents",
        type=int,
        default=None,
        help="Optional full SAE latent count for metadata field num_latents.",
    )
    parser.add_argument(
        "--repair-missing",
        action="store_true",
        help="Create zero files for missing experts (mean/sum/sum_squared/std).",
    )
    parser.add_argument(
        "--write-stats",
        action="store_true",
        help="Write collection_stats.npy",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwrite of existing collection_stats.npy",
    )
    return parser.parse_args()


def list_suffixes_and_experts(collection_dir: str) -> dict[str, set[int]]:
    suffix_to_experts: dict[str, set[int]] = {}
    for name in os.listdir(collection_dir):
        match = MEAN_RE.match(name)
        if not match:
            continue
        expert_id = int(match.group(1))
        suffix = match.group("suffix") or ""
        suffix_to_experts.setdefault(suffix, set()).add(expert_id)
    return suffix_to_experts


def suffix_sort_key(suffix: str) -> tuple[int, str]:
    batch_match = re.search(r"_batch_(\d+)$", suffix)
    if batch_match:
        return (int(batch_match.group(1)), suffix)
    if suffix == "":
        return (-1, suffix)
    return (-1, suffix)


def select_suffix(explicit_suffix: str | None, suffix_map: dict[str, set[int]]) -> str:
    if explicit_suffix is not None:
        if explicit_suffix not in suffix_map:
            known = ", ".join(sorted(suffix_map.keys(), key=suffix_sort_key))
            raise ValueError(
                f"Suffix '{explicit_suffix}' not found. Available suffixes: {known}"
            )
        return explicit_suffix

    if not suffix_map:
        raise FileNotFoundError("No expert_*_mean*.npy files found in collection dir")

    return sorted(suffix_map.keys(), key=suffix_sort_key)[-1]


def infer_hook_layer(path: str) -> int:
    basename = os.path.basename(os.path.normpath(path))
    match = re.match(r"^layer_(\d+)$", basename)
    if match:
        return int(match.group(1))
    raise ValueError(
        "Cannot infer hook layer from folder name. Pass --hook-layer explicitly."
    )


def parse_source_hook_layers(value: str | None, hook_layer: int) -> list[int]:
    if value is None:
        return [hook_layer]
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return [hook_layer]
    return [int(part) for part in parts]


def read_latent_indices(path: str | None, num_saved_latents: int) -> list[int]:
    if path is None:
        return list(range(num_saved_latents))

    import json

    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        indices = list(json.load(f))
    if len(indices) != num_saved_latents:
        raise ValueError(
            "latent_indices length does not match saved vectors length: "
            f"{len(indices)} != {num_saved_latents}"
        )
    return [int(x) for x in indices]


def must_exist(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def infer_count(sum_arr: np.ndarray, mean_arr: np.ndarray, eps: float = 1e-12) -> int:
    mask = np.abs(mean_arr) > eps
    if not np.any(mask):
        return 0
    estimates = sum_arr[mask] / mean_arr[mask]
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return 0
    return max(0, int(round(float(np.median(finite)))))


def expert_file_paths(collection_dir: str, expert_id: int, suffix: str) -> dict[str, str]:
    return {
        "mean": os.path.join(collection_dir, f"expert_{expert_id}_mean{suffix}.npy"),
        "sum_squared": os.path.join(
            collection_dir, f"expert_{expert_id}_sum_squared{suffix}.npy"
        ),
        "sum": os.path.join(collection_dir, f"expert_{expert_id}_sum{suffix}.npy"),
        "std": os.path.join(collection_dir, f"expert_{expert_id}_std{suffix}.npy"),
    }


def discover_num_saved_latents(
    collection_dir: str,
    suffix: str,
    expert_ids: Iterable[int],
) -> int:
    for expert_id in expert_ids:
        mean_path = expert_file_paths(collection_dir, expert_id, suffix)["mean"]
        if os.path.exists(mean_path):
            arr = np.load(mean_path)
            if arr.ndim != 1:
                raise ValueError(f"Expected 1-D mean vector, got shape={arr.shape}: {mean_path}")
            return int(arr.shape[0])
    raise FileNotFoundError("No mean file found to infer latent vector size")


def write_zeros_for_missing_expert(
    collection_dir: str,
    expert_id: int,
    suffix: str,
    num_saved_latents: int,
) -> int:
    paths = expert_file_paths(collection_dir, expert_id, suffix)
    zeros = np.zeros(num_saved_latents, dtype=np.float64)
    repaired = 0
    for key in ("mean", "sum_squared", "sum", "std"):
        if not os.path.exists(paths[key]):
            np.save(paths[key], zeros)
            repaired += 1
    return repaired


def validate_and_recover(args: argparse.Namespace) -> RecoveryReport:
    collection_dir = os.path.expanduser(args.collection_dir)
    if not os.path.isdir(collection_dir):
        raise FileNotFoundError(f"Collection dir not found: {collection_dir}")

    suffix_map = list_suffixes_and_experts(collection_dir)
    suffix = select_suffix(args.suffix, suffix_map)
    experts_with_means = suffix_map[suffix]

    if args.num_experts is None:
        if not experts_with_means:
            raise ValueError("Cannot infer num_experts from empty expert set")
        num_experts = max(experts_with_means) + 1
    else:
        num_experts = args.num_experts

    expected_experts = list(range(num_experts))
    num_saved_latents = discover_num_saved_latents(collection_dir, suffix, expected_experts)

    repaired_files = 0
    counts_per_expert: dict[int, int] = {}
    missing_experts: list[int] = []

    for expert_id in expected_experts:
        paths = expert_file_paths(collection_dir, expert_id, suffix)
        has_mean = os.path.exists(paths["mean"])
        has_sum_sq = os.path.exists(paths["sum_squared"])

        if not has_mean or not has_sum_sq:
            missing_experts.append(expert_id)
            if args.repair_missing:
                repaired_files += write_zeros_for_missing_expert(
                    collection_dir,
                    expert_id,
                    suffix,
                    num_saved_latents,
                )
                counts_per_expert[expert_id] = 0
                continue
            counts_per_expert[expert_id] = 0
            continue

        mean_arr = np.load(paths["mean"]).astype(np.float64)
        sum_sq_arr = np.load(paths["sum_squared"]).astype(np.float64)
        if mean_arr.shape != (num_saved_latents,) or sum_sq_arr.shape != (num_saved_latents,):
            raise ValueError(
                f"Inconsistent shape for expert {expert_id}: "
                f"mean={mean_arr.shape}, sum_squared={sum_sq_arr.shape}, "
                f"expected={(num_saved_latents,)}"
            )

        if os.path.exists(paths["sum"]):
            raw_sum = np.load(paths["sum"]).astype(np.float64)
            if raw_sum.shape != (num_saved_latents,):
                raise ValueError(
                    f"Inconsistent sum shape for expert {expert_id}: {raw_sum.shape}"
                )
            counts_per_expert[expert_id] = infer_count(raw_sum, mean_arr)
        else:
            counts_per_expert[expert_id] = 0

    hook_layer = args.hook_layer if args.hook_layer is not None else infer_hook_layer(collection_dir)
    source_hook_layers = parse_source_hook_layers(args.source_hook_layers, hook_layer)

    full_num_latents = (
        args.full_num_latents if args.full_num_latents is not None else num_saved_latents
    )

    latent_indices = read_latent_indices(args.latent_indices_path, num_saved_latents)
    collection_stats = {
        "collection_mode": "statistics",
        "hook_layer": hook_layer,
        "source_hook_layers": source_hook_layers,
        "num_experts": num_experts,
        "num_latents": int(full_num_latents),
        "latent_indices": latent_indices,
        "num_saved_latents": int(num_saved_latents),
        "counts_per_expert": counts_per_expert,
        "total_samples": int(sum(counts_per_expert.values())),
    }

    if args.write_stats:
        stats_path = os.path.join(collection_dir, "collection_stats.npy")
        if os.path.exists(stats_path) and not args.overwrite:
            raise FileExistsError(
                f"{stats_path} already exists. Use --overwrite to replace it."
            )
        np.save(stats_path, np.array(collection_stats, dtype=object))

    return RecoveryReport(
        suffix=suffix,
        num_experts=num_experts,
        num_saved_latents=num_saved_latents,
        full_num_latents=full_num_latents,
        missing_experts=missing_experts,
        counts_per_expert=counts_per_expert,
        repaired_files=repaired_files,
    )


def main() -> None:
    args = parse_args()
    report = validate_and_recover(args)

    print("=== Recovery report ===")
    print(f"collection_dir: {os.path.expanduser(args.collection_dir)}")
    print(f"suffix: {report.suffix!r}")
    print(f"num_experts: {report.num_experts}")
    print(f"num_saved_latents: {report.num_saved_latents}")
    print(f"num_latents (metadata): {report.full_num_latents}")
    print(f"missing_experts: {len(report.missing_experts)}")
    if report.missing_experts:
        preview = report.missing_experts[:20]
        print(f"missing_expert_ids (first 20): {preview}")
    print(f"repaired_files: {report.repaired_files}")
    print(f"total_samples: {sum(report.counts_per_expert.values())}")
    print("status: ok")


if __name__ == "__main__":
    main()

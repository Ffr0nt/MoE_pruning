#!/usr/bin/env python3
"""Aggregate per-layer pruning_plan.json files into one YAML file.

Usage examples:
  python -m pruning.aggregate_pruning_plans --plans-root /path/to/pruning_choice
  python pruning/aggregate_pruning_plans.py --plans-root /path/to/pruning_choice
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan layer_* directories with pruning_plan.json and write a single YAML "
            "mapping: pruned_experts_by_layer"
        )
    )
    parser.add_argument(
        "--plans-root",
        default=".",
        help="Root directory that contains layer_* subdirectories (default: current dir).",
    )
    parser.add_argument(
        "--output",
        default="pruned_experts_by_layer.yaml",
        help="Output YAML filename inside --plans-root (default: pruned_experts_by_layer.yaml).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if any layer_* folder is missing pruning_plan.json. "
            "By default such folders are skipped."
        ),
    )
    return parser.parse_args()


def _layer_sort_key(layer_key: str) -> tuple[int, str]:
    try:
        return (int(layer_key), layer_key)
    except ValueError:
        return (10**9, layer_key)


def _normalize_expert_list(values: list[object]) -> list[int]:
    normalized = sorted({int(x) for x in values})
    return normalized


def _read_plan_file(path: Path) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    mapping = payload.get("experts_to_remove_by_layer")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(
            f"Invalid or empty experts_to_remove_by_layer in: {path}"
        )

    out: dict[str, list[int]] = {}
    for layer_key, expert_ids in mapping.items():
        if not isinstance(expert_ids, list):
            raise ValueError(f"Layer '{layer_key}' has non-list value in: {path}")
        out[str(layer_key)] = _normalize_expert_list(expert_ids)
    return out


def aggregate_plans(plans_root: Path, strict: bool) -> dict[str, list[int]]:
    if not plans_root.is_dir():
        raise FileNotFoundError(f"Plans root not found: {plans_root}")

    layer_dirs = sorted(
        [p for p in plans_root.iterdir() if p.is_dir() and p.name.startswith("layer_")],
        key=lambda p: p.name,
    )
    if not layer_dirs:
        raise FileNotFoundError(
            f"No layer_* subdirectories found in: {plans_root}"
        )

    aggregated: dict[str, list[int]] = {}

    for layer_dir in layer_dirs:
        plan_path = layer_dir / "pruning_plan.json"
        if not plan_path.exists():
            if strict:
                raise FileNotFoundError(f"Missing pruning_plan.json in: {layer_dir}")
            continue

        layer_mapping = _read_plan_file(plan_path)
        for layer_key, experts in layer_mapping.items():
            if layer_key in aggregated and aggregated[layer_key] != experts:
                raise ValueError(
                    "Conflicting plans for layer "
                    f"{layer_key}: {aggregated[layer_key]} vs {experts} (file: {plan_path})"
                )
            aggregated[layer_key] = experts

    if not aggregated:
        raise FileNotFoundError(
            f"No pruning_plan.json files found under: {plans_root}"
        )

    return dict(sorted(aggregated.items(), key=lambda kv: _layer_sort_key(kv[0])))


def write_yaml(output_path: Path, data: dict[str, list[int]]) -> None:
    lines = ["pruned_experts_by_layer:"]
    for layer_key, experts in data.items():
        experts_inline = ", ".join(str(int(x)) for x in experts)
        lines.append(f'  "{layer_key}": [{experts_inline}]')

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    plans_root = Path(os.path.expanduser(args.plans_root)).resolve()
    output_path = plans_root / args.output

    aggregated = aggregate_plans(plans_root=plans_root, strict=args.strict)
    write_yaml(output_path=output_path, data=aggregated)

    print(f"Saved: {output_path}")
    print(f"Layers: {len(aggregated)}")


if __name__ == "__main__":
    main()

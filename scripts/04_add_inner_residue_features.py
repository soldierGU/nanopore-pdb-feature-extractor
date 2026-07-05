#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_add_inner_residue_features.py

Purpose:
    Add pore-axis related features:
    center_x, center_y, radial_distance, is_inner_candidate.

Usage:
    python scripts/04_add_inner_residue_features.py --config configs/T232K.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return get_project_root() / path


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = resolve_project_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_input_file(cfg: Dict[str, Any]) -> Path:
    output_dir = resolve_project_path(cfg["output"]["residue_features_dir"])
    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    return output_dir / f"{nanopore_id}_residue_physchem.csv"


def compute_xy_center(df: pd.DataFrame, center_method: str) -> tuple[float, float]:
    if center_method != "xy_mean":
        raise ValueError(
            f"Unsupported center_method: {center_method}. "
            f"Currently only 'xy_mean' is implemented."
        )

    center_x = float(df["ca_x"].mean())
    center_y = float(df["ca_y"].mean())
    return center_x, center_y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    center_method = cfg.get("pore", {}).get("center_method", "xy_mean")
    inner_radius_threshold = float(cfg.get("pore", {}).get("inner_radius_threshold", 20.0))

    input_file = infer_input_file(cfg)
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input physchem table not found: {input_file}\n"
            f"Run 03_add_physicochemical_features.py first."
        )

    residue_features_dir = resolve_project_path(cfg["output"]["residue_features_dir"])
    inner_residues_dir = resolve_project_path(cfg["output"]["inner_residues_dir"])
    residue_features_dir.mkdir(parents=True, exist_ok=True)
    inner_residues_dir.mkdir(parents=True, exist_ok=True)

    output_residue_file = residue_features_dir / f"{nanopore_id}_residue_features.csv"
    output_inner_file = inner_residues_dir / f"{nanopore_id}_inner_residues.csv"

    print("=" * 80)
    print("04_add_inner_residue_features.py")
    print("=" * 80)
    print(f"Input : {input_file}")
    print(f"Output residue features: {output_residue_file}")
    print(f"Output inner residues   : {output_inner_file}")

    df = pd.read_csv(input_file)
    if "insertion_code" in df.columns:
        df["insertion_code"] = df["insertion_code"].fillna("")

    required_cols = ["ca_x", "ca_y"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    center_x, center_y = compute_xy_center(df, center_method=center_method)

    df["pore_center_x"] = center_x
    df["pore_center_y"] = center_y
    df["radial_distance"] = np.sqrt(
        (df["ca_x"] - center_x) ** 2 + (df["ca_y"] - center_y) ** 2
    )
    df["is_inner_candidate"] = df["radial_distance"] <= inner_radius_threshold

    df = df.sort_values(["chain_id", "residue_number", "insertion_code"]).reset_index(drop=True)

    inner_df = df[df["is_inner_candidate"]].copy()
    inner_df = inner_df.sort_values(["z_norm", "chain_id", "residue_number"]).reset_index(drop=True)

    df.to_csv(output_residue_file, index=False)
    inner_df.to_csv(output_inner_file, index=False)

    print(f"[OK] Residue feature table saved: {output_residue_file}")
    print(f"[OK] Inner residue table saved  : {output_inner_file}")
    print(f"[INFO] Total residues: {len(df)}")
    print(f"[INFO] Inner candidate residues: {len(inner_df)}")
    print(f"[INFO] Center method: {center_method}")
    print(f"[INFO] Pore center: x={center_x:.3f}, y={center_y:.3f}")
    print(f"[INFO] Inner radius threshold: {inner_radius_threshold:.3f} Angstrom")

    print("\nMutation rows:")
    mutation_rows = df[df["is_mutation_site"]].copy()
    if len(mutation_rows) > 0:
        print(
            mutation_rows[
                [
                    "chain_id",
                    "residue_number",
                    "residue_name",
                    "one_letter",
                    "mutation_label",
                    "z_norm",
                    "radial_distance",
                    "is_inner_candidate",
                    "charge_pH7",
                    "hydrophobicity",
                    "polarity_class",
                    "residue_volume",
                ]
            ].to_string(index=False)
        )
    else:
        print("[WARN] No mutation rows found.")


if __name__ == "__main__":
    main()

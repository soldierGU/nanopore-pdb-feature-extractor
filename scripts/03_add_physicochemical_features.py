#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_add_physicochemical_features.py

Purpose:
    Add residue-level physicochemical features:
    charge_pH7, hydrophobicity, polarity_class, residue_volume.

Usage:
    python scripts/03_add_physicochemical_features.py --config configs/T232K.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Approximate net charge at physiological pH.
CHARGE_PH7 = {
    "D": -1.0,
    "E": -1.0,
    "K": 1.0,
    "R": 1.0,
    "H": 0.1,
}

# Kyte-Doolittle hydrophobicity scale.
HYDROPHOBICITY_KD = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "E": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "N": -3.5,
    "K": -3.9,
    "R": -4.5,
}

POLARITY_CLASS = {
    "A": "nonpolar",
    "V": "nonpolar",
    "L": "nonpolar",
    "I": "nonpolar",
    "M": "nonpolar",
    "F": "aromatic",
    "W": "aromatic",
    "P": "nonpolar",
    "G": "special",
    "S": "polar_uncharged",
    "T": "polar_uncharged",
    "C": "polar_uncharged",
    "Y": "polar_aromatic",
    "N": "polar_uncharged",
    "Q": "polar_uncharged",
    "D": "negative",
    "E": "negative",
    "K": "positive",
    "R": "positive",
    "H": "weak_positive",
}

# Approximate residue volume in Å^3.
RESIDUE_VOLUME = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0,
}


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
    return output_dir / f"{nanopore_id}_residue_table.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_file = infer_input_file(cfg)
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input residue table not found: {input_file}\n"
            f"Run 02_build_residue_table.py first."
        )

    output_file = input_file.with_name(input_file.stem.replace("_residue_table", "_residue_physchem") + ".csv")

    print("=" * 80)
    print("03_add_physicochemical_features.py")
    print("=" * 80)
    print(f"Input : {input_file}")
    print(f"Output: {output_file}")

    df = pd.read_csv(input_file)

    if "one_letter" not in df.columns:
        if "residue_name" not in df.columns:
            raise ValueError("Input table must contain either 'one_letter' or 'residue_name'.")
        df["one_letter"] = df["residue_name"].map(AA3_TO_1).fillna("X")

    df["charge_pH7"] = df["one_letter"].map(CHARGE_PH7).fillna(0.0)
    df["hydrophobicity"] = df["one_letter"].map(HYDROPHOBICITY_KD)
    df["polarity_class"] = df["one_letter"].map(POLARITY_CLASS).fillna("unknown")
    df["residue_volume"] = df["one_letter"].map(RESIDUE_VOLUME)

    df["is_positive"] = df["one_letter"].isin(["K", "R", "H"])
    df["is_negative"] = df["one_letter"].isin(["D", "E"])
    df["is_charged"] = df["is_positive"] | df["is_negative"]

    df.to_csv(output_file, index=False)

    print(f"[OK] Physicochemical feature table saved: {output_file}")
    print(f"[INFO] Number of residues: {len(df)}")
    print("\nMutation rows:")
    print(df[df["is_mutation_site"]].to_string(index=False))


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_add_sasa_features.py

Purpose:
    Add Shrake-Rupley solvent-accessible surface area (SASA) features
    to the residue-level feature table.

Input:
    data/processed/residue_features/{nanopore_id}_residue_ss8.csv
    or data/processed/residue_features/{nanopore_id}_residue_features.csv

Output:
    data/processed/residue_features/{nanopore_id}_residue_features.csv
    data/processed/residue_features/{nanopore_id}_residue_sasa.csv

Usage:
    python scripts/06_add_sasa_features.py --config configs/T232K.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.PDB.SASA import ShrakeRupley


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nanopore_features.features.sasa import (
    add_sasa_features,
    build_sasa_dataframe,
    extract_sasa_feature_table,
)


def get_project_root() -> Path:
    return PROJECT_ROOT


def resolve_project_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return get_project_root() / path


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = resolve_project_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_input_file(cfg: Dict[str, Any]) -> Path:
    residue_features_dir = resolve_project_path(cfg["output"]["residue_features_dir"])
    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    ss8_file = residue_features_dir / f"{nanopore_id}_residue_ss8.csv"
    if ss8_file.exists():
        return ss8_file
    return residue_features_dir / f"{nanopore_id}_residue_features.csv"


def infer_output_file(cfg: Dict[str, Any]) -> Path:
    residue_features_dir = resolve_project_path(cfg["output"]["residue_features_dir"])
    residue_features_dir.mkdir(parents=True, exist_ok=True)

    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    return residue_features_dir / f"{nanopore_id}_residue_features.csv"


def infer_sasa_output_file(cfg: Dict[str, Any]) -> Path:
    residue_features_dir = resolve_project_path(cfg["output"]["residue_features_dir"])
    residue_features_dir.mkdir(parents=True, exist_ok=True)

    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    return residue_features_dir / f"{nanopore_id}_residue_sasa.csv"


def load_structure(structure_file: Path, file_format: str, structure_id: str):
    fmt = file_format.lower()
    if fmt == "pdb":
        parser = PDBParser(QUIET=True)
    elif fmt in {"cif", "mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file_format: {file_format}")

    return parser.get_structure(structure_id, str(structure_file))


def structure_to_sasa_dataframe(model) -> pd.DataFrame:
    sr = ShrakeRupley()
    sr.compute(model, level="R")

    rows = []
    for chain in model:
        for residue in chain:
            hetfield, residue_number, insertion_code = residue.id
            if hetfield.strip():
                continue

            aa = protein_letters_3to1.get(residue.get_resname().upper(), "X")
            sasa_abs = getattr(residue, "sasa", None)
            rows.append(
                {
                    "chain_id": chain.id,
                    "residue_number": int(residue_number),
                    "insertion_code": insertion_code.strip() if insertion_code else "",
                    "dssp_aa": aa,
                    "sasa_abs": sasa_abs,
                }
            )

    return pd.DataFrame(rows)


def run_structure_sasa(cfg: Dict[str, Any]) -> pd.DataFrame:
    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    structure_file = resolve_project_path(cfg["input"]["structure_file"])
    file_format = cfg["input"].get("file_format", "pdb")
    model_index = int(cfg["structure"].get("model_index", 0))

    if not structure_file.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_file}")

    structure = load_structure(structure_file, file_format, nanopore_id)
    models = list(structure)
    if model_index >= len(models):
        raise IndexError(f"model_index={model_index} out of range. Available models: {len(models)}")

    return structure_to_sasa_dataframe(models[model_index])


def print_summary(df: pd.DataFrame) -> None:
    print("\nSASA summary:")
    print("-" * 80)

    total = len(df)
    matched = int(df["has_sasa"].sum())
    print(f"Total residues      : {total}")
    print(f"SASA matched residues: {matched}")
    print(f"SASA match ratio    : {matched / total:.4f}")

    print("\nExposure class counts:")
    print(df["sasa_exposure_class"].value_counts(dropna=False).sort_index().to_string())

    print("\nSASA relative statistics:")
    print(df["sasa_rel"].describe().to_string())

    mutation_rows = df[df["is_mutation_site"]].copy() if "is_mutation_site" in df.columns else pd.DataFrame()
    print("\nMutation rows:")
    if len(mutation_rows) == 0:
        print("[WARN] No mutation rows found.")
        return

    show_cols = [
        "chain_id",
        "residue_number",
        "residue_name",
        "one_letter",
        "mutation_label",
        "z_norm",
        "radial_distance",
        "is_inner_candidate",
        "sasa_rel",
        "sasa_abs",
        "sasa_exposure_class",
        "is_surface_exposed",
        "is_buried",
    ]
    show_cols = [col for col in show_cols if col in mutation_rows.columns]
    print(mutation_rows[show_cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    sasa_cfg = cfg.get("sasa", {})
    buried_threshold = float(sasa_cfg.get("buried_threshold", 0.05))
    exposed_threshold = float(sasa_cfg.get("exposed_threshold", 0.25))

    input_file = infer_input_file(cfg)
    output_file = infer_output_file(cfg)
    sasa_output_file = infer_sasa_output_file(cfg)

    print("=" * 80)
    print("06_add_sasa_features.py")
    print("=" * 80)
    print(f"nanopore_id      : {nanopore_id}")
    print(f"input table      : {input_file}")
    print(f"output table     : {output_file}")
    print(f"output SASA table: {sasa_output_file}")
    print(f"buried_threshold : {buried_threshold}")
    print(f"exposed_threshold: {exposed_threshold}")

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input residue feature table not found: {input_file}\n"
            f"Run 04_add_inner_residue_features.py first, and optionally 05_add_ss8_features.py."
        )

    residue_df = pd.read_csv(input_file)
    if "insertion_code" in residue_df.columns:
        residue_df["insertion_code"] = residue_df["insertion_code"].fillna("").astype(str)

    dssp_sasa_df = run_structure_sasa(cfg)
    sasa_df = build_sasa_dataframe(
        dssp_sasa_df,
        buried_threshold=buried_threshold,
        exposed_threshold=exposed_threshold,
    )

    merged = add_sasa_features(
        residue_df=residue_df,
        sasa_df=sasa_df,
        buried_threshold=buried_threshold,
        exposed_threshold=exposed_threshold,
    )
    merged = merged.sort_values(["chain_id", "residue_number", "insertion_code"]).reset_index(drop=True)
    merged.to_csv(output_file, index=False)

    sasa_feature_table = extract_sasa_feature_table(merged)
    sasa_feature_table.to_csv(sasa_output_file, index=False)

    print(f"\n[OK] Residue feature table saved: {output_file}")
    print(f"[OK] SASA-only feature table saved: {sasa_output_file}")
    print_summary(merged)


if __name__ == "__main__":
    main()

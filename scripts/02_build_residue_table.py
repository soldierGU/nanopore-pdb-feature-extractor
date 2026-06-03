#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_build_residue_table.py

Purpose:
    Build a basic residue-level table from PDB.
    Extract residue identity, chain, residue number, CA coordinates, z, z_norm,
    and mutation site labels.

Usage:
    python scripts/02_build_residue_table.py --config configs/T232K.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import protein_letters_3to1


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


def load_structure(structure_file: Path, file_format: str, structure_id: str):
    if file_format.lower() == "pdb":
        parser = PDBParser(QUIET=True)
    elif file_format.lower() in {"cif", "mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file_format: {file_format}")
    return parser.get_structure(structure_id, str(structure_file))


def is_standard_residue(residue) -> bool:
    hetfield, _, _ = residue.id
    if hetfield.strip():
        return False
    return residue.get_resname().upper() in protein_letters_3to1


def residue_to_one_letter(residue_name: str) -> str:
    return protein_letters_3to1.get(residue_name.upper(), "X")


def get_mutation_label(
    chain_id: str,
    residue_number: int,
    one_letter: str,
    mutations: List[Dict[str, Any]],
) -> tuple[bool, str]:
    labels = []

    for mut in mutations:
        mut_pos = int(mut["residue_number"])
        chain_scope = mut.get("chain_scope", "all")

        if chain_scope == "all":
            chain_match = True
        elif isinstance(chain_scope, list):
            chain_match = chain_id in chain_scope
        else:
            chain_match = chain_id == str(chain_scope)

        if chain_match and residue_number == mut_pos:
            labels.append(mut.get("label", f"{mut.get('original', '?')}{mut_pos}{mut.get('mutant', '?')}"))

    if labels:
        return True, ";".join(labels)
    return False, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    pdb_id = cfg["input"].get("pdb_id", "unknown_pdb")
    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    structure_file = resolve_project_path(cfg["input"]["structure_file"])
    file_format = cfg["input"].get("file_format", "pdb")
    model_index = int(cfg["structure"].get("model_index", 0))
    expected_chains = cfg["structure"].get("expected_chains", [])
    standard_only = bool(cfg["structure"].get("standard_residues_only", True))
    mutations = cfg.get("mutation", {}).get("mutations", [])

    output_dir = resolve_project_path(cfg["output"]["residue_features_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "T232K_residue_table.csv"
    if nanopore_id:
        output_file = output_dir / f"{nanopore_id}_residue_table.csv"

    print("=" * 80)
    print("02_build_residue_table.py")
    print("=" * 80)
    print(f"Input : {structure_file}")
    print(f"Output: {output_file}")

    structure = load_structure(structure_file, file_format, nanopore_id)
    model = list(structure)[model_index]

    rows = []

    for chain_id in expected_chains:
        if chain_id not in model:
            print(f"[WARN] Expected chain {chain_id} not found. Skipping.")
            continue

        chain = model[chain_id]

        for residue in chain:
            if standard_only and not is_standard_residue(residue):
                continue

            if "CA" not in residue:
                continue

            hetfield, residue_number, insertion_code = residue.id
            residue_name = residue.get_resname().upper()
            one_letter = residue_to_one_letter(residue_name)

            ca = residue["CA"].coord
            ca_x, ca_y, ca_z = float(ca[0]), float(ca[1]), float(ca[2])

            is_mutation_site, mutation_label = get_mutation_label(
                chain_id=chain_id,
                residue_number=int(residue_number),
                one_letter=one_letter,
                mutations=mutations,
            )

            rows.append(
                {
                    "pdb_id": pdb_id,
                    "nanopore_id": nanopore_id,
                    "chain_id": chain_id,
                    "residue_number": int(residue_number),
                    "insertion_code": insertion_code.strip() if insertion_code else "",
                    "residue_name": residue_name,
                    "one_letter": one_letter,
                    "is_mutation_site": bool(is_mutation_site),
                    "mutation_label": mutation_label,
                    "ca_x": ca_x,
                    "ca_y": ca_y,
                    "ca_z": ca_z,
                    "z": ca_z,
                }
            )

    if not rows:
        raise RuntimeError("No residues were extracted. Check PDB file and expected_chains.")

    df = pd.DataFrame(rows)

    z_min = df["z"].min()
    z_max = df["z"].max()
    if np.isclose(z_max, z_min):
        df["z_norm"] = 0.0
    else:
        df["z_norm"] = (df["z"] - z_min) / (z_max - z_min)

    df = df.sort_values(["chain_id", "residue_number", "insertion_code"]).reset_index(drop=True)
    df.to_csv(output_file, index=False)

    print(f"[OK] Residue table saved: {output_file}")
    print(f"[INFO] Number of residues: {len(df)}")
    print(f"[INFO] z range: {z_min:.3f} to {z_max:.3f}")
    print("\nMutation rows:")
    print(df[df["is_mutation_site"]].to_string(index=False))


if __name__ == "__main__":
    main()
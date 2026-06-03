#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_check_pdb.py

Purpose:
    Check whether the input PDB file can be parsed correctly.
    Check expected chains and mutation sites.

Usage:
    python scripts/01_check_pdb.py --config configs/T232K.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import protein_letters_3to1


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = get_project_root() / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return get_project_root() / path


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


def find_residue_by_number(chain, residue_number: int):
    matches = []
    for residue in chain:
        hetfield, resseq, icode = residue.id
        if hetfield.strip():
            continue
        if resseq == residue_number:
            matches.append(residue)

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(
            f"[WARN] Multiple residues found at position {residue_number} "
            f"in chain {chain.id}. Using the first one."
        )
    return matches[0]


def summarize_chain(chain, standard_only: bool = True) -> Tuple[int, int]:
    total_residues = 0
    ca_residues = 0

    for residue in chain:
        if standard_only and not is_standard_residue(residue):
            continue
        total_residues += 1
        if "CA" in residue:
            ca_residues += 1

    return total_residues, ca_residues


def check_mutations(model, expected_chains: List[str], mutations: List[Dict[str, Any]]) -> bool:
    all_ok = True

    print("\nMutation site check:")
    print("-" * 80)

    for mut in mutations:
        residue_number = int(mut["residue_number"])
        expected_mutant = mut["mutant"]
        label = mut.get("label", f"{mut.get('original', '?')}{residue_number}{expected_mutant}")
        chain_scope = mut.get("chain_scope", "all")

        if chain_scope == "all":
            chains_to_check = expected_chains
        elif isinstance(chain_scope, list):
            chains_to_check = chain_scope
        else:
            chains_to_check = [str(chain_scope)]

        for chain_id in chains_to_check:
            if chain_id not in model:
                print(f"[FAIL] {label}: chain {chain_id} not found.")
                all_ok = False
                continue

            chain = model[chain_id]
            residue = find_residue_by_number(chain, residue_number)

            if residue is None:
                print(f"[FAIL] {label}: residue {residue_number} not found in chain {chain_id}.")
                all_ok = False
                continue

            residue_name = residue.get_resname().upper()
            one_letter = residue_to_one_letter(residue_name)

            if one_letter == expected_mutant:
                status = "OK"
            else:
                status = "FAIL"
                all_ok = False

            print(
                f"[{status}] {label} | chain={chain_id} | "
                f"position={residue_number} | observed={residue_name}({one_letter}) | "
                f"expected={expected_mutant}"
            )

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    structure_file = resolve_project_path(cfg["input"]["structure_file"])
    file_format = cfg["input"].get("file_format", "pdb")
    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    model_index = int(cfg["structure"].get("model_index", 0))
    expected_chains = cfg["structure"].get("expected_chains", [])
    standard_only = bool(cfg["structure"].get("standard_residues_only", True))
    mutations = cfg.get("mutation", {}).get("mutations", [])

    print("=" * 80)
    print("01_check_pdb.py")
    print("=" * 80)
    print(f"structure_file: {structure_file}")
    print(f"file_format   : {file_format}")
    print(f"nanopore_id   : {nanopore_id}")

    if not structure_file.exists():
        raise FileNotFoundError(f"PDB file not found: {structure_file}")

    structure = load_structure(structure_file, file_format, nanopore_id)

    models = list(structure)
    if model_index >= len(models):
        raise IndexError(f"model_index={model_index} out of range. Available models: {len(models)}")

    model = models[model_index]

    print("\nModel check:")
    print("-" * 80)
    print(f"Number of models: {len(models)}")
    print(f"Using model_index: {model_index}")

    observed_chains = [chain.id for chain in model]
    print(f"Observed chains: {observed_chains}")
    print(f"Expected chains: {expected_chains}")

    print("\nChain summary:")
    print("-" * 80)

    chain_ok = True
    for chain_id in expected_chains:
        if chain_id not in model:
            print(f"[FAIL] chain {chain_id}: not found")
            chain_ok = False
            continue

        chain = model[chain_id]
        total_residues, ca_residues = summarize_chain(chain, standard_only=standard_only)
        print(
            f"[OK] chain {chain_id}: "
            f"standard_residues={total_residues}, residues_with_CA={ca_residues}"
        )

    mutation_ok = check_mutations(model, expected_chains, mutations)

    print("\nFinal result:")
    print("-" * 80)
    if chain_ok and mutation_ok:
        print("[PASS] PDB check passed.")
    else:
        print("[FAIL] PDB check failed. Please inspect chain IDs, residue numbering, or mutation file.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
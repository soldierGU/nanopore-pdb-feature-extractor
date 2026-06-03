#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05_add_ss8_features.py

Purpose:
    Add DSSP-derived SS8 and SS3 secondary structure features
    to the residue-level feature table.

Input:
    data/processed/residue_features/{nanopore_id}_residue_features.csv

Output:
    data/processed/residue_features/{nanopore_id}_residue_ss8.csv

Usage:
    python scripts/05_add_ss8_features.py --config configs/T232K.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import protein_letters_3to1


SS8_ALLOWED = {"H", "B", "E", "G", "I", "T", "S", "-", "C"}

SS8_TO_SS3 = {
    "H": "helix",
    "G": "helix",
    "I": "helix",
    "E": "strand",
    "B": "strand",
    "T": "coil",
    "S": "coil",
    "-": "coil",
    "C": "coil",
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
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_input_residue_file(cfg: Dict[str, Any]) -> Path:
    residue_features_dir = resolve_project_path(cfg["output"]["residue_features_dir"])
    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    return residue_features_dir / f"{nanopore_id}_residue_features.csv"


def infer_output_file(cfg: Dict[str, Any]) -> Path:
    residue_features_dir = resolve_project_path(cfg["output"]["residue_features_dir"])
    residue_features_dir.mkdir(parents=True, exist_ok=True)

    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    return residue_features_dir / f"{nanopore_id}_residue_ss8.csv"


def configure_libcifpp_data_dir(dssp_executable_path: str | Path) -> None:
    if os.environ.get("LIBCIFPP_DATA_DIR"):
        return

    exe_path = Path(dssp_executable_path).resolve()
    candidates = [
        exe_path.parents[2] / "share" / "libcifpp",
        exe_path.parents[1] / "share" / "libcifpp",
    ]

    for candidate in candidates:
        if (candidate / "components.cif").exists() and (candidate / "mmcif_pdbx.dic").exists():
            os.environ["LIBCIFPP_DATA_DIR"] = str(candidate)
            print(f"[INFO] LIBCIFPP_DATA_DIR set to: {candidate}")
            return


def check_dssp_executable(executable: str) -> None:
    found = shutil.which(executable)
    if found is None:
        raise FileNotFoundError(
            f"DSSP executable not found: {executable}\n"
            f"Install it first, for example:\n"
            f"    conda install -c conda-forge dssp\n"
            f"Then check:\n"
            f"    mkdssp --version\n"
        )

    configure_libcifpp_data_dir(found)

    try:
        result = subprocess.run(
            [executable, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        version_text = (result.stdout or result.stderr).strip()
        print(f"[OK] DSSP executable found: {found}")
        if version_text:
            print(f"[INFO] DSSP version: {version_text}")
    except Exception as exc:
        print(f"[WARN] Could not query DSSP version: {exc}")


def load_structure(structure_file: Path, file_format: str, structure_id: str):
    fmt = file_format.lower()
    if fmt == "pdb":
        parser = PDBParser(QUIET=True)
    elif fmt in {"cif", "mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file_format: {file_format}")

    return parser.get_structure(structure_id, str(structure_file))


def get_file_type_for_biopython(file_format: str) -> str:
    fmt = file_format.lower()
    if fmt == "pdb":
        return "PDB"
    if fmt in {"cif", "mmcif"}:
        return "MMCIF"
    raise ValueError(f"Unsupported file_format: {file_format}")


def clean_ss8(ss8: str, unknown_value: str = "C") -> str:
    if ss8 is None:
        return unknown_value

    ss8 = str(ss8).strip()
    if ss8 == "":
        return unknown_value

    if ss8 not in SS8_ALLOWED:
        return unknown_value

    # DSSP uses "-" for coil; for feature table, C is easier to read.
    if ss8 == "-":
        return "C"

    return ss8


def ss8_to_ss3(ss8: str) -> str:
    return SS8_TO_SS3.get(ss8, "coil")


def three_to_one(resname: str) -> str:
    return protein_letters_3to1.get(str(resname).upper(), "X")


def make_residue_key(chain_id: str, residue_number: int, insertion_code: str) -> Tuple[str, Tuple[str, int, str]]:
    icode = insertion_code if isinstance(insertion_code, str) and insertion_code.strip() else " "
    return chain_id, (" ", int(residue_number), icode)


def dssp_to_dataframe(dssp: DSSP, unknown_value: str = "C") -> pd.DataFrame:
    rows = []

    for key in dssp.keys():
        chain_id, residue_id = key
        hetfield, residue_number, insertion_code = residue_id

        values = dssp[key]

        # Biopython DSSP tuple indices:
        # 0 dssp index, 1 amino acid, 2 secondary structure,
        # 3 relative ASA, 4 phi, 5 psi, ...
        aa = values[1]
        ss8_raw = values[2]
        ss8 = clean_ss8(ss8_raw, unknown_value=unknown_value)
        ss3 = ss8_to_ss3(ss8)

        rows.append(
            {
                "chain_id": chain_id,
                "residue_number": int(residue_number),
                "insertion_code": insertion_code.strip() if insertion_code else "",
                "dssp_aa": aa,
                "ss8": ss8,
                "ss3": ss3,
                "has_dssp": True,
            }
        )

    return pd.DataFrame(rows)


def add_ss8_features(
    residue_df: pd.DataFrame,
    dssp_df: pd.DataFrame,
    unknown_value: str = "C",
) -> pd.DataFrame:
    df = residue_df.copy()

    if "insertion_code" not in df.columns:
        df["insertion_code"] = ""
    df["insertion_code"] = df["insertion_code"].fillna("").astype(str)

    dssp_df = dssp_df.copy()
    dssp_df["insertion_code"] = dssp_df["insertion_code"].fillna("").astype(str)

    merged = df.merge(
        dssp_df,
        on=["chain_id", "residue_number", "insertion_code"],
        how="left",
    )

    merged["has_dssp"] = merged["has_dssp"].fillna(False).astype(bool)
    merged["ss8"] = merged["ss8"].fillna(unknown_value)
    merged["ss3"] = merged["ss8"].map(ss8_to_ss3).fillna("coil")

    return merged


def print_summary(df: pd.DataFrame) -> None:
    print("\nSS8 summary:")
    print("-" * 80)

    total = len(df)
    matched = int(df["has_dssp"].sum())
    print(f"Total residues       : {total}")
    print(f"DSSP matched residues: {matched}")
    print(f"DSSP match ratio     : {matched / total:.4f}")

    print("\nSS8 counts:")
    print(df["ss8"].value_counts(dropna=False).sort_index().to_string())

    print("\nSS3 counts:")
    print(df["ss3"].value_counts(dropna=False).sort_index().to_string())

    missing = df[df["has_dssp"] == False].copy()
    print("\nDSSP-unmatched residues:")
    if len(missing) == 0:
        print("[OK] All residues matched DSSP.")
    else:
        show_cols = [
            "chain_id",
            "residue_number",
            "insertion_code",
            "residue_name",
            "one_letter",
            "z_norm",
            "is_inner_candidate",
            "is_mutation_site",
        ]
        show_cols = [col for col in show_cols if col in missing.columns]
        print(missing[show_cols].to_string(index=False))

    mutation_rows = df[df["is_mutation_site"]].copy()
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
        "ss8",
        "ss3",
        "has_dssp",
    ]
    show_cols = [col for col in show_cols if col in mutation_rows.columns]
    print(mutation_rows[show_cols].to_string(index=False))

def infer_element_from_atom_name(atom_name: str) -> str:
    """
    Infer element symbol from PDB atom name.

    For standard protein ATOM records:
        " CA " means alpha carbon -> C
        " N  " -> N
        " O  " -> O
        " CB " -> C
    """
    name = atom_name.strip()

    if not name:
        return ""

    # Remove leading digits if present, e.g. 1HG -> H
    while name and name[0].isdigit():
        name = name[1:]

    if not name:
        return ""

    # For standard protein atoms, first alphabetic character is usually enough.
    first = name[0].upper()

    if first in {"C", "N", "O", "S", "H", "P"}:
        return first

    return first


def sanitize_pdb_for_dssp(input_pdb: Path, output_pdb: Path) -> Path:
    """
    Create a DSSP-friendly PDB file.

    Fixes:
        1. Remove FoldX/non-PDB header records.
        2. Keep only ATOM/HETATM/TER/END.
        3. Pad ATOM/HETATM lines to 80 characters.
        4. Fill element column if missing.
    """
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    fixed_element = 0
    skipped = 0

    with open(input_pdb, "r", encoding="utf-8", errors="ignore") as fin, \
            open(output_pdb, "w", encoding="utf-8") as fout:

        for raw_line in fin:
            line = raw_line.rstrip("\n\r")
            record = line[0:6].strip()

            if record in {"ATOM", "HETATM"}:
                if len(line) < 80:
                    line = line.ljust(80)

                atom_name = line[12:16]
                element = line[76:78].strip()

                if not element:
                    inferred = infer_element_from_atom_name(atom_name)
                    line = line[:76] + inferred.rjust(2) + line[78:]
                    fixed_element += 1

                fout.write(line[:80] + "\n")
                kept += 1

            elif record == "TER":
                fout.write("TER\n")
                kept += 1

            elif record == "END":
                fout.write("END\n")
                kept += 1

            else:
                skipped += 1

        fout.write("END\n")

    print(f"[OK] Cleaned PDB for DSSP: {output_pdb}")
    print(f"[INFO] PDB records kept: {kept}")
    print(f"[INFO] Atom element fields fixed: {fixed_element}")
    print(f"[INFO] Non-structural records skipped: {skipped}")

    return output_pdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    structure_file = resolve_project_path(cfg["input"]["structure_file"])
    file_format = cfg["input"].get("file_format", "pdb")
    model_index = int(cfg["structure"].get("model_index", 0))

    dssp_cfg = cfg.get("dssp", {})
    dssp_executable = dssp_cfg.get("executable", "mkdssp")
    ss8_unknown = dssp_cfg.get("ss8_unknown", "C")

    input_residue_file = infer_input_residue_file(cfg)
    output_file = infer_output_file(cfg)

    print("=" * 80)
    print("05_add_ss8_features.py")
    print("=" * 80)
    print(f"structure_file      : {structure_file}")
    print(f"file_format         : {file_format}")
    print(f"nanopore_id         : {nanopore_id}")
    print(f"input residue table : {input_residue_file}")
    print(f"output SS8 table    : {output_file}")
    print(f"dssp executable     : {dssp_executable}")

    if not structure_file.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_file}")

    if not input_residue_file.exists():
        raise FileNotFoundError(
            f"Input residue feature table not found: {input_residue_file}\n"
            f"Run 04_add_inner_residue_features.py first."
        )

    check_dssp_executable(dssp_executable)

    residue_df = pd.read_csv(input_residue_file)
    if "insertion_code" in residue_df.columns:
        residue_df["insertion_code"] = residue_df["insertion_code"].fillna("").astype(str)

    # DSSP is strict about PDB format. FoldX output may miss element columns.
    # Therefore, for PDB input, generate a cleaned temporary PDB before DSSP.
    if file_format.lower() == "pdb":
        tmp_dir = resolve_project_path("data/processed/tmp_dssp")
        dssp_input_file = tmp_dir / f"{nanopore_id}_dssp_input.pdb"
        dssp_input_file = sanitize_pdb_for_dssp(structure_file, dssp_input_file)

        structure_for_dssp = load_structure(
            dssp_input_file,
            file_format="pdb",
            structure_id=nanopore_id,
        )
        file_type = "PDB"
    else:
        dssp_input_file = structure_file
        structure_for_dssp = load_structure(
            structure_file,
            file_format=file_format,
            structure_id=nanopore_id,
        )
        file_type = get_file_type_for_biopython(file_format)

    models = list(structure_for_dssp)
    if model_index >= len(models):
        raise IndexError(f"model_index={model_index} out of range. Available models: {len(models)}")

    model = models[model_index]

    try:
        dssp = DSSP(
            model=model,
            in_file=str(dssp_input_file),
            dssp=dssp_executable,
            file_type=file_type,
        )
    except TypeError:
        # Older Biopython versions may not support file_type explicitly.
        dssp = DSSP(
            model,
            str(dssp_input_file),
            dssp=dssp_executable,
        )

    dssp_df = dssp_to_dataframe(dssp, unknown_value=ss8_unknown)

    merged = add_ss8_features(
        residue_df=residue_df,
        dssp_df=dssp_df,
        unknown_value=ss8_unknown,
    )

    merged = merged.sort_values(["chain_id", "residue_number", "insertion_code"]).reset_index(drop=True)
    merged.to_csv(output_file, index=False)

    print(f"\n[OK] SS8 feature table saved: {output_file}")
    print_summary(merged)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
07_build_zbin_profile.py

Purpose:
    Build a z-axis binned nanopore profile from residue-level features.

Input:
    data/processed/residue_features/{nanopore_id}_residue_features.csv

Output:
    data/processed/pore_profiles/{nanopore_id}_zbin_profile.csv

Usage:
    python scripts/07_build_zbin_profile.py --config configs/T232K.yaml
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
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_input_file(cfg: Dict[str, Any]) -> Path:
    residue_features_dir = resolve_project_path(cfg["output"]["residue_features_dir"])
    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    return residue_features_dir / f"{nanopore_id}_residue_features.csv"


def infer_output_file(cfg: Dict[str, Any]) -> Path:
    pore_profiles_dir = resolve_project_path(
        cfg["output"].get("pore_profiles_dir", "data/processed/pore_profiles")
    )
    pore_profiles_dir.mkdir(parents=True, exist_ok=True)

    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")
    return pore_profiles_dir / f"{nanopore_id}_zbin_profile.csv"


def check_required_columns(df: pd.DataFrame) -> None:
    required_cols = [
        "pdb_id",
        "nanopore_id",
        "chain_id",
        "residue_number",
        "residue_name",
        "one_letter",
        "z_norm",
        "radial_distance",
        "is_inner_candidate",
        "is_mutation_site",
        "charge_pH7",
        "hydrophobicity",
        "residue_volume",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in residue feature table: {missing}")


def safe_mean(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return float(valid.mean())


def safe_std(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) <= 1:
        return 0.0
    return float(valid.std(ddof=1))


def assign_z_bins(df: pd.DataFrame, zbin_count: int) -> pd.DataFrame:
    df = df.copy()

    if zbin_count <= 0:
        raise ValueError("zbin_count must be a positive integer.")

    # Clip avoids z_norm=1.0 being assigned to bin index zbin_count.
    z_norm_clipped = df["z_norm"].clip(lower=0.0, upper=np.nextafter(1.0, 0.0))
    df["z_bin_id"] = np.floor(z_norm_clipped * zbin_count).astype(int)

    df["z_bin_start"] = df["z_bin_id"] / zbin_count
    df["z_bin_end"] = (df["z_bin_id"] + 1) / zbin_count
    df["z_bin_center"] = (df["z_bin_start"] + df["z_bin_end"]) / 2.0

    return df


def build_empty_profile_template(
    cfg: Dict[str, Any],
    zbin_count: int,
) -> pd.DataFrame:
    pdb_id = cfg["input"].get("pdb_id", "unknown_pdb")
    nanopore_id = cfg["input"].get("nanopore_id", "unknown_nanopore")

    rows = []
    for bin_id in range(zbin_count):
        z_start = bin_id / zbin_count
        z_end = (bin_id + 1) / zbin_count
        z_center = (z_start + z_end) / 2.0

        rows.append(
            {
                "pdb_id": pdb_id,
                "nanopore_id": nanopore_id,
                "z_bin_id": bin_id,
                "z_bin_start": z_start,
                "z_bin_end": z_end,
                "z_bin_center": z_center,
            }
        )

    return pd.DataFrame(rows)


def aggregate_zbin_profile(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    zbin_count: int,
) -> pd.DataFrame:
    template = build_empty_profile_template(cfg, zbin_count=zbin_count)

    grouped_rows = []

    for z_bin_id, g in df.groupby("z_bin_id"):
        row = {
            "z_bin_id": int(z_bin_id),

            # Count features.
            "residue_count": int(len(g)),
            "chain_count": int(g["chain_id"].nunique()),
            "mutation_site_count": int(g["is_mutation_site"].sum()),

            # Geometry features.
            "z_norm_mean": safe_mean(g["z_norm"]),
            "z_norm_min": float(g["z_norm"].min()),
            "z_norm_max": float(g["z_norm"].max()),
            "radial_distance_mean": safe_mean(g["radial_distance"]),
            "radial_distance_min": float(g["radial_distance"].min()),
            "radial_distance_max": float(g["radial_distance"].max()),
            "radial_distance_std": safe_std(g["radial_distance"]),

            # Physicochemical features.
            "charge_sum": float(g["charge_pH7"].sum()),
            "charge_mean": safe_mean(g["charge_pH7"]),
            "positive_count": int(g.get("is_positive", pd.Series(False, index=g.index)).sum()),
            "negative_count": int(g.get("is_negative", pd.Series(False, index=g.index)).sum()),
            "charged_count": int(g.get("is_charged", pd.Series(False, index=g.index)).sum()),

            "hydrophobicity_mean": safe_mean(g["hydrophobicity"]),
            "hydrophobicity_std": safe_std(g["hydrophobicity"]),

            "residue_volume_mean": safe_mean(g["residue_volume"]),
            "residue_volume_sum": float(g["residue_volume"].sum()),

            # Mutation-related summaries.
            "has_mutation_site": bool(g["is_mutation_site"].any()),
            "mutation_labels": ";".join(
                sorted(
                    {
                        str(x)
                        for x in g.loc[g["is_mutation_site"], "mutation_label"].dropna().tolist()
                        if str(x).strip() != ""
                    }
                )
            ),
        }

        # Amino-acid composition, useful for quick inspection.
        aa_counts = g["one_letter"].value_counts()
        for aa in list("ACDEFGHIKLMNPQRSTVWY"):
            row[f"aa_{aa}_count"] = int(aa_counts.get(aa, 0))

        if "sasa_rel" in g.columns:
            row["sasa_rel_mean"] = safe_mean(g["sasa_rel"])
            row["sasa_rel_min"] = float(g["sasa_rel"].min())
            row["sasa_rel_max"] = float(g["sasa_rel"].max())
            row["sasa_rel_std"] = safe_std(g["sasa_rel"])
        if "sasa_abs" in g.columns:
            row["sasa_abs_mean"] = safe_mean(g["sasa_abs"])
            row["sasa_abs_sum"] = float(g["sasa_abs"].sum())
        if "is_surface_exposed" in g.columns:
            row["surface_exposed_count"] = int(g["is_surface_exposed"].sum())
        if "is_buried" in g.columns:
            row["buried_count"] = int(g["is_buried"].sum())

        grouped_rows.append(row)

    profile = pd.DataFrame(grouped_rows)

    if len(profile) == 0:
        profile = template.copy()
    else:
        profile = template.merge(profile, on="z_bin_id", how="left")

    # Fill empty bins.
    count_cols = [
        "residue_count",
        "chain_count",
        "mutation_site_count",
        "positive_count",
        "negative_count",
        "charged_count",
        "surface_exposed_count",
        "buried_count",
    ] + [f"aa_{aa}_count" for aa in list("ACDEFGHIKLMNPQRSTVWY")]

    for col in count_cols:
        if col in profile.columns:
            profile[col] = profile[col].fillna(0).astype(int)

    if "has_mutation_site" in profile.columns:
        profile["has_mutation_site"] = profile["has_mutation_site"].fillna(False).astype(bool)
    else:
        profile["has_mutation_site"] = False

    if "mutation_labels" in profile.columns:
        profile["mutation_labels"] = profile["mutation_labels"].fillna("")
    else:
        profile["mutation_labels"] = ""

    numeric_fill_zero_cols = [
        "charge_sum",
        "residue_volume_sum",
        "sasa_abs_sum",
    ]
    for col in numeric_fill_zero_cols:
        if col in profile.columns:
            profile[col] = profile[col].fillna(0.0)

    return profile


def print_profile_summary(profile: pd.DataFrame) -> None:
    print("\nZ-bin profile summary:")
    print("-" * 80)

    print(f"Number of z-bins: {len(profile)}")
    print(f"Total residues in profile: {int(profile['residue_count'].sum())}")

    mutation_bins = profile[profile["has_mutation_site"]]
    if len(mutation_bins) > 0:
        print("\nMutation-containing bins:")
        show_cols = [
            "z_bin_id",
            "z_bin_start",
            "z_bin_end",
            "residue_count",
            "mutation_site_count",
            "mutation_labels",
            "charge_sum",
            "charge_mean",
            "hydrophobicity_mean",
            "radial_distance_mean",
        ]
        show_cols = [col for col in show_cols if col in mutation_bins.columns]
        print(mutation_bins[show_cols].to_string(index=False))
    else:
        print("\n[WARN] No mutation site found in z-bin profile.")

    print("\nTop bins by positive residue count:")
    show_cols = [
        "z_bin_id",
        "z_bin_start",
        "z_bin_end",
        "residue_count",
        "positive_count",
        "negative_count",
        "charge_sum",
        "charge_mean",
    ]
    show_cols = [col for col in show_cols if col in profile.columns]
    print(
        profile.sort_values(
            ["positive_count", "charge_sum"],
            ascending=[False, False],
        )[show_cols]
        .head(8)
        .to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    zbin_count = int(cfg.get("profile", {}).get("zbin_count", 20))
    use_inner_only = bool(cfg.get("profile", {}).get("use_inner_only", True))

    input_file = infer_input_file(cfg)
    output_file = infer_output_file(cfg)

    print("=" * 80)
    print("07_build_zbin_profile.py")
    print("=" * 80)
    print(f"Input residue features: {input_file}")
    print(f"Output z-bin profile  : {output_file}")
    print(f"zbin_count            : {zbin_count}")
    print(f"use_inner_only        : {use_inner_only}")

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input residue feature table not found: {input_file}\n"
            f"Run 04_add_inner_residue_features.py first."
        )

    df = pd.read_csv(input_file)

    # Avoid pandas NaN from empty insertion code / mutation label.
    if "insertion_code" in df.columns:
        df["insertion_code"] = df["insertion_code"].fillna("")
    if "mutation_label" in df.columns:
        df["mutation_label"] = df["mutation_label"].fillna("")

    check_required_columns(df)

    original_count = len(df)

    if use_inner_only:
        df = df[df["is_inner_candidate"] == True].copy()

    if len(df) == 0:
        raise RuntimeError(
            "No residues left for z-bin profile. "
            "Check is_inner_candidate or set profile.use_inner_only=false."
        )

    df = assign_z_bins(df, zbin_count=zbin_count)
    profile = aggregate_zbin_profile(df, cfg=cfg, zbin_count=zbin_count)

    profile.to_csv(output_file, index=False)

    print(f"\n[OK] Z-bin profile saved: {output_file}")
    print(f"[INFO] Original residue count: {original_count}")
    print(f"[INFO] Used residue count    : {len(df)}")

    print_profile_summary(profile)


if __name__ == "__main__":
    main()

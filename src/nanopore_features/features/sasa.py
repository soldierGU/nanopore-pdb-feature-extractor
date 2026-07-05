from __future__ import annotations

from typing import Iterable

import pandas as pd


# Maximum solvent-accessible surface area in Angstrom^2.
# Values match the Tien et al. scale used by Biopython DSSP by default.
MAX_ASA_TIEN = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "Q": 225.0,
    "E": 223.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
}

SASA_KEY_COLUMNS = ["chain_id", "residue_number", "insertion_code"]
SASA_FEATURE_COLUMNS = [
    "dssp_aa",
    "sasa_rel",
    "sasa_abs",
    "sasa_max_tien",
    "sasa_buried_fraction",
    "sasa_exposure_class",
    "is_surface_exposed",
    "is_buried",
    "has_sasa",
]


def normalize_insertion_code(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def validate_required_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def classify_sasa_exposure(
    sasa_rel: pd.Series,
    buried_threshold: float = 0.05,
    exposed_threshold: float = 0.25,
) -> pd.Series:
    if not 0.0 <= buried_threshold <= exposed_threshold <= 1.0:
        raise ValueError(
            "Expected thresholds to satisfy "
            "0.0 <= buried_threshold <= exposed_threshold <= 1.0."
        )

    exposure = pd.Series("intermediate", index=sasa_rel.index, dtype="object")
    exposure = exposure.mask(sasa_rel <= buried_threshold, "buried")
    exposure = exposure.mask(sasa_rel >= exposed_threshold, "exposed")
    exposure = exposure.mask(sasa_rel.isna(), "unknown")
    return exposure


def build_sasa_dataframe(
    dssp_rows: pd.DataFrame,
    buried_threshold: float = 0.05,
    exposed_threshold: float = 0.25,
) -> pd.DataFrame:
    validate_required_columns(dssp_rows, SASA_KEY_COLUMNS + ["dssp_aa"])

    sasa_df = dssp_rows.copy()
    sasa_df["insertion_code"] = normalize_insertion_code(sasa_df["insertion_code"])
    sasa_df["dssp_aa"] = sasa_df["dssp_aa"].fillna("X").astype(str).str.upper()
    sasa_df["sasa_max_tien"] = sasa_df["dssp_aa"].map(MAX_ASA_TIEN)

    has_rel = "sasa_rel" in sasa_df.columns
    has_abs = "sasa_abs" in sasa_df.columns
    if not has_rel and not has_abs:
        raise ValueError("SASA input must contain either 'sasa_rel' or 'sasa_abs'.")

    if has_rel:
        sasa_df["sasa_rel"] = pd.to_numeric(sasa_df["sasa_rel"], errors="coerce").clip(0.0, 1.0)
    if has_abs:
        sasa_df["sasa_abs"] = pd.to_numeric(sasa_df["sasa_abs"], errors="coerce")

    if not has_rel:
        sasa_df["sasa_rel"] = (sasa_df["sasa_abs"] / sasa_df["sasa_max_tien"]).clip(0.0, 1.0)
    if not has_abs:
        sasa_df["sasa_abs"] = sasa_df["sasa_rel"] * sasa_df["sasa_max_tien"]

    sasa_df["sasa_buried_fraction"] = 1.0 - sasa_df["sasa_rel"]
    sasa_df["sasa_exposure_class"] = classify_sasa_exposure(
        sasa_df["sasa_rel"],
        buried_threshold=buried_threshold,
        exposed_threshold=exposed_threshold,
    )
    sasa_df["is_surface_exposed"] = sasa_df["sasa_rel"] >= exposed_threshold
    sasa_df["is_buried"] = sasa_df["sasa_rel"] <= buried_threshold
    sasa_df["has_sasa"] = sasa_df["sasa_rel"].notna()

    return sasa_df[
        SASA_KEY_COLUMNS
        + SASA_FEATURE_COLUMNS
    ]


def add_sasa_features(
    residue_df: pd.DataFrame,
    sasa_df: pd.DataFrame,
    buried_threshold: float = 0.05,
    exposed_threshold: float = 0.25,
) -> pd.DataFrame:
    validate_required_columns(residue_df, SASA_KEY_COLUMNS)

    df = residue_df.copy()
    df["insertion_code"] = normalize_insertion_code(df["insertion_code"])

    if "sasa_rel" not in sasa_df.columns or "sasa_abs" not in sasa_df.columns:
        sasa_df = build_sasa_dataframe(
            sasa_df,
            buried_threshold=buried_threshold,
            exposed_threshold=exposed_threshold,
        )
    else:
        sasa_df = sasa_df.copy()
        sasa_df["insertion_code"] = normalize_insertion_code(sasa_df["insertion_code"])

    df = df.drop(columns=[col for col in SASA_FEATURE_COLUMNS if col in df.columns])

    merged = df.merge(
        sasa_df,
        on=SASA_KEY_COLUMNS,
        how="left",
    )

    merged["has_sasa"] = merged["has_sasa"].fillna(False).astype(bool)
    merged["is_surface_exposed"] = merged["is_surface_exposed"].fillna(False).astype(bool)
    merged["is_buried"] = merged["is_buried"].fillna(False).astype(bool)
    merged["sasa_exposure_class"] = merged["sasa_exposure_class"].fillna("unknown")

    return merged


def extract_sasa_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    identity_cols = [
        "pdb_id",
        "nanopore_id",
        "chain_id",
        "residue_number",
        "insertion_code",
        "residue_name",
        "one_letter",
        "is_mutation_site",
        "mutation_label",
        "z_norm",
        "radial_distance",
        "is_inner_candidate",
    ]
    output_cols = [col for col in identity_cols + SASA_FEATURE_COLUMNS if col in df.columns]
    return df[output_cols].copy()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_phenolics_raw.py

Purpose
-------
Clean the raw phenolics absorbance spreadsheet and produce:
1) a cleaned wide table
2) a cleaned long table
3) a QC summary table

Designed for the uploaded file:
    Phenolics_RawData.xlsx
Sheet name:
    Absorbance Data

Key rules
---------
- Standard samples are identified ONLY by Sample ID matching:
      Std0, Std50, std100, std150, std250, std500, std750
  case-insensitive.
- Entries such as STD1 / STD2 / STD3 / STDC1 are NOT treated as standards.
- For standards, the true concentration is taken from Tube#.
- The script does not edit the raw Excel file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SHEET = "Absorbance Data"
EXPECTED_COLUMNS = [
    "Date",
    "Sample ID",
    "Tube#",
    "Rep1 Absorbance 765 nm",
    "Rep2 Absorbance 765 nm",
    "Rep3 Absorbance 765 nm",
]

STANDARD_REGEX = re.compile(r"^std(0|50|100|150|250|500|750)$", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw phenolics absorbance data.")
    parser.add_argument(
        "--input",
        type=str,
        default="Phenolics_RawData.xlsx",
        help="Path to input Excel file.",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=EXPECTED_SHEET,
        help="Worksheet name to read.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory for output CSV files.",
    )
    return parser.parse_args()


def is_standard_sample(sample_id: str) -> bool:
    """
    True standard IDs are only:
      Std0, Std50, std100, std150, std250, std500, std750
    case-insensitive.
    """
    if pd.isna(sample_id):
        return False
    return STANDARD_REGEX.fullmatch(str(sample_id).strip()) is not None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the original six columns but also add normalized internal names.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    out = df.copy()

    out = out.rename(
        columns={
            "Date": "date",
            "Sample ID": "sample_id",
            "Tube#": "tube_no",
            "Rep1 Absorbance 765 nm": "absorbance_rep1",
            "Rep2 Absorbance 765 nm": "absorbance_rep2",
            "Rep3 Absorbance 765 nm": "absorbance_rep3",
        }
    )
    return out


def clean_wide(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Standardize core fields
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["sample_id"] = out["sample_id"].astype(str).str.strip()
    out["tube_no"] = pd.to_numeric(out["tube_no"], errors="coerce")

    for col in ["absorbance_rep1", "absorbance_rep2", "absorbance_rep3"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Identify standards robustly
    out["is_standard"] = out["sample_id"].apply(is_standard_sample)

    # True concentration is only defined for standards
    out["std_conc"] = np.where(out["is_standard"], out["tube_no"], np.nan)

    # Keep a note for unusual IDs that begin with STD but are not true standards
    out["std_like_but_not_standard"] = (
        out["sample_id"].str.upper().str.startswith("STD") & (~out["is_standard"])
    )

    # Row-level QC flags
    out["date_missing"] = out["date"].isna()
    out["tube_no_missing"] = out["tube_no"].isna()

    rep_cols = ["absorbance_rep1", "absorbance_rep2", "absorbance_rep3"]
    out["n_missing_absorbance"] = out[rep_cols].isna().sum(axis=1)
    out["all_absorbance_missing"] = out["n_missing_absorbance"] == len(rep_cols)

    # Optional sanity range check only for flagging, not dropping
    for col in rep_cols:
        out[f"{col}_out_of_range"] = (~out[col].isna()) & ((out[col] < -0.1) | (out[col] > 5))

    out["any_absorbance_out_of_range"] = out[
        [f"{c}_out_of_range" for c in rep_cols]
    ].any(axis=1)

    # Preserve original order
    out["row_id"] = np.arange(1, len(out) + 1)

    return out


def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    rep_map = {
        "absorbance_rep1": 1,
        "absorbance_rep2": 2,
        "absorbance_rep3": 3,
    }

    long_df = df_wide.melt(
        id_vars=[
            "row_id",
            "date",
            "sample_id",
            "tube_no",
            "is_standard",
            "std_conc",
            "std_like_but_not_standard",
            "date_missing",
            "tube_no_missing",
            "n_missing_absorbance",
            "all_absorbance_missing",
            "any_absorbance_out_of_range",
        ],
        value_vars=list(rep_map.keys()),
        var_name="lab_rep_name",
        value_name="absorbance",
    )

    long_df["lab_rep"] = long_df["lab_rep_name"].map(rep_map).astype("Int64")
    long_df["absorbance_missing"] = long_df["absorbance"].isna()
    long_df["absorbance_out_of_range"] = (~long_df["absorbance"].isna()) & (
        (long_df["absorbance"] < -0.1) | (long_df["absorbance"] > 5)
    )

    # Useful text label for standards
    long_df["standard_label"] = np.where(
        long_df["is_standard"],
        long_df["sample_id"].str.lower(),
        pd.NA,
    )

    # Reorder columns
    ordered_cols = [
        "row_id",
        "date",
        "sample_id",
        "tube_no",
        "is_standard",
        "std_conc",
        "standard_label",
        "std_like_but_not_standard",
        "lab_rep",
        "lab_rep_name",
        "absorbance",
        "absorbance_missing",
        "absorbance_out_of_range",
        "date_missing",
        "tube_no_missing",
        "n_missing_absorbance",
        "all_absorbance_missing",
        "any_absorbance_out_of_range",
    ]
    return long_df[ordered_cols].sort_values(["row_id", "lab_rep"]).reset_index(drop=True)


def build_qc(df_wide: pd.DataFrame, df_long: pd.DataFrame) -> pd.DataFrame:
    qc_rows = []

    qc_rows.append(
        {"metric": "n_rows_total", "value": int(len(df_wide))}
    )
    qc_rows.append(
        {"metric": "n_standard_rows", "value": int(df_wide["is_standard"].sum())}
    )
    qc_rows.append(
        {"metric": "n_nonstandard_rows", "value": int((~df_wide["is_standard"]).sum())}
    )
    qc_rows.append(
        {
            "metric": "n_std_like_but_not_standard_rows",
            "value": int(df_wide["std_like_but_not_standard"].sum()),
        }
    )
    qc_rows.append(
        {"metric": "n_missing_date_rows", "value": int(df_wide["date_missing"].sum())}
    )
    qc_rows.append(
        {"metric": "n_missing_tube_rows", "value": int(df_wide["tube_no_missing"].sum())}
    )
    qc_rows.append(
        {
            "metric": "n_all_absorbance_missing_rows",
            "value": int(df_wide["all_absorbance_missing"].sum()),
        }
    )
    qc_rows.append(
        {
            "metric": "n_any_absorbance_out_of_range_rows",
            "value": int(df_wide["any_absorbance_out_of_range"].sum()),
        }
    )
    qc_rows.append(
        {"metric": "n_long_rows_total", "value": int(len(df_long))}
    )
    qc_rows.append(
        {
            "metric": "n_long_missing_absorbance",
            "value": int(df_long["absorbance_missing"].sum()),
        }
    )
    qc_rows.append(
        {
            "metric": "n_long_out_of_range_absorbance",
            "value": int(df_long["absorbance_out_of_range"].sum()),
        }
    )

    # Count standards by nominal concentration
    std_counts = (
        df_wide.loc[df_wide["is_standard"], "std_conc"]
        .value_counts(dropna=False)
        .sort_index()
    )
    for conc, count in std_counts.items():
        qc_rows.append(
            {"metric": f"n_standard_rows_conc_{int(conc)}", "value": int(count)}
        )

    # Dates
    date_counts = df_wide["date"].dt.strftime("%Y-%m-%d").value_counts(dropna=False).sort_index()
    for dt, count in date_counts.items():
        qc_rows.append({"metric": f"n_rows_date_{dt}", "value": int(count)})

    return pd.DataFrame(qc_rows)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_excel(input_path, sheet_name=args.sheet)
    raw = normalize_columns(raw)
    wide = clean_wide(raw)
    long_df = wide_to_long(wide)
    qc = build_qc(wide, long_df)

    wide_out = outdir / "phenolics_clean_wide.csv"
    long_out = outdir / "phenolics_clean_long.csv"
    qc_out = outdir / "phenolics_clean_qc.csv"

    wide.to_csv(wide_out, index=False)
    long_df.to_csv(long_out, index=False)
    qc.to_csv(qc_out, index=False)

    print(f"[OK] Wrote: {wide_out}")
    print(f"[OK] Wrote: {long_out}")
    print(f"[OK] Wrote: {qc_out}")


if __name__ == "__main__":
    main()

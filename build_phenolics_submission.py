#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_phenolics_submission.py

Step 3 of phenolics pipeline.

Purpose
-------
Build the required submission file:
    Phenolics_concentrations.csv

Inputs
------
1) Phenolics_RawData.xlsx
2) phenolics_concentration_long.csv

Output
------
Phenolics_concentrations.csv

Rules
-----
- Keep the same wide-table structure as the original phenolics sheet.
- Remove all standard rows.
- Replace the 3 absorbance columns with:
    concentration_rep1
    concentration_rep2
    concentration_rep3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


EXPECTED_SHEET = "Absorbance Data"
STANDARD_REGEX = re.compile(r"^std(0|50|100|150|250|500|750)$", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phenolics_concentrations.csv")
    parser.add_argument(
        "--raw",
        type=str,
        default="Phenolics_RawData.xlsx",
        help="Path to raw phenolics Excel file",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=EXPECTED_SHEET,
        help="Worksheet name in the raw Excel file",
    )
    parser.add_argument(
        "--conc",
        type=str,
        default="phenolics_concentration_long.csv",
        help="Path to Step 2 output phenolics_concentration_long.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory",
    )
    return parser.parse_args()


def is_standard_sample(sample_id: str) -> bool:
    if pd.isna(sample_id):
        return False
    return STANDARD_REGEX.fullmatch(str(sample_id).strip()) is not None


def load_raw_phenolics(raw_path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(raw_path, sheet_name=sheet)

    expected_cols = [
        "Date",
        "Sample ID",
        "Tube#",
        "Rep1 Absorbance 765 nm",
        "Rep2 Absorbance 765 nm",
        "Rep3 Absorbance 765 nm",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Raw file missing expected columns: {missing}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sample ID"] = df["Sample ID"].astype(str).str.strip()
    df["Tube#"] = pd.to_numeric(df["Tube#"], errors="coerce")
    df["is_standard"] = df["Sample ID"].apply(is_standard_sample)
    return df


def build_concentration_wide(conc_long: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "date",
        "sample_id",
        "tube_no",
        "lab_rep",
        "conc_undiluted",
        "is_standard",
    ]
    missing = [c for c in needed if c not in conc_long.columns]
    if missing:
        raise ValueError(f"Concentration file missing expected columns: {missing}")

    df = conc_long.copy()

    # only non-standard rows go into the submission file
    df = df.loc[~df["is_standard"]].copy()

    # pivot rep1/rep2/rep3 back to wide
    wide = df.pivot_table(
        index=["date", "sample_id", "tube_no"],
        columns="lab_rep",
        values="conc_undiluted",
        aggfunc="first",
    ).reset_index()

    # rename columns
    rename_map = {
        1: "concentration_rep1",
        2: "concentration_rep2",
        3: "concentration_rep3",
    }
    wide = wide.rename(columns=rename_map)

    for col in ["concentration_rep1", "concentration_rep2", "concentration_rep3"]:
        if col not in wide.columns:
            wide[col] = pd.NA

    # keep a stable order
    wide = wide[
        [
            "date",
            "sample_id",
            "tube_no",
            "concentration_rep1",
            "concentration_rep2",
            "concentration_rep3",
        ]
    ]

    return wide


def merge_back_to_raw_structure(raw_nonstd: pd.DataFrame, conc_wide: pd.DataFrame) -> pd.DataFrame:
    out = raw_nonstd.merge(
        conc_wide,
        how="left",
        left_on=["Date", "Sample ID", "Tube#"],
        right_on=["date", "sample_id", "tube_no"],
        validate="one_to_one",
    )

    # drop helper keys from merge
    drop_cols = [c for c in ["date", "sample_id", "tube_no", "is_standard"] if c in out.columns]
    out = out.drop(columns=drop_cols)

    # remove original absorbance columns
    absorbance_cols = [
        "Rep1 Absorbance 765 nm",
        "Rep2 Absorbance 765 nm",
        "Rep3 Absorbance 765 nm",
    ]
    out = out.drop(columns=absorbance_cols)

    # final required structure:
    # same leading columns, but last 3 columns are concentration columns
    final_cols = [
        "Date",
        "Sample ID",
        "Tube#",
        "concentration_rep1",
        "concentration_rep2",
        "concentration_rep3",
    ]
    out = out[final_cols]

    return out


def main() -> None:
    args = parse_args()

    raw_path = Path(args.raw)
    conc_path = Path(args.conc)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    if not conc_path.exists():
        raise FileNotFoundError(f"Concentration file not found: {conc_path}")

    raw = load_raw_phenolics(raw_path, args.sheet)
    raw_nonstd = raw.loc[~raw["is_standard"]].copy()

    conc_long = pd.read_csv(conc_path)
    # align types with raw keys
    conc_long["date"] = pd.to_datetime(conc_long["date"], errors="coerce")
    conc_long["sample_id"] = conc_long["sample_id"].astype(str).str.strip()
    conc_long["tube_no"] = pd.to_numeric(conc_long["tube_no"], errors="coerce")

    conc_wide = build_concentration_wide(conc_long)
    submission = merge_back_to_raw_structure(raw_nonstd, conc_wide)

    out_file = outdir / "Phenolics_concentrations.csv"
    submission.to_csv(out_file, index=False)

    print(f"[OK] Wrote: {out_file}")
    print(f"[INFO] Rows in submission: {len(submission)}")


if __name__ == "__main__":
    main()

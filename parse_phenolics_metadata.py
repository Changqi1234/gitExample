#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parse_phenolics_metadata.py

Step 4 of phenolics pipeline.

Purpose
-------
Parse experiment/treatment/control metadata from Sample ID and produce
an analysis-ready table.

Input
-----
phenolics_concentration_long.csv

Output
------
phenolics_analysis_ready.csv
phenolics_metadata_qc.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


STANDARD_REGEX = re.compile(r"^std(0|50|100|150|250|500|750)$", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse metadata from phenolics sample IDs.")
    parser.add_argument(
        "--input",
        type=str,
        default="phenolics_concentration_long.csv",
        help="Path to phenolics_concentration_long.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory",
    )
    return parser.parse_args()


def is_standard_id(sample_id: str) -> bool:
    if pd.isna(sample_id):
        return False
    return STANDARD_REGEX.fullmatch(str(sample_id).strip()) is not None


def parse_sample_id(sample_id: str) -> dict:
    """
    Parse Sample ID into structured metadata.

    Expected families of IDs:
    1) Standard rows:
       Std0, Std50, Std100, ..., Std750

    2) Long-term experiment:
       STLT1, STLT2, ..., MQLT16
       -> cultivar = ST or MQ
       -> experiment = long_term
       -> plot_id = 1..16

    3) Acute treatment rows:
       STA1, STB3, MQC2, STD1, ...
       -> cultivar = ST or MQ
       -> treatment = A/B/C/D
       -> bio_rep = integer
       -> is_control = False

    4) Acute control rows:
       STAC1, STBC2, STCD3, MQCC1, ...
       -> cultivar = ST or MQ
       -> treatment = base treatment letter A/B/C/D
       -> is_control = True
       -> bio_rep = integer

    Special rule:
       A0 should be retained but flagged with drop_A0 = True.
    """
    result = {
        "sample_id_clean": pd.NA,
        "sample_class": pd.NA,
        "cultivar": pd.NA,
        "experiment": pd.NA,
        "treatment": pd.NA,
        "treatment_group": pd.NA,
        "is_control": pd.NA,
        "bio_rep": pd.NA,
        "plot_id": pd.NA,
        "drop_A0": False,
        "parse_status": "unparsed",
        "parse_note": pd.NA,
    }

    if pd.isna(sample_id):
        result["parse_status"] = "missing_sample_id"
        return result

    s = str(sample_id).strip()
    s_upper = s.upper()
    result["sample_id_clean"] = s_upper

    # 1) Standards
    if is_standard_id(s):
        result.update(
            {
                "sample_class": "standard",
                "experiment": "standard",
                "parse_status": "ok",
                "parse_note": "standard_sample",
            }
        )
        return result

    # 2) Long-term: (ST|MQ)LT<number>
    m = re.fullmatch(r"(ST|MQ)LT(\d+)", s_upper)
    if m:
        cultivar, plot = m.groups()
        result.update(
            {
                "sample_class": "nonstandard",
                "cultivar": cultivar,
                "experiment": "long_term",
                "treatment": "LT",
                "treatment_group": "LT",
                "is_control": pd.NA,
                "bio_rep": pd.NA,
                "plot_id": int(plot),
                "parse_status": "ok",
                "parse_note": "long_term_plot",
            }
        )
        return result

    # 3) Acute control: (ST|MQ)(A|B|C|D)C(\d+)
    # Examples: STAC1, STBC2, STCD3, MQCC1
    m = re.fullmatch(r"(ST|MQ)(A|B|C|D)C(\d+)", s_upper)
    if m:
        cultivar, treatment, rep = m.groups()
        rep = int(rep)
        result.update(
            {
                "sample_class": "nonstandard",
                "cultivar": cultivar,
                "experiment": "acute",
                "treatment": treatment,
                "treatment_group": treatment,
                "is_control": True,
                "bio_rep": rep,
                "plot_id": pd.NA,
                "drop_A0": treatment == "A" and rep == 0,
                "parse_status": "ok",
                "parse_note": "acute_control",
            }
        )
        return result

    # 4) Acute treatment: (ST|MQ)(A|B|C|D)(\d+)
    # Examples: STA1, STB3, MQC2, STD1
    m = re.fullmatch(r"(ST|MQ)(A|B|C|D)(\d+)", s_upper)
    if m:
        cultivar, treatment, rep = m.groups()
        rep = int(rep)
        result.update(
            {
                "sample_class": "nonstandard",
                "cultivar": cultivar,
                "experiment": "acute",
                "treatment": treatment,
                "treatment_group": treatment,
                "is_control": False,
                "bio_rep": rep,
                "plot_id": pd.NA,
                "drop_A0": treatment == "A" and rep == 0,
                "parse_status": "ok",
                "parse_note": "acute_treatment",
            }
        )
        return result

    # fallback
    result["parse_status"] = "unmatched_pattern"
    result["parse_note"] = s_upper
    return result


def build_metadata_qc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({"metric": "n_rows_total", "value": int(len(df))})
    rows.append({"metric": "n_unique_sample_id", "value": int(df["sample_id"].nunique(dropna=True))})

    for col in ["sample_class", "experiment", "cultivar", "treatment", "parse_status"]:
        vc = df[col].astype("string").fillna("NA").value_counts(dropna=False).sort_index()
        for k, v in vc.items():
            rows.append({"metric": f"{col}__{k}", "value": int(v)})

    rows.append({"metric": "n_control_rows", "value": int((df["is_control"] == True).sum())})
    rows.append({"metric": "n_treatment_rows", "value": int((df["is_control"] == False).sum())})
    rows.append({"metric": "n_drop_A0_rows", "value": int(df["drop_A0"].fillna(False).sum())})
    rows.append({"metric": "n_unparsed_rows", "value": int((df["parse_status"] != "ok").sum())})

    id_level = (
        df[["sample_id", "parse_status", "parse_note"]]
        .drop_duplicates()
        .sort_values(["parse_status", "sample_id"])
        .reset_index(drop=True)
    )
    id_level["metric"] = "sample_id_parse"
    id_level["value"] = 1

    qc_main = pd.DataFrame(rows)
    id_level = id_level.rename(columns={"sample_id": "detail"})

    if "detail" not in qc_main.columns:
        qc_main["detail"] = pd.NA
    qc_main["parse_status"] = pd.NA
    qc_main["parse_note"] = pd.NA

    id_level = id_level[["metric", "value", "detail", "parse_status", "parse_note"]]
    qc_main = qc_main[["metric", "value", "detail", "parse_status", "parse_note"]]

    return pd.concat([qc_main, id_level], ignore_index=True)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if "sample_id" not in df.columns:
        raise ValueError("Expected column 'sample_id' not found.")

    df["sample_id"] = df["sample_id"].astype(str).str.strip()

    parsed = df["sample_id"].apply(parse_sample_id).apply(pd.Series)
    out = pd.concat([df, parsed], axis=1)

    out["include_for_submission"] = ~out["sample_class"].eq("standard")
    out["include_for_analysis"] = (~out["sample_class"].eq("standard")) & (~out["drop_A0"].fillna(False))
    out["acute_or_long_term"] = out["experiment"]

    # optional plot/set helpers for long-term
    if "plot_id" in out.columns:
        plot_num = pd.to_numeric(out["plot_id"], errors="coerce")
        lt_mask = out["experiment"].eq("long_term") & plot_num.notna()

        out["lt_set"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out.loc[lt_mask, "lt_set"] = (((plot_num.loc[lt_mask] - 1) // 4) + 1).astype("Int64")
    else:
        out["lt_set"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    qc = build_metadata_qc(out)

    out_file = outdir / "phenolics_analysis_ready.csv"
    qc_file = outdir / "phenolics_metadata_qc.csv"

    out.to_csv(out_file, index=False)
    qc.to_csv(qc_file, index=False)

    print(f"[OK] Wrote: {out_file}")
    print(f"[OK] Wrote: {qc_file}")


if __name__ == "__main__":
    main()

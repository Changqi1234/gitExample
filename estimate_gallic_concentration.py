#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
estimate_gallic_concentration.py

Step 2 of phenolics pipeline.

Input
-----
phenolics_clean_long.csv

Output
------
phenolics_concentration_long.csv
phenolics_calibration_qc.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="phenolics_clean_long.csv",
        help="Clean long phenolics table"
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory"
    )
    return parser.parse_args()


def fit_calibration(group):
    """
    Fit y = m x + b

    y = absorbance
    x = concentration
    """

    std = group[group["is_standard"]]

    if len(std) < 2:
        return None

    x = std["std_conc"].values
    y = std["absorbance"].values

    m, b = np.polyfit(x, y, 1)

    y_hat = m * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return m, b, r2, len(std)


def main():

    args = parse_args()

    df = pd.read_csv(args.input)

    results = []
    qc_rows = []

    for (date, rep), g in df.groupby(["date", "lab_rep"]):

        fit = fit_calibration(g)

        if fit is None:

            qc_rows.append({
                "date": date,
                "lab_rep": rep,
                "n_standard": 0,
                "slope": np.nan,
                "intercept": np.nan,
                "r2": np.nan,
                "status": "no_standard_points"
            })

            continue

        m, b, r2, n_std = fit

        qc_rows.append({
            "date": date,
            "lab_rep": rep,
            "n_standard": n_std,
            "slope": m,
            "intercept": b,
            "r2": r2,
            "status": "ok"
        })

        g = g.copy()

        g["slope"] = m
        g["intercept"] = b

        g["conc_diluted"] = (g["absorbance"] - b) / m

        g["conc_undiluted"] = 10 * g["conc_diluted"]

        results.append(g)

    out = pd.concat(results, ignore_index=True)

    qc = pd.DataFrame(qc_rows)

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    out_file = outdir / "phenolics_concentration_long.csv"
    qc_file = outdir / "phenolics_calibration_qc.csv"

    out.to_csv(out_file, index=False)
    qc.to_csv(qc_file, index=False)

    print("Saved:", out_file)
    print("Saved:", qc_file)


if __name__ == "__main__":
    main()

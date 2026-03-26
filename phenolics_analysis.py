#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze phenolics concentration results.")
    parser.add_argument(
        "--input",
        type=str,
        default="phenolics_analysis_ready.csv",
        help="Path to phenolics_analysis_ready.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory",
    )
    return parser.parse_args()


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "conc_undiluted" not in df.columns:
        raise ValueError("Expected column 'conc_undiluted' not found.")

    df["conc_undiluted"] = pd.to_numeric(df["conc_undiluted"], errors="coerce")

    for col in ["sample_id", "experiment", "cultivar", "treatment_group"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    if "include_for_analysis" in df.columns:
        df["include_for_analysis"] = (
            df["include_for_analysis"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )
    else:
        df["include_for_analysis"] = True

    if "is_control" in df.columns:
        df["is_control"] = (
            df["is_control"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
        )
    else:
        df["is_control"] = pd.NA

    if "plot_id" in df.columns:
        df["plot_id"] = pd.to_numeric(df["plot_id"], errors="coerce")
    else:
        df["plot_id"] = np.nan

    if "lt_set" in df.columns:
        df["lt_set"] = pd.to_numeric(df["lt_set"], errors="coerce")
    else:
        df["lt_set"] = np.nan

    if "lab_rep" in df.columns:
        df["lab_rep"] = pd.to_numeric(df["lab_rep"], errors="coerce")

    return df


def build_qc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({"metric": "n_rows_total", "value": int(len(df))})
    rows.append({
        "metric": "n_rows_included_for_analysis",
        "value": int(df["include_for_analysis"].fillna(False).sum())
    })
    rows.append({
        "metric": "n_unique_sample_id",
        "value": int(df["sample_id"].nunique(dropna=True))
    })
    rows.append({
        "metric": "n_missing_conc_undiluted",
        "value": int(df["conc_undiluted"].isna().sum())
    })

    for col in ["experiment", "cultivar", "treatment_group"]:
        if col in df.columns:
            vc = df[col].astype("string").fillna("NA").value_counts(dropna=False).sort_index()
            for k, v in vc.items():
                rows.append({"metric": f"{col}__{k}", "value": int(v)})

    if "is_control" in df.columns:
        vc = df["is_control"].astype("string").fillna("NA").value_counts(dropna=False).sort_index()
        for k, v in vc.items():
            rows.append({"metric": f"is_control__{k}", "value": int(v)})

    return pd.DataFrame(rows)


def build_sample_level(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "sample_id",
        "sample_id_clean",
        "sample_class",
        "cultivar",
        "experiment",
        "treatment",
        "treatment_group",
        "is_control",
        "bio_rep",
        "plot_id",
        "drop_A0",
        "include_for_analysis",
        "acute_or_long_term",
        "lt_set",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    out = (
        df.groupby(keep_cols, dropna=False)
        .agg(
            n_rep_used=("conc_undiluted", lambda x: int(x.notna().sum())),
            gallic_mean=("conc_undiluted", "mean"),
            gallic_sd=("conc_undiluted", "std"),
            gallic_median=("conc_undiluted", "median"),
            gallic_min=("conc_undiluted", "min"),
            gallic_max=("conc_undiluted", "max"),
        )
        .reset_index()
    )

    return out


def build_experiment_treatment_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    use = sample_df.loc[
        sample_df["include_for_analysis"].fillna(False) & sample_df["gallic_mean"].notna()
    ].copy()

    out = (
        use.groupby(["experiment", "treatment_group", "is_control"], dropna=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            gallic_mean=("gallic_mean", "mean"),
            gallic_sd=("gallic_mean", "std"),
            gallic_median=("gallic_mean", "median"),
            gallic_min=("gallic_mean", "min"),
            gallic_max=("gallic_mean", "max"),
        )
        .reset_index()
    )
    return out


def build_experiment_treatment_cultivar_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    use = sample_df.loc[
        sample_df["include_for_analysis"].fillna(False) & sample_df["gallic_mean"].notna()
    ].copy()

    out = (
        use.groupby(["experiment", "treatment_group", "is_control", "cultivar"], dropna=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            gallic_mean=("gallic_mean", "mean"),
            gallic_sd=("gallic_mean", "std"),
            gallic_median=("gallic_mean", "median"),
        )
        .reset_index()
    )
    return out


def fit_models(sample_df: pd.DataFrame) -> pd.DataFrame:
    if not HAS_STATSMODELS:
        return pd.DataFrame([{
            "model_name": "no_statsmodels",
            "term": "NOT_RUN",
            "coef": np.nan,
            "p_value": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "note": "statsmodels not available"
        }])

    rows = []

    use = sample_df.loc[
        sample_df["include_for_analysis"].fillna(False) & sample_df["gallic_mean"].notna()
    ].copy()

    acute = use.loc[use["experiment"].astype(str) == "acute"].copy()
    if len(acute) >= 5:
        try:
            acute["is_control_str"] = acute["is_control"].astype("string")
            m = smf.ols(
                "gallic_mean ~ C(treatment_group) + C(is_control_str) + C(cultivar)",
                data=acute
            ).fit()
            for term in m.params.index:
                rows.append({
                    "model_name": "acute_main",
                    "term": term,
                    "coef": m.params[term],
                    "p_value": m.pvalues[term],
                    "n_obs": int(m.nobs),
                    "r_squared": m.rsquared,
                    "note": pd.NA,
                })
        except Exception as e:
            rows.append({
                "model_name": "acute_main",
                "term": "MODEL_FAILED",
                "coef": np.nan,
                "p_value": np.nan,
                "n_obs": len(acute),
                "r_squared": np.nan,
                "note": str(e),
            })

    long_df = use.loc[use["experiment"].astype(str) == "long_term"].copy()
    if len(long_df) >= 5:
        try:
            if long_df["lt_set"].notna().sum() > 0:
                m = smf.ols(
                    "gallic_mean ~ C(cultivar) + C(lt_set)",
                    data=long_df
                ).fit()
            else:
                m = smf.ols(
                    "gallic_mean ~ C(cultivar)",
                    data=long_df
                ).fit()
            for term in m.params.index:
                rows.append({
                    "model_name": "long_term_main",
                    "term": term,
                    "coef": m.params[term],
                    "p_value": m.pvalues[term],
                    "n_obs": int(m.nobs),
                    "r_squared": m.rsquared,
                    "note": pd.NA,
                })
        except Exception as e:
            rows.append({
                "model_name": "long_term_main",
                "term": "MODEL_FAILED",
                "coef": np.nan,
                "p_value": np.nan,
                "n_obs": len(long_df),
                "r_squared": np.nan,
                "note": str(e),
            })

    if not rows:
        rows.append({
            "model_name": "no_model_fit",
            "term": "NOT_RUN",
            "coef": np.nan,
            "p_value": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "note": "not enough usable rows",
        })

    return pd.DataFrame(rows)


def safe_boxplot(data_dict, title, ylabel, outfile):
    labels = []
    values = []
    for k, v in data_dict.items():
        arr = pd.Series(v).dropna().values
        if len(arr) > 0:
            labels.append(k)
            values.append(arr)

    if len(values) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(values, tick_labels=labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def make_plots(sample_df: pd.DataFrame, outdir: Path) -> None:
    use = sample_df.loc[
        sample_df["include_for_analysis"].fillna(False) & sample_df["gallic_mean"].notna()
    ].copy()

    if len(use) == 0:
        return

    use["group1"] = (
        use["experiment"].astype(str) + "_"
        + use["treatment_group"].astype(str) + "_"
        + np.where(use["is_control"] == True, "control", "treated")
    )
    data1 = {g: sub["gallic_mean"] for g, sub in use.groupby("group1", dropna=False)}
    safe_boxplot(
        data1,
        "Gallic acid concentration by experiment / treatment / control",
        "Gallic acid concentration",
        outdir / "plot1_gallic_by_experiment_treatment.png"
    )

    use["group2"] = (
        use["experiment"].astype(str) + "_"
        + use["treatment_group"].astype(str) + "_"
        + use["cultivar"].astype(str)
    )
    data2 = {g: sub["gallic_mean"] for g, sub in use.groupby("group2", dropna=False)}
    safe_boxplot(
        data2,
        "Gallic acid concentration by treatment and cultivar",
        "Gallic acid concentration",
        outdir / "plot2_gallic_by_treatment_cultivar.png"
    )

    rep_df = use.loc[:, ["sample_id", "gallic_mean", "gallic_sd", "n_rep_used"]].drop_duplicates()
    if len(rep_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(rep_df["gallic_mean"], rep_df["gallic_sd"])
        ax.set_title("Replicate consistency: sample mean vs sample SD")
        ax.set_xlabel("Sample mean gallic acid concentration")
        ax.set_ylabel("Sample SD across lab replicates")
        plt.tight_layout()
        plt.savefig(outdir / "plot3_replicate_consistency.png", dpi=200)
        plt.close()

    long_df = use.loc[(use["experiment"].astype(str) == "long_term") & use["plot_id"].notna()].copy()
    if len(long_df) > 0:
        long_df = long_df.sort_values("plot_id")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(long_df["plot_id"], long_df["gallic_mean"], marker="o")
        ax.set_title("Long-term experiment: gallic acid concentration by plot")
        ax.set_xlabel("Plot ID")
        ax.set_ylabel("Gallic acid concentration")
        plt.tight_layout()
        plt.savefig(outdir / "plot4_longterm_by_plot.png", dpi=200)
        plt.close()


def main():
    args = parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df = normalize_df(df)

    qc = build_qc(df)
    sample_df = build_sample_level(df)
    exp_treat = build_experiment_treatment_summary(sample_df)
    exp_treat_cultivar = build_experiment_treatment_cultivar_summary(sample_df)
    model_results = fit_models(sample_df)

    qc.to_csv(outdir / "phenolics_qc_overall.csv", index=False)
    sample_df.to_csv(outdir / "gallic_by_sample.csv", index=False)
    exp_treat.to_csv(outdir / "gallic_by_experiment_treatment.csv", index=False)
    exp_treat_cultivar.to_csv(outdir / "gallic_by_experiment_treatment_cultivar.csv", index=False)
    model_results.to_csv(outdir / "model_results.csv", index=False)

    make_plots(sample_df, outdir)

    print(f"[OK] Wrote: {outdir / 'phenolics_qc_overall.csv'}")
    print(f"[OK] Wrote: {outdir / 'gallic_by_sample.csv'}")
    print(f"[OK] Wrote: {outdir / 'gallic_by_experiment_treatment.csv'}")
    print(f"[OK] Wrote: {outdir / 'gallic_by_experiment_treatment_cultivar.csv'}")
    print(f"[OK] Wrote: {outdir / 'model_results.csv'}")
    print("[OK] Wrote plot PNG files")


if __name__ == "__main__":
    main()

"""Retain complete curves and create descriptive tables for the development and external files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mlk_common import (
    COW_COL, DIM_COLS, K_VALUES, MILK_COLS, PARITY_COL, curve_descriptors, ensure_dir,
    read_csv, retain_complete_curves, retention_flow, target_counts,
)


def parity_counts(df: pd.DataFrame) -> dict:
    p = df[PARITY_COL].copy()
    return {
        "parity_1": int((p == 1).sum()),
        "parity_2": int((p == 2).sum()),
        "parity_3": int((p == 3).sum()),
        "parity_ge4": int((p >= 4).sum()),
    }


def dataset_summary(df: pd.DataFrame, label: str) -> dict:
    desc = curve_descriptors(df)
    row = {
        "dataset": label,
        "curves": int(len(df)),
        "unique_cows": int(df[COW_COL].nunique()),
        **parity_counts(df),
        "record_1_milk_kg_d": float(df["Milk_recorod_1"].mean()),
        "record_3_milk_kg_d": float(df["Milk_recorod_3"].mean()),
        "record_9_milk_kg_d": float(df["Milk_recorod_9"].mean()),
        "record_1_DIM_d": float(df["Days_in_milk_1"].mean()),
        "record_9_DIM_d": float(df["Days_in_milk_9"].mean()),
        "peak_milk_kg_d_mean": float(desc["peak_milk_kg_d"].mean()),
        "peak_milk_kg_d_sd": float(desc["peak_milk_kg_d"].std(ddof=1)),
        "persistency_mean": float(desc["persistency_record9_over_peak"].mean()),
        "persistency_sd": float(desc["persistency_record9_over_peak"].std(ddof=1)),
        "approx_305d_yield_kg_mean": float(desc["approx_305d_testday_yield_kg"].mean()),
        "approx_305d_yield_kg_sd": float(desc["approx_305d_testday_yield_kg"].std(ddof=1)),
    }
    for k, n in target_counts(len(df)).items():
        row[f"target_records_k{k}"] = n
    return row


def repeated_curve_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(COW_COL).size().value_counts().sort_index()
    return pd.DataFrame({"curves_per_cow": counts.index.astype(int), "n_cows": counts.values.astype(int)})


def run(development: str, external: str, outdir: str) -> None:
    outdir = ensure_dir(outdir)
    raw_dev = read_csv(development)
    raw_ext = read_csv(external)

    flow_dev = retention_flow(raw_dev, require_increasing_dim=False)
    flow_ext = retention_flow(raw_ext, require_increasing_dim=True)
    flow_dev.to_csv(outdir / "development_retention_flow.csv", index=False)
    flow_ext.to_csv(outdir / "external_retention_flow.csv", index=False)

    dev = retain_complete_curves(raw_dev, require_increasing_dim=False)
    ext = retain_complete_curves(raw_ext, require_increasing_dim=True)
    dev.to_csv(outdir / "development_retained_complete_curves.csv", index=False)
    ext.to_csv(outdir / "external_retained_complete_curves.csv", index=False)

    curve_descriptors(dev).to_csv(outdir / "development_curve_descriptors.csv", index=False)
    curve_descriptors(ext).to_csv(outdir / "external_curve_descriptors.csv", index=False)
    repeated_curve_summary(dev).to_csv(outdir / "development_repeated_curves_per_cow.csv", index=False)
    repeated_curve_summary(ext).to_csv(outdir / "external_repeated_curves_per_cow.csv", index=False)

    summary = pd.DataFrame([dataset_summary(dev, "development"), dataset_summary(ext, "external_validation")])
    summary.to_csv(outdir / "dataset_structure_summary.csv", index=False)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True)
    parser.add_argument("--external", required=True)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    run(args.development, args.external, args.outdir)


if __name__ == "__main__":
    main()

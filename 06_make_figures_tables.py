"""Generate manuscript-style derived tables and figures from workflow outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlk_common import ensure_dir, K_VALUES, MILK_COLS, DIM_COLS


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def make_internal_mae_plot(outdir: Path) -> None:
    metrics_path = outdir / "internal_pooled_metrics_core_models.csv"
    if not metrics_path.exists():
        return
    m = pd.read_csv(metrics_path)
    show = ["Empirical-Bayes prior", "Prefix residual prior", "Bayesian Ridge", "Ridge", "Lasso", "Direct LightGBM"]
    plot = m[m["model"].isin(show)].copy()
    plt.figure(figsize=(7.2, 4.6))
    for model, g in plot.groupby("model"):
        g = g.sort_values("k")
        plt.plot(g["k"], g["MAE"], marker="o", label=model)
    plt.xlabel("Observed current-lactation records (k)")
    plt.ylabel("Internal future-record MAE, kg/d")
    plt.xticks(K_VALUES)
    plt.legend(fontsize=8)
    plt.title("Internal validation: residual learning improves future-record MAE")
    savefig(outdir / "figure_internal_core_model_mae.png")


def make_external_mae_plot(outdir: Path) -> None:
    metrics_path = outdir / "external_metrics_core_models.csv"
    if not metrics_path.exists():
        return
    m = pd.read_csv(metrics_path)
    m = m[m["validation"].eq("external_all_cows")]
    show = ["Empirical-Bayes prior", "Bayesian Ridge", "Ridge", "Prefix residual prior"]
    plot = m[m["model"].isin(show)].copy()
    plt.figure(figsize=(7.2, 4.6))
    for model, g in plot.groupby("model"):
        g = g.sort_values("k")
        plt.plot(g["k"], g["MAE"], marker="o", label=model)
    plt.xlabel("Observed current-lactation records (k)")
    plt.ylabel("External future-record MAE, kg/d")
    plt.xticks(K_VALUES)
    plt.legend(fontsize=8)
    plt.title("External validation under distribution shift")
    savefig(outdir / "figure_external_core_model_mae.png")


def make_covariate_plot(outdir: Path) -> None:
    metrics_path = outdir / "internal_pooled_metrics_covariate_enrichment.csv"
    if not metrics_path.exists():
        return
    m = pd.read_csv(metrics_path)
    plt.figure(figsize=(7.2, 4.6))
    for model, g in m.groupby("model"):
        g = g.sort_values("k")
        plt.plot(g["k"], g["MAE"], marker="o", label=model)
    plt.xlabel("Observed current-lactation records (k)")
    plt.ylabel("MAE, kg/d")
    plt.xticks(K_VALUES)
    plt.legend(fontsize=7)
    plt.title("Ancillary covariate enrichment")
    savefig(outdir / "figure_covariate_enrichment_mae.png")


def make_dataset_shift_plot(outdir: Path) -> None:
    dev_path = outdir / "development_retained_complete_curves.csv"
    ext_path = outdir / "external_retained_complete_curves.csv"
    if not (dev_path.exists() and ext_path.exists()):
        return
    dev = pd.read_csv(dev_path)
    ext = pd.read_csv(ext_path)
    recs = np.arange(1, 10)
    dev_mean = [dev[f"Milk_recorod_{i}"].mean() for i in recs]
    ext_mean = [ext[f"Milk_recorod_{i}"].mean() for i in recs]
    dim_dev = [dev[f"Days_in_milk_{i}"].mean() for i in recs]
    dim_ext = [ext[f"Days_in_milk_{i}"].mean() for i in recs]
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(dim_dev, dev_mean, marker="o", label=f"Development (n = {len(dev):,})")
    plt.plot(dim_ext, ext_mean, marker="s", label=f"External (n = {len(ext):,})")
    plt.xlabel("DIM, d")
    plt.ylabel("Mean milk yield, kg/d")
    plt.legend(fontsize=8)
    plt.title("Development versus external validation mean curves")
    savefig(outdir / "figure_dataset_shift_mean_curves.png")


def make_table4_like(outdir: Path) -> None:
    internal = outdir / "internal_pooled_metrics_core_models.csv"
    external = outdir / "external_metrics_core_models.csv"
    if not (internal.exists() and external.exists()):
        return
    i = pd.read_csv(internal)
    e = pd.read_csv(external)
    rows = []
    for k in K_VALUES:
        row = {"Scenario": f"k = {k}"}
        def val_int(model):
            z = i[(i["k"] == k) & (i["model"] == model)]
            return float(z["MAE"].iloc[0]) if len(z) else np.nan
        def val_ext(model, validation="external_all_cows"):
            z = e[(e["k"] == k) & (e["model"] == model) & (e["validation"] == validation)]
            return float(z["MAE"].iloc[0]) if len(z) else np.nan
        row["Dev. prior"] = val_int("Empirical-Bayes prior")
        row["Dev. Bayesian Ridge"] = val_int("Bayesian Ridge")
        row["Dev. Ridge"] = val_int("Ridge")
        row["Dev. Direct LightGBM"] = val_int("Direct LightGBM")
        row["Dev. improvement vs prior, %"] = 100 * (row["Dev. prior"] - row["Dev. Bayesian Ridge"]) / row["Dev. prior"]
        row["Ext. prior"] = val_ext("Empirical-Bayes prior")
        row["Ext. Bayesian Ridge"] = val_ext("Bayesian Ridge")
        row["Ext. Ridge"] = val_ext("Ridge")
        row["Ext. Bayesian Ridge excluding ID overlaps"] = val_ext("Bayesian Ridge", "external_excluding_numeric_id_overlaps")
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "table_internal_external_core_MAE_summary.csv", index=False)


def run(outdir: str) -> None:
    outdir = ensure_dir(outdir)
    make_internal_mae_plot(outdir)
    make_external_mae_plot(outdir)
    make_covariate_plot(outdir)
    make_dataset_shift_plot(outdir)
    make_table4_like(outdir)
    print(f"Figures and summary tables written to {outdir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    run(args.outdir)


if __name__ == "__main__":
    main()

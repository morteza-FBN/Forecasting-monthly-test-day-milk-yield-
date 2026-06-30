"""External validation of development-trained prior-residual models."""
from __future__ import annotations

import argparse

import pandas as pd

from mlk_common import (
    COW_COL, DEFAULT_LAMBDA_PRIOR, K_VALUES, build_long_targets, cow_cluster_bootstrap,
    ensure_dir, estimate_prior, metric_summary, read_csv, retain_complete_curves,
)
from modeling import add_auc_summary, fit_all_models, summarize_predictions


def run_external_validation(development: str, external: str, outdir: str,
                            lambda_prior: float = DEFAULT_LAMBDA_PRIOR,
                            n_boot: int = 1000) -> None:
    outdir = ensure_dir(outdir)
    dev = retain_complete_curves(read_csv(development), require_increasing_dim=False)
    ext = retain_complete_curves(read_csv(external), require_increasing_dim=True)
    prior = estimate_prior(dev, lambda_prior=lambda_prior)

    pred_rows = []
    for k in K_VALUES:
        train_long = build_long_targets(dev, k=k, prior=prior, feature_set="core")
        ext_long = build_long_targets(ext, k=k, prior=prior, feature_set="core")
        pred = fit_all_models(train_long, ext_long, model_names=["Ridge", "Bayesian Ridge"])
        pred["validation"] = "external_all_cows"
        pred_rows.append(pred)
        print(f"external k={k}, all cows done")

        overlap_ids = sorted(set(dev[COW_COL].astype(int)).intersection(set(ext[COW_COL].astype(int))))
        if overlap_ids:
            ext_no = ext.loc[~ext[COW_COL].astype(int).isin(overlap_ids)].reset_index(drop=True)
            ext_no_long = build_long_targets(ext_no, k=k, prior=prior, feature_set="core")
            pred_no = fit_all_models(train_long, ext_no_long, model_names=["Ridge", "Bayesian Ridge"])
            pred_no["validation"] = "external_excluding_numeric_id_overlaps"
            pred_rows.append(pred_no)
        else:
            print("No numeric ID overlap detected.")

    pred_all = pd.concat(pred_rows, ignore_index=True)
    pred_all.to_csv(outdir / "external_predictions_core_models.csv", index=False)

    metrics = []
    for (validation, k, model), g in pred_all.groupby(["validation", "k", "model"]):
        m = metric_summary(g["y_true"].values, g["y_pred"].values)
        m.update({"validation": validation, "k": k, "model": model})
        metrics.append(m)
    pd.DataFrame(metrics).sort_values(["validation", "k", "MAE"]).to_csv(outdir / "external_metrics_core_models.csv", index=False)

    boot_rows = []
    for (validation, k, model), g in pred_all.groupby(["validation", "k", "model"]):
        b = cow_cluster_bootstrap(g, n_boot=n_boot)
        b.update({"validation": validation, "k": k, "model": model})
        boot_rows.append(b)
    pd.DataFrame(boot_rows).sort_values(["validation", "k", "model"]).to_csv(outdir / "external_bootstrap_mae_ci.csv", index=False)

    auc = add_auc_summary(pred_all[pred_all["validation"].eq("external_all_cows")])
    auc.to_csv(outdir / "external_residual_underperformance_auc.csv", index=False)

    overlap_ids = sorted(set(dev[COW_COL].astype(int)).intersection(set(ext[COW_COL].astype(int))))
    pd.DataFrame({"overlapping_numeric_cow_id": overlap_ids}).to_csv(outdir / "external_numeric_id_overlaps.csv", index=False)
    print(f"Numeric ID overlaps: {len(overlap_ids)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True)
    parser.add_argument("--external", required=True)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--lambda-prior", type=float, default=DEFAULT_LAMBDA_PRIOR)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    run_external_validation(args.development, args.external, args.outdir, args.lambda_prior, args.n_boot)


if __name__ == "__main__":
    main()

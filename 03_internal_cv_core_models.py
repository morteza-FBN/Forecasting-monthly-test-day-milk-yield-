"""Cow-grouped internal cross-validation for core prior-residual forecasting models."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from mlk_common import (
    COW_COL, DEFAULT_LAMBDA_PRIOR, K_VALUES, build_long_targets, describe_versions,
    ensure_dir, estimate_prior, metric_summary, read_csv, retain_complete_curves,
)
from modeling import add_auc_summary, fit_all_models, summarize_predictions


def internal_cv(development: str, outdir: str, lambda_prior: float = DEFAULT_LAMBDA_PRIOR,
                n_splits: int = 3) -> None:
    outdir = ensure_dir(outdir)
    raw = read_csv(development)
    df = retain_complete_curves(raw, require_increasing_dim=False)
    groups = df[COW_COL].values
    cv = GroupKFold(n_splits=n_splits)

    all_preds = []
    fold_rows = []
    fold_assignments = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(df, groups=groups), start=1):
        train = df.iloc[train_idx].reset_index(drop=True)
        valid = df.iloc[valid_idx].reset_index(drop=True)
        fold_assignments.append(pd.DataFrame({
            "curve_id": valid["curve_id"].values,
            "cow_id": valid[COW_COL].values,
            "fold": fold,
        }))
        prior = estimate_prior(train, lambda_prior=lambda_prior)
        for k in K_VALUES:
            train_long = build_long_targets(train, k=k, prior=prior, feature_set="core")
            valid_long = build_long_targets(valid, k=k, prior=prior, feature_set="core")
            pred = fit_all_models(train_long, valid_long)
            pred["fold"] = fold
            pred["lambda_prior"] = lambda_prior
            all_preds.append(pred)
            for model, g in pred.groupby("model"):
                m = metric_summary(g["y_true"].values, g["y_pred"].values)
                m.update({"fold": fold, "k": k, "model": model})
                fold_rows.append(m)
            print(f"fold={fold}, k={k}, done")

    pred_all = pd.concat(all_preds, ignore_index=True)
    pred_all.to_csv(outdir / "internal_oof_predictions_core_models.csv", index=False)
    pd.concat(fold_assignments, ignore_index=True).drop_duplicates().to_csv(outdir / "internal_groupkfold_assignments.csv", index=False)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(outdir / "internal_fold_metrics_core_models.csv", index=False)
    pooled = summarize_predictions(pred_all)
    pooled.to_csv(outdir / "internal_pooled_metrics_core_models.csv", index=False)

    # Mean ± SD across folds for manuscript-style summaries.
    fold_summary = fold_metrics.groupby(["k", "model"], as_index=False).agg(
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        Bias_mean=("Bias", "mean"), n_records=("n", "sum"),
    )
    fold_summary.to_csv(outdir / "internal_fold_mean_sd_core_models.csv", index=False)

    auc = add_auc_summary(pred_all[pred_all["model"].isin(["Bayesian Ridge", "Ridge", "Empirical-Bayes prior"])])
    auc.to_csv(outdir / "internal_residual_underperformance_auc.csv", index=False)
    describe_versions().to_csv(outdir / "python_package_versions.csv", index=False)
    print(pooled.head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--lambda-prior", type=float, default=DEFAULT_LAMBDA_PRIOR)
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args()
    internal_cv(args.development, args.outdir, lambda_prior=args.lambda_prior, n_splits=args.n_splits)


if __name__ == "__main__":
    main()

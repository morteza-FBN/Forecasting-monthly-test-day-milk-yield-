"""Ancillary covariate-enrichment and sensitivity analyses."""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import GroupKFold

from mlk_common import (
    COW_COL, DEFAULT_LAMBDA_PRIOR, DIM_COLS, K_VALUES, build_long_targets,
    ensure_dir, estimate_prior, metric_summary, read_csv, retain_complete_curves,
)
from modeling import fit_all_models, summarize_predictions


def run_covariate_enrichment(development: str, outdir: str, n_splits: int = 3) -> None:
    outdir = ensure_dir(outdir)
    raw = read_csv(development)
    df = retain_complete_curves(raw, require_increasing_dim=False)
    groups = df[COW_COL].values
    cv = GroupKFold(n_splits=n_splits)

    pred_rows = []
    fold_metrics = []
    feature_sets = [
        ("Ridge core", "core"),
        ("Ridge + prefix composition/SCC", "prefix_composition_scc"),
        ("Ridge + BCS/calving set", "background"),
        ("Ridge + all covariates", "all_covariates"),
    ]
    for fold, (train_idx, valid_idx) in enumerate(cv.split(df, groups=groups), start=1):
        train = df.iloc[train_idx].reset_index(drop=True)
        valid = df.iloc[valid_idx].reset_index(drop=True)
        prior = estimate_prior(train, lambda_prior=DEFAULT_LAMBDA_PRIOR)
        for k in K_VALUES:
            for label, feature_set in feature_sets:
                train_long = build_long_targets(train, k=k, prior=prior, feature_set=feature_set)
                valid_long = build_long_targets(valid, k=k, prior=prior, feature_set=feature_set)
                pred = fit_all_models(train_long, valid_long, model_names=["Ridge"])
                pred = pred[pred["model"].eq("Ridge")].copy()
                pred["model"] = label
                pred["feature_set"] = feature_set
                pred["fold"] = fold
                pred_rows.append(pred)
                m = metric_summary(pred["y_true"].values, pred["y_pred"].values)
                m.update({"fold": fold, "k": k, "model": label, "analysis": "covariate_enrichment"})
                fold_metrics.append(m)
                print(f"covariates fold={fold}, k={k}, {label}")
    pred_all = pd.concat(pred_rows, ignore_index=True)
    pred_all.to_csv(outdir / "internal_oof_predictions_covariate_enrichment.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(outdir / "internal_fold_metrics_covariate_enrichment.csv", index=False)
    summarize_predictions(pred_all).to_csv(outdir / "internal_pooled_metrics_covariate_enrichment.csv", index=False)


def run_sensitivity(development: str, outdir: str, n_splits: int = 3) -> None:
    outdir = ensure_dir(outdir)
    raw = read_csv(development)
    df = retain_complete_curves(raw, require_increasing_dim=False)
    groups = df[COW_COL].values
    cv = GroupKFold(n_splits=n_splits)
    pred_rows = []
    fold_metrics = []

    sensitivity_specs = []
    sensitivity_specs.append({"analysis": "actual_future_DIM", "lambda_prior": 30.0, "target_dim_mode": "actual", "parity_collapse_ge": None})
    sensitivity_specs.append({"analysis": "record_mean_future_DIM", "lambda_prior": 30.0, "target_dim_mode": "record_mean", "parity_collapse_ge": None})
    sensitivity_specs.append({"analysis": "parity_ge5_collapsed", "lambda_prior": 30.0, "target_dim_mode": "actual", "parity_collapse_ge": 5})
    for lam in [10.0, 30.0, 60.0, 100.0]:
        sensitivity_specs.append({"analysis": f"lambda_{int(lam)}", "lambda_prior": lam, "target_dim_mode": "actual", "parity_collapse_ge": None})

    for fold, (train_idx, valid_idx) in enumerate(cv.split(df, groups=groups), start=1):
        train = df.iloc[train_idx].reset_index(drop=True)
        valid = df.iloc[valid_idx].reset_index(drop=True)
        train_record_mean_dim = {m: float(train[f"Days_in_milk_{m}"].mean()) for m in range(1, 10)}
        for spec in sensitivity_specs:
            prior = estimate_prior(train, lambda_prior=spec["lambda_prior"], parity_collapse_ge=spec["parity_collapse_ge"])
            for k in K_VALUES:
                train_long = build_long_targets(train, k=k, prior=prior,
                                                target_dim_mode=spec["target_dim_mode"],
                                                train_record_mean_dim=train_record_mean_dim,
                                                feature_set="core",
                                                parity_collapse_ge=spec["parity_collapse_ge"])
                valid_long = build_long_targets(valid, k=k, prior=prior,
                                                target_dim_mode=spec["target_dim_mode"],
                                                train_record_mean_dim=train_record_mean_dim,
                                                feature_set="core",
                                                parity_collapse_ge=spec["parity_collapse_ge"])
                pred = fit_all_models(train_long, valid_long, model_names=["Bayesian Ridge"])
                pred = pred[pred["model"].isin(["Empirical-Bayes prior", "Bayesian Ridge"])].copy()
                pred["analysis"] = spec["analysis"]
                pred["lambda_prior"] = spec["lambda_prior"]
                pred["fold"] = fold
                pred_rows.append(pred)
                for model, g in pred.groupby("model"):
                    m = metric_summary(g["y_true"].values, g["y_pred"].values)
                    m.update({"fold": fold, "k": k, "model": model, "analysis": spec["analysis"], "lambda_prior": spec["lambda_prior"]})
                    fold_metrics.append(m)
                print(f"sensitivity fold={fold}, k={k}, {spec['analysis']}")
    pred_all = pd.concat(pred_rows, ignore_index=True)
    pred_all.to_csv(outdir / "internal_oof_predictions_sensitivity.csv", index=False)
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(outdir / "internal_fold_metrics_sensitivity.csv", index=False)
    pooled_rows = []
    for (analysis, k, model), g in pred_all.groupby(["analysis", "k", "model"]):
        m = metric_summary(g["y_true"].values, g["y_pred"].values)
        m.update({"analysis": analysis, "k": k, "model": model})
        pooled_rows.append(m)
    pd.DataFrame(pooled_rows).sort_values(["analysis", "k", "model"]).to_csv(outdir / "internal_pooled_metrics_sensitivity.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args()
    run_covariate_enrichment(args.development, args.outdir, n_splits=args.n_splits)
    run_sensitivity(args.development, args.outdir, n_splits=args.n_splits)


if __name__ == "__main__":
    main()

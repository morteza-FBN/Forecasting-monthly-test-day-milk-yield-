from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

RECORDS = range(1, 10)
FORECAST_ORIGINS = range(1, 6)


def read_settings(path="config/model_settings.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def required_core_columns():
    cols = ["cow_id", "parity"]
    cols += [f"milk_{i}" for i in RECORDS]
    cols += [f"dim_{i}" for i in RECORDS]
    return cols


def standardize_dev_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {"BODY_ID":"cow_id", "Parity":"parity", "Date_of_calving":"calving_date"}
    for i in RECORDS:
        rename[f"Milk_recorod_{i}"] = f"milk_{i}"
        rename[f"Days_in_milk_{i}"] = f"dim_{i}"
    out = df.rename(columns={k:v for k,v in rename.items() if k in df.columns}).copy()
    return out


def clean_complete_curves(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_dev_columns(df)
    missing = [c for c in required_core_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=["cow_id", "parity"])
    core = [f"milk_{i}" for i in RECORDS] + [f"dim_{i}" for i in RECORDS]
    df = df.dropna(subset=core)
    for c in core + ["parity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=core+["parity"])
    positive = np.ones(len(df), dtype=bool)
    for i in RECORDS:
        positive &= (df[f"milk_{i}"] > 0) & (df[f"dim_{i}"] > 0)
    df = df.loc[positive].copy()
    df["cow_id"] = df["cow_id"].astype(str)
    df["parity"] = df["parity"].astype(int)
    return df.reset_index(drop=True)


def assign_group_folds(df: pd.DataFrame, n_splits=3, seed=42) -> pd.DataFrame:
    cows = np.array(sorted(df["cow_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(cows)
    fold_map = {cow: int(i % n_splits) for i, cow in enumerate(cows)}
    out = df.copy()
    out["fold"] = out["cow_id"].astype(str).map(fold_map)
    return out


def estimate_prior(train: pd.DataFrame, lam=30.0) -> pd.DataFrame:
    rows=[]
    for m in RECORDS:
        global_mean = train[f"milk_{m}"].mean()
        for p, grp in train.groupby("parity"):
            n = len(grp)
            raw = grp[f"milk_{m}"].mean()
            w = n/(n+lam)
            mu = w*raw + (1-w)*global_mean
            rows.append({"parity": int(p), "record": int(m), "n": int(n), "global_mean": global_mean, "raw_mean": raw, "prior_mu": mu})
    return pd.DataFrame(rows)


def prior_lookup(prior_df):
    d = {(int(r.parity), int(r.record)): float(r.prior_mu) for r in prior_df.itertuples()}
    global_by_record = prior_df.groupby("record")["global_mean"].first().to_dict()
    return d, global_by_record


def make_long_features(curves: pd.DataFrame, priors_by_fold: dict[int, pd.DataFrame]|None=None, lam=30.0, use_fold=True) -> pd.DataFrame:
    rows=[]
    if use_fold and "fold" not in curves.columns:
        curves = assign_group_folds(curves)
    for fold in sorted(curves["fold"].unique()) if use_fold else [None]:
        if use_fold:
            valid = curves[curves["fold"] == fold]
            train = curves[curves["fold"] != fold]
            prior = priors_by_fold.get(int(fold)) if priors_by_fold else estimate_prior(train, lam=lam)
        else:
            valid = curves
            prior = estimate_prior(curves, lam=lam)
        d, g = prior_lookup(prior)
        for idx, row in valid.iterrows():
            parity = int(row["parity"])
            for k in FORECAST_ORIGINS:
                prefix_milk = np.array([row[f"milk_{t}"] for t in range(1,k+1)], dtype=float)
                prefix_dim = np.array([row[f"dim_{t}"] for t in range(1,k+1)], dtype=float)
                prefix_prior = np.array([d.get((parity,t), g.get(t, np.nan)) for t in range(1,k+1)], dtype=float)
                prefix_resid = prefix_milk - prefix_prior
                if k >= 2 and prefix_dim[-1] != prefix_dim[0]:
                    raw_slope = (prefix_milk[-1] - prefix_milk[0])/(prefix_dim[-1] - prefix_dim[0])
                    resid_slope = (prefix_resid[-1] - prefix_resid[0])/(prefix_dim[-1] - prefix_dim[0])
                else:
                    raw_slope = 0.0
                    resid_slope = 0.0
                for m in range(k+1, 10):
                    mu = d.get((parity,m), g.get(m, np.nan))
                    y = float(row[f"milk_{m}"])
                    rows.append({
                        "curve_index": idx,
                        "cow_id": row["cow_id"],
                        "fold": fold if use_fold else -1,
                        "parity": parity,
                        "k": k,
                        "target_record": m,
                        "target_dim": float(row[f"dim_{m}"]),
                        "prior": mu,
                        "target_milk": y,
                        "target_residual": y - mu,
                        "prefix_mean": float(np.mean(prefix_milk)),
                        "prefix_max": float(np.max(prefix_milk)),
                        "prefix_last": float(prefix_milk[-1]),
                        "early_rise": float(prefix_milk[-1] - prefix_milk[0]),
                        "raw_slope": float(raw_slope),
                        "resid_mean": float(np.mean(prefix_resid)),
                        "resid_last": float(prefix_resid[-1]),
                        "resid_slope": float(resid_slope),
                        "prior_prefix_mean": float(np.mean(prefix_prior)),
                        "prior_prefix_last": float(prefix_prior[-1])
                    })
    return pd.DataFrame(rows)


def metric_summary(y, yhat):
    y=np.asarray(y, dtype=float); yhat=np.asarray(yhat, dtype=float)
    mae=np.mean(np.abs(y-yhat))
    rmse=np.sqrt(np.mean((y-yhat)**2))
    bias=np.mean(yhat-y)
    denom=np.sum((y-y.mean())**2)
    r2=1-np.sum((y-yhat)**2)/denom if denom>0 else np.nan
    return {"MAE":mae, "RMSE":rmse, "Bias":bias, "R2":r2}

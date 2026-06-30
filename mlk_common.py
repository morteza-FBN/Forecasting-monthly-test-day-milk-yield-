"""Common utilities for monthly test-day milk-yield forecasting.

The code is written to reproduce the analytical workflow described in the
JDS manuscript. Raw commercial herd files are not distributed with the code.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_RECORDS = 9
K_VALUES = [1, 2, 3, 4, 5]
DEFAULT_LAMBDA_PRIOR = 30.0

MILK_COLS = [f"Milk_recorod_{i}" for i in range(1, N_RECORDS + 1)]
DIM_COLS = [f"Days_in_milk_{i}" for i in range(1, N_RECORDS + 1)]
FAT_COLS = [f"Milk_fat_%{i}" for i in range(1, N_RECORDS + 1)]
PROTEIN_COLS = [f"Milk_protein_%{i}" for i in range(1, N_RECORDS + 1)]
SCC_COLS = [f"Milk_somatic_cell_count_{i}" for i in range(1, N_RECORDS + 1)]

COW_COL = "BODY_ID"
PARITY_COL = "Parity"

BACKGROUND_COVARIATES = [
    "Body_condition_score_at_dry_off",
    "Body_condition_score_at_calving",
    "DeltaBCS",
    "dry_period_days",
    "SEASON",
    "Calving_situation_code",
    "placentacod",
    "CALFSEX",
    "sexcod",
    "CALFWEIGHT",
    "CALF2WEIGHT",
]

ALL_COVARIATES = BACKGROUND_COVARIATES + [
    "BCS_Dry_off",
    "BCS_calving",
    "Delta",
    "New_class_delta",
    "Calving_situation",
    "Twining_status",
    "Placenta",
    "calfstatus",
    "METRIT",
    "Milk_fever",
    "number_of_help",
    "Left_displaced_abomasum_(LDA)",
    "MASTITIS",
    "heat_detection_rate",
    "Pregnancy_rare",
    "CALVINGI NTERVAL",
    "Early_ED",
    "LED",
    "ABORTION",
    "STILLBIRTH",
    "age_first_calving",
    "age_first_calving_days",
    "FIRSTIDATE1",
    "LASTIDATE1",
    "Days_to_first_service",
    "Days_Open",
    "number_of_inseminations_per_pregnancy",
    "conception_rates",
    "CYSTICL",
    "CYSTICR",
    "Pregnant_days",
    "dry_period_days",
]

@dataclass
class PriorTables:
    lambda_prior: float
    global_record_mean: Dict[int, float]
    parity_record_prior: Dict[Tuple[int, int], float]
    parity_record_count: Dict[Tuple[int, int], int]

    def lookup(self, parity: int, record: int) -> float:
        key = (int(parity), int(record))
        if key in self.parity_record_prior:
            return float(self.parity_record_prior[key])
        return float(self.global_record_mean[int(record)])


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def standardize_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure common ID/parity columns exist and are numeric where needed."""
    out = df.copy()
    if COW_COL not in out.columns:
        for alt in ["Dam ID", "Dam_ID", "Animal_Number", "cow_id", "Cow_ID"]:
            if alt in out.columns:
                out[COW_COL] = out[alt]
                break
    if PARITY_COL not in out.columns:
        for alt in ["dam calving parity", "Dam_calving_parity", "Lactation", "lactation"]:
            if alt in out.columns:
                out[PARITY_COL] = out[alt]
                break
    if COW_COL in out.columns:
        out[COW_COL] = pd.to_numeric(out[COW_COL], errors="coerce")
    if PARITY_COL in out.columns:
        out[PARITY_COL] = pd.to_numeric(out[PARITY_COL], errors="coerce")
    for col in MILK_COLS + DIM_COLS + FAT_COLS + PROTEIN_COLS + SCC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def retain_complete_curves(df: pd.DataFrame, require_increasing_dim: bool = False) -> pd.DataFrame:
    """Retain rows with cow ID, parity, 9 milk records, 9 DIM records, and positive values."""
    x = standardize_basic_columns(df)
    required = [COW_COL, PARITY_COL] + MILK_COLS + DIM_COLS
    missing_cols = [c for c in required if c not in x.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    mask = x[COW_COL].notna() & x[PARITY_COL].notna()
    mask &= x[MILK_COLS].notna().all(axis=1) & x[DIM_COLS].notna().all(axis=1)
    mask &= (x[MILK_COLS] > 0).all(axis=1) & (x[DIM_COLS] > 0).all(axis=1)
    if require_increasing_dim:
        dim_values = x[DIM_COLS].to_numpy(dtype=float)
        mask &= np.all(np.diff(dim_values, axis=1) > 0, axis=1)
    out = x.loc[mask].copy().reset_index(drop=True)
    out["curve_id"] = np.arange(len(out), dtype=int)
    out[COW_COL] = out[COW_COL].astype(int)
    out[PARITY_COL] = out[PARITY_COL].astype(int)
    return out


def retention_flow(df: pd.DataFrame, require_increasing_dim: bool = False) -> pd.DataFrame:
    x = standardize_basic_columns(df)
    rows = []
    current = pd.Series(True, index=x.index)
    steps = [
        ("Cow ID present", COW_COL in x.columns, x[COW_COL].notna() if COW_COL in x.columns else pd.Series(False, index=x.index)),
        ("Parity present", PARITY_COL in x.columns, x[PARITY_COL].notna() if PARITY_COL in x.columns else pd.Series(False, index=x.index)),
        ("All 9 milk-yield records present", all(c in x.columns for c in MILK_COLS), x[MILK_COLS].notna().all(axis=1) if all(c in x.columns for c in MILK_COLS) else pd.Series(False, index=x.index)),
        ("All 9 DIM records present", all(c in x.columns for c in DIM_COLS), x[DIM_COLS].notna().all(axis=1) if all(c in x.columns for c in DIM_COLS) else pd.Series(False, index=x.index)),
        ("All required milk/DIM values positive", True, ((x[MILK_COLS] > 0).all(axis=1) & (x[DIM_COLS] > 0).all(axis=1)) if all(c in x.columns for c in MILK_COLS + DIM_COLS) else pd.Series(False, index=x.index)),
    ]
    if require_increasing_dim and all(c in x.columns for c in DIM_COLS):
        dim_values = x[DIM_COLS].to_numpy(dtype=float)
        inc = pd.Series(np.all(np.diff(dim_values, axis=1) > 0, axis=1), index=x.index)
        steps.append(("Strictly increasing DIM", True, inc))
    for name, ok, condition in steps:
        before = int(current.sum())
        current &= condition if ok else False
        after = int(current.sum())
        rows.append({"filter": name, "rows_removed": before - after, "rows_remaining": after})
    return pd.DataFrame(rows)


def collapse_parity(parity: pd.Series | np.ndarray, threshold: Optional[int] = None) -> pd.Series:
    p = pd.Series(parity).astype(int)
    if threshold is not None:
        p = p.where(p < threshold, threshold)
    return p


def estimate_prior(train_df: pd.DataFrame, lambda_prior: float = DEFAULT_LAMBDA_PRIOR,
                   parity_collapse_ge: Optional[int] = None) -> PriorTables:
    train = train_df.copy()
    train[PARITY_COL] = collapse_parity(train[PARITY_COL], parity_collapse_ge).values
    global_mean = {}
    prior = {}
    counts = {}
    for m in range(1, N_RECORDS + 1):
        col = f"Milk_recorod_{m}"
        global_mean[m] = float(train[col].mean())
        grp = train.groupby(PARITY_COL)[col].agg(["mean", "count"])
        for parity, row in grp.iterrows():
            n = int(row["count"])
            w = n / (n + lambda_prior)
            mu = w * float(row["mean"]) + (1.0 - w) * global_mean[m]
            prior[(int(parity), m)] = float(mu)
            counts[(int(parity), m)] = n
    return PriorTables(lambda_prior=lambda_prior, global_record_mean=global_mean,
                       parity_record_prior=prior, parity_record_count=counts)


def add_prior_columns(df: pd.DataFrame, prior: PriorTables,
                      parity_collapse_ge: Optional[int] = None) -> pd.DataFrame:
    out = df.copy()
    parity_values = collapse_parity(out[PARITY_COL], parity_collapse_ge)
    for m in range(1, N_RECORDS + 1):
        out[f"prior_{m}"] = [prior.lookup(p, m) for p in parity_values]
        out[f"resid_{m}"] = out[f"Milk_recorod_{m}"] - out[f"prior_{m}"]
    return out


def _safe_slope(y1: float, y2: float, x1: float, x2: float) -> float:
    dx = float(x2) - float(x1)
    if not np.isfinite(dx) or abs(dx) < 1e-9:
        return 0.0
    return (float(y2) - float(y1)) / dx


def build_long_targets(df: pd.DataFrame, k: int, prior: PriorTables,
                       target_dim_mode: str = "actual",
                       train_record_mean_dim: Optional[Dict[int, float]] = None,
                       feature_set: str = "core",
                       parity_collapse_ge: Optional[int] = None) -> pd.DataFrame:
    """Build long-format target rows using only forecast-origin-controlled features.

    feature_set options: core, prefix_composition_scc, background, all_covariates.
    """
    if k < 1 or k >= N_RECORDS:
        raise ValueError("k must be between 1 and 8; manuscript uses k = 1 to 5")
    x = add_prior_columns(df, prior, parity_collapse_ge=parity_collapse_ge)
    rows = []
    for _, row in x.iterrows():
        cow = int(row[COW_COL])
        parity = int(collapse_parity(pd.Series([row[PARITY_COL]]), parity_collapse_ge).iloc[0])
        prefix_milk = np.array([row[f"Milk_recorod_{t}"] for t in range(1, k + 1)], dtype=float)
        prefix_dim = np.array([row[f"Days_in_milk_{t}"] for t in range(1, k + 1)], dtype=float)
        prefix_resid = np.array([row[f"resid_{t}"] for t in range(1, k + 1)], dtype=float)
        prefix_prior = np.array([row[f"prior_{t}"] for t in range(1, k + 1)], dtype=float)

        feat_base = {
            "curve_id": int(row.get("curve_id", -1)),
            "cow_id": cow,
            "parity": parity,
            "parity_cat": str(parity),
            "k": int(k),
            "prefix_mean_milk": float(np.mean(prefix_milk)),
            "prefix_max_milk": float(np.max(prefix_milk)),
            "prefix_last_milk": float(prefix_milk[-1]),
            "prefix_mean_resid": float(np.mean(prefix_resid)),
            "prefix_last_resid": float(prefix_resid[-1]),
            "prefix_mean_prior": float(np.mean(prefix_prior)),
            "prefix_last_prior": float(prefix_prior[-1]),
            "early_rise": float(prefix_milk[min(1, k - 1)] - prefix_milk[0]) if k >= 2 else 0.0,
            "milk_slope": _safe_slope(prefix_milk[0], prefix_milk[-1], prefix_dim[0], prefix_dim[-1]) if k >= 2 else 0.0,
            "resid_slope": _safe_slope(prefix_resid[0], prefix_resid[-1], prefix_dim[0], prefix_dim[-1]) if k >= 2 else 0.0,
            "dim_first": float(prefix_dim[0]),
            "dim_last": float(prefix_dim[-1]),
            "dim_span": float(prefix_dim[-1] - prefix_dim[0]) if k >= 2 else 0.0,
        }
        for t in range(1, k + 1):
            feat_base[f"milk_{t}"] = float(row[f"Milk_recorod_{t}"])
            feat_base[f"dim_{t}"] = float(row[f"Days_in_milk_{t}"])
            feat_base[f"resid_{t}"] = float(row[f"resid_{t}"])
            feat_base[f"prior_prefix_{t}"] = float(row[f"prior_{t}"])

        if feature_set in {"prefix_composition_scc", "all_covariates"}:
            for t in range(1, k + 1):
                for col, label in [(f"Milk_fat_%{t}", "fat"), (f"Milk_protein_%{t}", "protein"), (f"Milk_somatic_cell_count_{t}", "scc")]:
                    if col in x.columns:
                        feat_base[f"{label}_{t}"] = row[col]
            for cols, label in [(FAT_COLS[:k], "fat"), (PROTEIN_COLS[:k], "protein"), (SCC_COLS[:k], "scc")]:
                present = [c for c in cols if c in x.columns]
                if present:
                    values = pd.to_numeric(row[present], errors="coerce")
                    feat_base[f"prefix_mean_{label}"] = float(np.nanmean(values)) if np.isfinite(np.nanmean(values)) else np.nan
                    feat_base[f"prefix_last_{label}"] = float(values.iloc[-1])
        if feature_set in {"background", "all_covariates"}:
            for col in BACKGROUND_COVARIATES:
                if col in x.columns:
                    feat_base[col] = row[col]
        if feature_set == "all_covariates":
            for col in ALL_COVARIATES:
                if col in x.columns:
                    feat_base[col] = row[col]

        for m in range(k + 1, N_RECORDS + 1):
            target_dim = float(row[f"Days_in_milk_{m}"])
            if target_dim_mode == "record_mean":
                if train_record_mean_dim is None:
                    raise ValueError("train_record_mean_dim must be supplied for target_dim_mode='record_mean'")
                target_dim = float(train_record_mean_dim[m])
            prior_target = float(row[f"prior_{m}"])
            y_true = float(row[f"Milk_recorod_{m}"])
            rec = dict(feat_base)
            rec.update({
                "target_record": int(m),
                "target_record_cat": str(m),
                "target_dim": target_dim,
                "prior_target": prior_target,
                "y_true": y_true,
                "resid_true": y_true - prior_target,
            })
            rows.append(rec)
    return pd.DataFrame(rows)


def target_counts(n_curves: int) -> Dict[int, int]:
    return {k: int(n_curves * (N_RECORDS - k)) for k in K_VALUES}


def curve_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    x = standardize_basic_columns(df)
    milk = x[MILK_COLS].to_numpy(dtype=float)
    dim = x[DIM_COLS].to_numpy(dtype=float)
    peak = np.max(milk, axis=1)
    persist = milk[:, -1] / peak
    approx_305 = []
    for yy, dd in zip(milk, dim):
        grid = np.arange(1, 306)
        # np.interp uses first/last values for out-of-range points.
        approx_305.append(float(np.trapz(np.interp(grid, dd, yy), grid)))
    return pd.DataFrame({
        "curve_id": x.get("curve_id", pd.Series(np.arange(len(x)))).values,
        "cow_id": x[COW_COL].values,
        "parity": x[PARITY_COL].values,
        "peak_milk_kg_d": peak,
        "persistency_record9_over_peak": persist,
        "approx_305d_testday_yield_kg": approx_305,
    })


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.nanmean(np.abs(err / y_true)) * 100.0)
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else np.nan
    bias = float(np.mean(err))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "Bias": bias, "n": int(len(y_true))}


def cow_cluster_bootstrap(pred_df: pd.DataFrame, metric_col_true: str = "y_true", metric_col_pred: str = "y_pred",
                          cow_col: str = "cow_id", n_boot: int = 1000, seed: int = RANDOM_SEED) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    cows = pred_df[cow_col].dropna().unique()
    if len(cows) == 0:
        return {"MAE": np.nan, "CI_low": np.nan, "CI_high": np.nan, "n_boot": 0}
    observed = metric_summary(pred_df[metric_col_true].values, pred_df[metric_col_pred].values)["MAE"]
    vals = []
    grouped = {cow: idx.values for cow, idx in pred_df.groupby(cow_col).groups.items()}
    for _ in range(n_boot):
        sample = rng.choice(cows, size=len(cows), replace=True)
        idx = np.concatenate([grouped[c] for c in sample])
        b = pred_df.iloc[idx]
        vals.append(metric_summary(b[metric_col_true].values, b[metric_col_pred].values)["MAE"])
    return {"MAE": observed, "CI_low": float(np.percentile(vals, 2.5)), "CI_high": float(np.percentile(vals, 97.5)), "n_boot": int(n_boot)}


def describe_versions() -> pd.DataFrame:
    import platform
    import numpy
    import pandas
    import scipy
    import sklearn
    rows = [
        {"package": "python", "version": platform.python_version()},
        {"package": "numpy", "version": numpy.__version__},
        {"package": "pandas", "version": pandas.__version__},
        {"package": "scipy", "version": scipy.__version__},
        {"package": "scikit-learn", "version": sklearn.__version__},
    ]
    try:
        import lightgbm
        rows.append({"package": "lightgbm", "version": lightgbm.__version__})
    except Exception:
        rows.append({"package": "lightgbm", "version": "not installed"})
    try:
        import matplotlib
        rows.append({"package": "matplotlib", "version": matplotlib.__version__})
    except Exception:
        rows.append({"package": "matplotlib", "version": "not installed"})
    return pd.DataFrame(rows)

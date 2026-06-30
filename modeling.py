"""Model fitting utilities for prior-residual milk-yield forecasting."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge, SGDRegressor
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

RANDOM_SEED = 42

ID_AND_RESPONSE_COLS = {
    "curve_id", "cow_id", "k", "y_true", "resid_true", "y_pred", "resid_pred", "model",
}


def make_ohe():
    """Version-compatible OneHotEncoder with unknown categories ignored."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def split_feature_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in X.columns if c not in ID_AND_RESPONSE_COLS]
    categorical_cols = []
    numeric_cols = []
    for c in feature_cols:
        if X[c].dtype == object or str(X[c].dtype).startswith("category") or c.endswith("_cat"):
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    return numeric_cols, categorical_cols


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    numeric_cols, categorical_cols = split_feature_columns(X)
    if scale_numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def model_registry() -> Dict[str, object]:
    models = {
        "Ridge": Ridge(alpha=5.0),
        "Bayesian Ridge": BayesianRidge(max_iter=300, tol=0.001,
                                         alpha_1=1e-6, alpha_2=1e-6,
                                         lambda_1=1e-6, lambda_2=1e-6),
        "Lasso": Lasso(alpha=0.002, max_iter=10000, random_state=RANDOM_SEED),
        "ElasticNet": ElasticNet(alpha=0.006, l1_ratio=0.25, max_iter=10000, random_state=RANDOM_SEED),
        "SGD-Huber": SGDRegressor(loss="huber", alpha=0.0005, max_iter=5000,
                                   random_state=RANDOM_SEED, tol=1e-4),
        "Decision tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20,
                                                random_state=RANDOM_SEED),
    }
    if LGBMRegressor is not None:
        lightgbm_kwargs = dict(
            n_estimators=80,
            learning_rate=0.04,
            num_leaves=20,
            min_child_samples=25,
            subsample=0.90,
            colsample_bytree=1.0,
            reg_alpha=0.05,
            reg_lambda=0.10,
            random_state=RANDOM_SEED,
            verbose=-1,
        )
        models["Residual LightGBM"] = LGBMRegressor(**lightgbm_kwargs)
        models["Direct LightGBM"] = LGBMRegressor(**lightgbm_kwargs)
    return models


def fit_predict_model(model_name: str, train_long: pd.DataFrame, valid_long: pd.DataFrame,
                      feature_cols: List[str] | None = None) -> pd.DataFrame:
    """Fit a residual learner or direct learner and return prediction rows."""
    if feature_cols is None:
        feature_cols = [c for c in train_long.columns if c not in ID_AND_RESPONSE_COLS]
    X_train = train_long[feature_cols].copy()
    X_valid = valid_long[feature_cols].copy()
    registry = model_registry()
    if model_name not in registry:
        raise ValueError(f"Unknown model_name: {model_name}")
    base = clone(registry[model_name])
    scale = model_name not in {"Decision tree", "Residual LightGBM", "Direct LightGBM"}

    if model_name == "PLS":
        raise ValueError("PLS should be handled by fit_predict_pls because n_components depends on transformed columns")

    pre = make_preprocessor(X_train, scale_numeric=scale)
    response = "y_true" if model_name == "Direct LightGBM" else "resid_true"
    pipe = Pipeline([("preprocess", pre), ("model", base)])
    pipe.fit(X_train, train_long[response].values)
    pred_response = pipe.predict(X_valid)

    out = valid_long[["curve_id", "cow_id", "k", "target_record", "target_dim", "prior_target", "y_true", "resid_true"]].copy()
    out["model"] = model_name
    if model_name == "Direct LightGBM":
        out["y_pred"] = pred_response
        out["resid_pred"] = out["y_pred"] - out["prior_target"]
    else:
        out["resid_pred"] = pred_response
        out["y_pred"] = out["prior_target"] + out["resid_pred"]
    return out


def fit_predict_pls(train_long: pd.DataFrame, valid_long: pd.DataFrame,
                    feature_cols: List[str] | None = None) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = [c for c in train_long.columns if c not in ID_AND_RESPONSE_COLS]
    X_train = train_long[feature_cols].copy()
    X_valid = valid_long[feature_cols].copy()
    pre = make_preprocessor(X_train, scale_numeric=True)
    Xt = pre.fit_transform(X_train)
    Xv = pre.transform(X_valid)
    n_components = max(1, min(5, Xt.shape[1], Xt.shape[0] - 1))
    pls = PLSRegression(n_components=n_components)
    pls.fit(Xt, train_long["resid_true"].values)
    pred = pls.predict(Xv).ravel()
    out = valid_long[["curve_id", "cow_id", "k", "target_record", "target_dim", "prior_target", "y_true", "resid_true"]].copy()
    out["model"] = "PLS"
    out["resid_pred"] = pred
    out["y_pred"] = out["prior_target"] + out["resid_pred"]
    return out


def predict_baselines(valid_long: pd.DataFrame) -> pd.DataFrame:
    """Prior-only and prefix-mean residual prior baselines."""
    base_cols = ["curve_id", "cow_id", "k", "target_record", "target_dim", "prior_target", "y_true", "resid_true"]
    prior = valid_long[base_cols].copy()
    prior["model"] = "Empirical-Bayes prior"
    prior["resid_pred"] = 0.0
    prior["y_pred"] = prior["prior_target"]

    pref = valid_long[base_cols + ["prefix_mean_resid"]].copy()
    pref["model"] = "Prefix residual prior"
    pref["resid_pred"] = pref["prefix_mean_resid"].astype(float)
    pref["y_pred"] = pref["prior_target"] + pref["resid_pred"]
    pref = pref.drop(columns=["prefix_mean_resid"])
    return pd.concat([prior, pref], ignore_index=True)


def fit_all_models(train_long: pd.DataFrame, valid_long: pd.DataFrame,
                   model_names: Iterable[str] | None = None) -> pd.DataFrame:
    if model_names is None:
        model_names = [
            "Ridge", "Bayesian Ridge", "Lasso", "ElasticNet", "SGD-Huber", "PLS",
            "Decision tree", "Residual LightGBM", "Direct LightGBM",
        ]
    feature_cols = [c for c in train_long.columns if c not in ID_AND_RESPONSE_COLS]
    preds = [predict_baselines(valid_long)]
    for name in model_names:
        if name in {"Residual LightGBM", "Direct LightGBM"} and LGBMRegressor is None:
            continue
        if name == "PLS":
            preds.append(fit_predict_pls(train_long, valid_long, feature_cols=feature_cols))
        else:
            preds.append(fit_predict_model(name, train_long, valid_long, feature_cols=feature_cols))
    return pd.concat(preds, ignore_index=True)


def summarize_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    from mlk_common import metric_summary
    rows = []
    for keys, g in pred.groupby(["k", "model"], sort=True):
        k, model = keys
        m = metric_summary(g["y_true"].values, g["y_pred"].values)
        m.update({"k": k, "model": model})
        rows.append(m)
    out = pd.DataFrame(rows)
    if len(out):
        out = out[["k", "model", "n", "MAE", "RMSE", "MAPE", "R2", "Bias"]]
    return out.sort_values(["k", "MAE"]).reset_index(drop=True)


def add_auc_summary(pred: pd.DataFrame, event_threshold: float = -5.0) -> pd.DataFrame:
    rows = []
    for (k, model), g in pred.groupby(["k", "model"]):
        y_event = (g["resid_true"] <= event_threshold).astype(int).values
        score = -g["resid_pred"].values
        if len(np.unique(y_event)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_event, score)
        rows.append({
            "k": k,
            "model": model,
            "n": int(len(g)),
            "event_rate_pct": float(y_event.mean() * 100.0),
            "AUC": float(auc) if np.isfinite(auc) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["k", "model"]).reset_index(drop=True)

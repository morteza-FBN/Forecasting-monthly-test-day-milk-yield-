import argparse
from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from _common import assign_group_folds, estimate_prior, make_long_features, metric_summary

parser = argparse.ArgumentParser()
parser.add_argument("--development", required=True)
parser.add_argument("--external", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
dev = pd.read_csv(args.development)
ext = pd.read_csv(args.external)
# Use a full-development prior for external transport.
prior = estimate_prior(dev, lam=30.0)
ext = ext.copy(); ext["fold"] = 0
features = make_long_features(ext, priors_by_fold={0:prior}, use_fold=True)
# Train full-development features using full-development prior, then apply to external features.
dev_full = dev.copy(); dev_full["fold"] = 0
dev_features = make_long_features(dev_full, priors_by_fold={0:prior}, use_fold=True)
feature_cols = ["parity","target_record","target_dim","prior","prefix_mean","prefix_max","prefix_last","early_rise","raw_slope","resid_mean","resid_last","resid_slope","prior_prefix_mean","prior_prefix_last"]
models = {
    "Prior": None,
    "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=5.0)),
    "Bayesian Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), BayesianRidge(max_iter=300, tol=0.001))
}
rows=[]
for k,gext in features.groupby("k"):
    train = dev_features[dev_features["k"] == k]
    for name, model in models.items():
        if model is None:
            pred = gext["prior"].to_numpy()
        else:
            model.fit(train[feature_cols], train["target_residual"])
            pred = gext["prior"].to_numpy() + model.predict(gext[feature_cols])
        metrics = metric_summary(gext["target_milk"], pred)
        metrics.update({"k":int(k), "model":name})
        rows.append(metrics)
pd.DataFrame(rows).to_csv(args.output, index=False)
print(f"Wrote external performance: {args.output}")

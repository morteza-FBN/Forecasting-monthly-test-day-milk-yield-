import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge, Lasso, ElasticNet
from _common import metric_summary

parser = argparse.ArgumentParser()
parser.add_argument("--features", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.features)
feature_cols = ["parity","target_record","target_dim","prior","prefix_mean","prefix_max","prefix_last","early_rise","raw_slope","resid_mean","resid_last","resid_slope","prior_prefix_mean","prior_prefix_last"]
models = {
    "Prior": None,
    "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=5.0)),
    "Bayesian Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), BayesianRidge(max_iter=300, tol=0.001)),
    "Lasso": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Lasso(alpha=0.002, max_iter=10000)),
    "ElasticNet": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), ElasticNet(alpha=0.006, l1_ratio=0.25, max_iter=10000))
}
perf=[]
preds=[]
for k,gk in df.groupby("k"):
    for model_name, model in models.items():
        pred_all = np.full(len(gk), np.nan)
        for fold,gvalid in gk.groupby("fold"):
            valid_idx = gvalid.index
            if model is None:
                pred = gvalid["prior"].to_numpy()
            else:
                train = gk[gk["fold"] != fold]
                model.fit(train[feature_cols], train["target_residual"])
                pred_resid = model.predict(gvalid[feature_cols])
                pred = gvalid["prior"].to_numpy() + pred_resid
            pred_all[gk.index.get_indexer(valid_idx)] = pred
            for idx,p in zip(valid_idx,pred):
                preds.append({"row_index":int(idx),"cow_id":df.loc[idx,"cow_id"],"curve_index":int(df.loc[idx,"curve_index"]),"k":int(k),"target_record":int(df.loc[idx,"target_record"]),"model":model_name,"observed":float(df.loc[idx,"target_milk"]),"predicted":float(p),"absolute_error":abs(float(df.loc[idx,"target_milk"])-float(p))})
        metrics = metric_summary(gk["target_milk"], pred_all)
        metrics.update({"k":int(k),"model":model_name})
        perf.append(metrics)
perf = pd.DataFrame(perf)[["k","model","MAE","RMSE","Bias","R2"]]
perf.to_csv(args.output, index=False)
pd.DataFrame(preds).to_csv(Path(args.output).with_name("internal_predictions.csv"), index=False)
print(f"Wrote internal performance: {args.output}")

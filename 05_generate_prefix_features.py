import argparse
from pathlib import Path
import pandas as pd
from _common import assign_group_folds, make_long_features

parser = argparse.ArgumentParser()
parser.add_argument("--curves", required=True)
parser.add_argument("--priors", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
curves = assign_group_folds(pd.read_csv(args.curves))
prior_df = pd.read_csv(args.priors)
priors_by_fold = {int(f): g.drop(columns=["fold"]).reset_index(drop=True) for f,g in prior_df.groupby("fold")}
features = make_long_features(curves, priors_by_fold=priors_by_fold, use_fold=True)
features.to_csv(args.output, index=False)
print(f"Wrote features: {features.shape}")

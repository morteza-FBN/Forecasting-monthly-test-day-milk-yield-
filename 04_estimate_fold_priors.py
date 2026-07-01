import argparse
from pathlib import Path
import pandas as pd
from _common import assign_group_folds, estimate_prior

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--lambda_prior", type=float, default=30.0)
args = parser.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
df = assign_group_folds(pd.read_csv(args.input))
all_rows=[]
for fold in sorted(df["fold"].unique()):
    train = df[df["fold"] != fold]
    prior = estimate_prior(train, lam=args.lambda_prior)
    prior.insert(0, "fold", fold)
    all_rows.append(prior)
out = pd.concat(all_rows, ignore_index=True)
out.to_csv(args.output, index=False)
print(f"Wrote fold priors: {args.output}")

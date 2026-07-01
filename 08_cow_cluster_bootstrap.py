import argparse
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--n_boot", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.predictions)
rng = np.random.default_rng(args.seed)
rows=[]
for (k,model),g in df.groupby(["k","model"]):
    cow_errors = g.groupby(g["cow_id"].astype(str))["absolute_error"].apply(lambda s: s.to_numpy()).to_dict()
    cows = np.array(list(cow_errors.keys()))
    maes=[]
    for _ in range(args.n_boot):
        sampled = rng.choice(cows, size=len(cows), replace=True)
        vals = np.concatenate([cow_errors[c] for c in sampled])
        maes.append(vals.mean())
    rows.append({"k":k,"model":model,"MAE":g["absolute_error"].mean(),"CI_low":np.percentile(maes,2.5),"CI_high":np.percentile(maes,97.5),"n_boot":args.n_boot})
pd.DataFrame(rows).to_csv(args.output, index=False)
print(f"Wrote bootstrap summaries: {args.output}")

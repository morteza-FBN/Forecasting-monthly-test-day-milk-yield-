import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.input)
fig, ax = plt.subplots(figsize=(6,4))
for model, g in df.groupby("model"):
    ax.plot(g["k"], g["MAE"], marker="o", label=model)
ax.set_xlabel("Observed current-lactation records, k")
ax.set_ylabel("MAE, kg/d")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(args.output, dpi=300)
print(f"Wrote figure: {args.output}")

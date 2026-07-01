import argparse
from pathlib import Path
import pandas as pd
from _common import FORECAST_ORIGINS

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.input)
rows=[]
for idx,row in df.iterrows():
    for k in FORECAST_ORIGINS:
        for m in range(k+1,10):
            rows.append({"curve_index":idx,"cow_id":row["cow_id"],"parity":row["parity"],"k":k,"target_record":m,"target_milk":row[f"milk_{m}"],"target_dim":row[f"dim_{m}"]})
long = pd.DataFrame(rows)
long.to_csv(args.output, index=False)
print(f"Wrote {len(long)} future target rows.")

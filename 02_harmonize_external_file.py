import argparse
from pathlib import Path
import pandas as pd
from _common import clean_complete_curves

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.input)
rename = {"Dam_ID":"cow_id", "Dam_calving_parity":"parity", "Calving_Date":"calving_date"}
for i in range(1,10):
    rename[f"Milk_recorod_{i}"] = f"milk_{i}"
    rename[f"Days_in_milk_{i}"] = f"dim_{i}"
df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
clean = clean_complete_curves(df)
# External harmonization required strictly increasing DIM in the manuscript.
keep = pd.Series(True, index=clean.index)
for i in range(1,9):
    keep &= clean[f"dim_{i+1}"] > clean[f"dim_{i}"]
clean = clean.loc[keep].reset_index(drop=True)
clean.to_csv(args.output, index=False)
print(f"Retained {len(clean)} complete external curves from {clean['cow_id'].nunique()} cow IDs.")

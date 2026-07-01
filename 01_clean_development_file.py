import argparse
import pandas as pd
from pathlib import Path
from _common import clean_complete_curves

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.input)
clean = clean_complete_curves(df)
clean.to_csv(args.output, index=False)
print(f"Retained {len(clean)} complete curves from {clean['cow_id'].nunique()} cow IDs.")

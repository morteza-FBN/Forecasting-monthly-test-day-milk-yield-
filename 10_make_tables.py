import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()
out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
for src in ["outputs/main_table4_core_model_mae.csv", "outputs/main_table5_covariate_sensitivity_mae.csv", "outputs/supplemental_table_s6_curve_cow_paired_bootstrap.csv"]:
    p=Path(src)
    if p.exists(): shutil.copy2(p, out/p.name)
print(f"Copied table-ready CSV files to {out}")

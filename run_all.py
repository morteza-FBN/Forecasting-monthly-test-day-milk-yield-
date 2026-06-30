"""Run the complete milk-yield forecasting workflow."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print("\nRunning:", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True, help="Development CSV file")
    parser.add_argument("--external", required=True, help="External-validation CSV file")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--skip-covariates", action="store_true", help="Skip slower covariate/sensitivity analyses")
    parser.add_argument("--n-boot-external", type=int, default=1000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    script_dir = Path(__file__).resolve().parent

    run_cmd([py, script_dir / "01_date_conversion_QA.py", "--development", args.development, "--external", args.external, "--outdir", outdir])
    dev_g = outdir / "development_clean_complete_curves_gregorian_dates.csv"
    ext_g = outdir / "external_validation_clean_complete_curves_gregorian_dates.csv"
    run_cmd([py, script_dir / "02_descriptive_retention.py", "--development", dev_g, "--external", ext_g, "--outdir", outdir])
    run_cmd([py, script_dir / "03_internal_cv_core_models.py", "--development", dev_g, "--outdir", outdir])
    if not args.skip_covariates:
        run_cmd([py, script_dir / "04_covariate_enrichment_sensitivity.py", "--development", dev_g, "--outdir", outdir])
    run_cmd([py, script_dir / "05_external_validation.py", "--development", dev_g, "--external", ext_g, "--outdir", outdir, "--n-boot", str(args.n_boot_external)])
    run_cmd([py, script_dir / "06_make_figures_tables.py", "--outdir", outdir])


if __name__ == "__main__":
    main()

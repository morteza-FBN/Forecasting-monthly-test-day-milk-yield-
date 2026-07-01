# Empirical-Bayes-inspired residual learning for sparse monthly milk-yield forecasting

This repository accompanies a Journal of Dairy Science manuscript evaluating a leakage-controlled empirical-Bayes-inspired parity-by-record prior combined with residual learning to forecast future monthly test-day milk yield from 1 to 5 observed current-lactation records.

## Repository purpose

The repository is designed to support reproducibility and auditability of the analysis workflow while respecting commercial-farm data-use restrictions. Raw and cleaned cow-level herd-recording files are **not** included because they contain confidential commercial-farm information. Instead, this repository provides:

- executable Python scripts for the full workflow structure;
- a toy dataset with the same core structure as the analytical files;
- variable maps and data dictionaries;
- model settings and random seeds;
- nonidentifiable aggregate outputs used in manuscript tables;
- rendered manuscript figure/table PDFs supplied for review;
- a restricted-input manifest with checksums for nonpublic source files.

The toy data are synthetic and are provided only to demonstrate workflow execution and file structure. Numerical results produced from toy data will not match the manuscript.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

The workflow writes intermediate and toy-output files to `work/` and `outputs/toy_run/`.

## Main workflow

The scripts follow the analysis sequence described in the manuscript:

1. clean development file;
2. harmonize external file;
3. construct long-format future-target rows for k = 1 to 5;
4. estimate fold-specific parity-by-record priors;
5. generate prefix-residual features;
6. perform cow-grouped internal validation;
7. perform external validation;
8. compute cow-cluster bootstrap summaries;
9. run implementation sensitivities;
10. generate manuscript tables;
11. generate manuscript figures.

## Confidential data policy

Do not commit raw or cleaned commercial-farm cow-level files to this public repository. To run the workflow on restricted data, place source files in `data_restricted/` locally. This directory is excluded by `.gitignore`.

## Manuscript outputs

Nonidentifiable aggregate CSV outputs are stored in `outputs/` and table-ready files are stored in `tables/`. Main and supplemental figure/table PDFs are stored in `figures/` as manuscript-support files.

## Citation

Use the citation metadata in `CITATION.cff` when citing this repository. For journal submission, create a release and archive it in Zenodo, Figshare, OSF, or an institutional repository to obtain a DOI.

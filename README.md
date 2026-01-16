# ALM Optimizer Bench

This repository benchmarks an Augmented Lagrangian Method (ALM)-style constrained training variant against standard optimizers on image classification.

## What is included
- A reproducible notebook with full experiment code and logs:
  - `notebooks/alm_experiments.ipynb`
- A code-export for review/searchability (auto-generated from the notebook):
  - `src/alm_experiments.py`
- Paper-ready result artifacts (CSV/TXT):
  - `results/tables/c10_r18_paired_seed_results.csv`
  - `results/tables/c10_r18_stats.txt`
- Utility scripts:
  - `scripts/make_plots.py` generates figures from CSV results.
  - `scripts/recompute_stats.py` recomputes bootstrap CI and paired permutation p-value from the CSV.
  - `scripts/export_notebook_to_src.sh` re-exports the notebook to `src/` (optional helper).

## Setup (local)
Create a fresh environment and install:
1) `python -m venv .venv`
2) Activate it:
   - Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
   - Git Bash: `source .venv/Scripts/activate`
3) `pip install -r requirements.txt`

## Results: CIFAR-10 / ResNet-18 (paired seeds)
See:
- `results/tables/c10_r18_paired_seed_results.csv`
- `results/tables/c10_r18_stats.txt`

To generate plots and recompute stats:
- `python scripts/make_plots.py`
- `python scripts/recompute_stats.py`

## Notes on reproducibility
These experiments are sensitive to training recipes (data augmentation, schedules, batch size, etc.). If you add new experiments, save them under `results/tables/` (CSV + TXT summary), and optionally add figures to `results/figures/`.

## Repo structure
- `notebooks/` : interactive experiments (source of truth)
- `src/`       : exported script version (for review/search)
- `results/`   : saved artifacts (tables/figures) suitable for papers
- `scripts/`   : utilities for export/plot/stat recomputation

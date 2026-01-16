#!/usr/bin/env bash
set -euo pipefail
python -m nbconvert --to script notebooks/alm_experiments.ipynb --output-dir=src --output alm_experiments
echo "Exported to src/alm_experiments.py"

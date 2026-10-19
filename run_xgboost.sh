#!/bin/bash
#SBATCH --job-name=xgboost_final
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=Results/%x_%j.log


MODEL=${MODEL:-XGBoost}       # must match main.py --models choices (case-sensitive)
EXTRA_ARGS=${EXTRA_ARGS:-}
export PYTHONUNBUFFERED=1     # print progress to the log immediately


# Experiments
cd ~/MIE2027
.venv/bin/python main.py --models ${MODEL} ${EXTRA_ARGS}


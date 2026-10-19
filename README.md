# Label efficiency in clinical trial risk prediction

Code for the paper
*Label Efficiency in Clinical Trial Risk Prediction: Comparing Supervised Learning, LLMs, and Decision-One Models*.

We compare XGBoost, trained with label budgets from 25 to 12,800 clinical trials (CTs), with four models used without task-specific training: Qwen3.5-397B, Jev, Kev-9B and Kev-0.8B. The task is predicting dosing-error risk in CTs registered on ClinicalTrials.gov. Discrimination is assessed with the AUROC and calibration with the Brier skill score after Platt scaling.

## Repository structure

```
.
├── main.py               # Entry point: runs one model across all label budgets
├── constants.py          # Paths, label budgets and model specifications
├── subsets.py            # Creates and caches the labelled subsets shared by all models
├── calibration.py        # Platt scaling
├── utils.py              # Data loading and evaluation utilities
├── ourXGBoost.py         # XGBoost with Optuna hyperparameter search
├── ZeroShot.py           # Common interface for zero-shot models, with score caching
├── Qwen.py               # Qwen3.5-397B (self-hosted)
├── jev.py                # Jev (public API)
├── kev_predictor.py      # Kev-9B and Kev-0.8B (self-hosted)
├── prompts.py            # Prompts and questions given to the zero-shot models
├── run_kev.sh            # Example Slurm job: Kev server and predictions (GPU)
├── run_xgboost.sh        # Example Slurm job: XGBoost experiments (CPU)
├── requirements.txt
├── Analysis/
│   ├── analysis.py           # Loads results, computes metrics, plots figures
│   ├── paper_figures.ipynb   # Reproduces the figures and tables of the paper
│   └── Figures/
└── Results/              # Cached zero-shot scores and experiment results
```

## Installation

Tested with Python 3.12.

```bash
git clone https://github.com/ds4dh/zero-shot-vs-supervised-ct-risk.git
cd zero-shot-vs-supervised-ct-risk
pip install -r requirements.txt
```


## Data


The dataset is available on Hugging Face: [ds4dh/ct-dosing-errors](https://huggingface.co/datasets/ds4dh/ct-dosing-errors).

Download it into a folder of your choice and point `DATASETS_ROOT` to that folder:

```bash
export DATASETS_ROOT=/path/to/datasets
```

## Usage

### Model setup

Each zero-shot model is configured in `ZERO_SHOT_MODELS` (`constants.py`) by the URL of its server (`base_url`) and, if required, the environment variable holding its API key (`api_key_env`).

- **Qwen3.5-397B**: requires access to a server hosting the model. Adapt its entry in `ZERO_SHOT_MODELS` to your setup; the server must return token log-probabilities.
- **Jev** (version 1.13.0): accessed through the public TypeSafe API. Export your API key with `export TYPESAFE_API_KEY=...`.
- **Kev-9B and Kev-0.8B**: served locally on port 8009 from Hugging Face weights (revisions `db029f08b290afd9fee4aa4bbcd9ae48602d1eb0` and `<hash>`). We used a single GPU with 24 GB of memory. `run_kev.sh` starts the server and runs the predictions.

### Running experiments

Run all label budgets for one model:

```bash
python main.py --model <model> --sample_list
```

On a Slurm cluster, `run_kev.sh` and `run_xgboost.sh` give example jobs; adapt the partitions and environment to your cluster.

### Reproducing the paper results

Once the experiments have been run for all models, run `Analysis/paper_figures.ipynb` to reproduce the results and figures presented in the paper.


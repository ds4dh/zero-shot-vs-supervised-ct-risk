import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import precision_recall_curve

import pandas as pd
from constants import RESULTS_DIR
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def construct_xgboost_hyperparameter_search(param, trial, scale_pos_weight) -> dict:
    """

    Construct the hyperparameter search space for XGBoost model.
    """

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": param.random_seed,
    }

    # specific to the binary task
    if param.label == 'wilson_label':
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "aucpr"
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.5 * scale_pos_weight,
                                                         2.0 * scale_pos_weight)

    # specific to the regression task
    if param.label in ['dosing_error_rate', 'sum_dosing_errors']:
        # sample_weight is used to weight the loss function -> is used to manage class imbalance
        # sample_alpha = trial.suggest_float("sample_alpha", 5.0, 100.0, log=True) # F1 score label_frequencies 22.68 with mae on positive label
        # sample_alpha = trial.suggest_float("sample_alpha", 5.0, 200.0, log=True) # best for label_sum test F1 25.95 with mae
        sample_alpha = trial.suggest_float("sample_alpha", 5.0, 200.0, log=True)
        params["objective"] = trial.suggest_categorical("objective", ["reg:absoluteerror", "reg:pseudohubererror"])

    return params

import argparse
from random import seed

from datasets import Dataset, DatasetDict


def _draw(split: Dataset, k: int, name: str, seed: int) -> Dataset:
    if k > len(split):
        raise ValueError(
            f"Requested {k} CTs from '{name}', which only contains {len(split)}."
        )
    return split.shuffle(seed=seed).select(range(k))


import numpy as np


def get_labels(data, label_name: str) -> np.ndarray:
    """Return the binary labels of a Dataset as an integer array.

    Labels are stored in the nested 'labels' field of each CT, e.g.
    data["labels"][i][label_name].

    Args:
        data: Hugging Face Dataset (or subset) containing a 'labels' field.
        label_name: Name of the label to extract (e.g. "wilson_label").

    Returns:
        np.ndarray of shape (len(data),) with values in {0, 1}.
    """
    labels = np.array([ct_labels[label_name] for ct_labels in data["labels"]], dtype=int)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Label '{label_name}' is not binary.")
    return labels


def sample_labelled_subset(
    dataset: DatasetDict,
    n: int,
    seed: int,
    config: argparse.Namespace,
) -> DatasetDict:
    """Randomly sample n labelled CTs from the training and validation splits.

    train_fraction * n CTs are drawn from the training split and the remainder
    from the validation split, without stratification on the outcome. Each
    split is shuffled with the same seed and its first k CTs are kept, so
    subsets are nested across n for a given seed. If n is at least the total
    number of available CTs, both splits are returned in full.

    Args:
        dataset: DatasetDict with "train" and "validation" splits.
        n: Total number of labelled CTs to sample.
        seed: Random seed controlling the sampling.
        config: Provides train_fraction.

    Returns:
        DatasetDict with "train" and "validation" subsets.
    """
    train, validation = dataset["train"], dataset["validation"]

    # Full budget: use all available labelled CTs
    if n >= len(train) + len(validation):
        return DatasetDict({"train": train, "validation": validation})

    n_train = round(config.train_fraction * n)
    n_val = n - n_train

    return DatasetDict({
        "train": _draw(train, n_train, "train", seed),
        "validation": _draw(validation, n_val, "validation", seed),
    })



def performance_evaluation(labels, probabilities, predictions) -> dict:
    """Compute discrimination, calibration, and decision metrics.

    Args:
        labels: Binary ground-truth labels.
        probabilities: Predicted probabilities of the positive class.
        predictions: Binary decisions (threshold already applied).
    """
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    prevalence = labels.mean()

    return {
        "auroc": roc_auc_score(labels, probabilities),
        "auprc": average_precision_score(labels, probabilities),
        "brier": brier_score_loss(labels, probabilities),
        "brier_reference": prevalence * (1 - prevalence),
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def print_summary(results_df: pd.DataFrame,
                  metrics=("auroc", "brier", "f1")) -> None:
    """Print mean ± std of selected metrics per sample size, raw vs calibrated."""
    summary = (results_df
               .groupby(["model", "n", "calibrated"])[list(metrics)]
               .agg(["mean", "std"]))

    for model_name in summary.index.get_level_values("model").unique():
        print(f"\n=== {model_name} ===")
        table = pd.DataFrame(index=summary.loc[model_name].index.get_level_values("n").unique())
        for metric in metrics:
            for calibrated, tag in [(False, "raw"), (True, "cal")]:
                mean = summary.loc[(model_name, slice(None), calibrated), (metric, "mean")].to_numpy()
                std = summary.loc[(model_name, slice(None), calibrated), (metric, "std")].to_numpy()
                table[f"{metric}_{tag}"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean, std)]
        print(table.to_string())



def save_experiment(model_name: str, config, dataset,
                    metrics_df: pd.DataFrame, predictions_df: pd.DataFrame) -> str:
    """Save the configuration, metrics, predictions and test labels of a run.

    All files are written to a new timestamped directory, so previous runs are
    never overwritten:
        config.json          - all arguments + model name, timestamp, git commit
        metrics.csv          - one row per (n, repeat, calibrated)
        predictions.parquet  - one row per (n, repeat, calibrated, test CT)
        test_labels.csv      - nctId and label of each test CT (stored once)

    Args:
        model_name: Name of the evaluated model (used in the directory name).
        config: Parsed arguments (argparse.Namespace).
        dataset: DatasetDict, used to save the test labels.
        metrics_df: Metrics returned by run_experiment.
        predictions_df: Predictions returned by run_experiment.

    Returns:
        Path of the created run directory.
    """
    timestamp = datetime.now()
    run_dir = os.path.join(RESULTS_DIR, f"{model_name}_{timestamp:%Y-%m-%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=False)

    config_record = {**vars(config), "model": model_name, "timestamp": timestamp.isoformat()}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_record, f, indent=2, default=str)

    # Metrics
    metrics_df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    # Predictions (Parquet if available, CSV otherwise)
    try:
        predictions_df.to_parquet(os.path.join(run_dir, "predictions.parquet"), index=False)
    except ImportError:
        predictions_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
        print("pyarrow not installed: predictions saved as CSV.")

    # Test labels, stored once
    pd.DataFrame({
        "nctId": dataset["test"]["nctId"],
        "label": get_labels(dataset["test"], config.label),
    }).to_csv(os.path.join(run_dir, "test_labels.csv"), index=False)

    print(f"Results saved to {run_dir}")
    return run_dir


def select_decision_threshold(labels, probabilities):
    """Return the threshold maximising F1 on (labels, probabilities).

    Exact: every distinct predicted probability is tried as a threshold.
    If there is no positive CT, F1 is undefined; return +inf (predict all negative).
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)

    if labels.sum() == 0:
        return np.inf

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    precision, recall = precision[:-1], recall[:-1]          # last point has no threshold
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1)])
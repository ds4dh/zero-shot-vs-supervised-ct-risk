
from datasets import DatasetDict, load_from_disk
import os
from OurXGBoost import XGBoostModel
from constants import END_POINT_HF_DATASET_PATH
import argparse
from utils import *
import pandas as pd
from calibration import fit_platt
from tqdm import tqdm
from ZeroShot import ZeroShotModel
from subsets import load_or_create_subsets, build_id_index, get_subset


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    # dataset
    parser.add_argument('--label', type=str, default='wilson_label')

    # debugging mode
    parser.add_argument("--debug_n", type=int, default=None, help="Keep only n CTs per split (debugging).")

    # model
    parser.add_argument('--models', type=str, choices=['XGBoost', 'qwen3.5-397b', 
                                                       'jev', 'kev-0.8b', 'kev-9b'], default='qwen3.5-397b',
                        help='The model to use for experiments.')

    # sampling parameters
    # [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
    parser.add_argument('--sample_list', type=list, default=[25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800], 
                        help='List of sample sizes to use for experiments. The default is [100, 200].')
    parser.add_argument('--n_repeats_small', type=int, default=30,
                    help='Number of repeats for sample sizes below --small_sample_threshold.')
    parser.add_argument('--n_repeats_large', type=int, default=10,
                    help='Number of repeats for sample sizes at or above --small_sample_threshold.')
    parser.add_argument('--small_sample_threshold', type=int, default=800,
                    help='Sample sizes strictly below this value use --n_repeats_small.')
    parser.add_argument('--train_fraction', type=float, default=0.8,
                        help='Fraction of the dataset to use for training in each dataset sampling')

    
    return parser.parse_args()



def run_experiment(model, dataset, config, plan) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a model across increasing numbers of labelled CTs.

    For each sample size n and repetition, n labelled CTs are sampled from the
    training and validation splits, the model is fitted (no-op for zero-shot
    models), recalibrated with Platt scaling, and evaluated on the fixed test
    set before and after recalibration. Subsets are nested across n within a
    repetition (seed = config.random_seed + repeat). The number of repetitions
    depends on n (see get_n_repeats). Labelled subsets are read from a saved sampling 
    plan (see subsets.py), so every model is evaluated on exactly the same CTs for 
    each (n, repeat).

    Args:
        model: Implements fit(train, validation, seed, progress_label),
            construct_predictions(data) and name.
        dataset: DatasetDict with "train", "validation" and "test" splits.
        config: Provides sample_list, n_repeats_small, n_repeats_large,
            small_sample_threshold, random_seed and label.
        plan: DataFrame (n, repeat, split, nctId) from load_or_create_subsets.

    Returns:
        metrics_df: One row per (n, repeat, calibrated) with performance metrics.
        predictions_df: One row per (n, repeat, calibrated, test CT) with the
            predicted probability and binary decision, keyed by nctId.
    """
    # Test-set identifiers and labels, extracted once
    test_ids = dataset["test"]["nctId"]
    test_labels = get_labels(dataset["test"], config.label)
    pool, id_to_index = build_id_index(dataset) 

    metrics, predictions = [], []

    for n in sorted(plan["n"].unique()):   
        n_repeats = plan.loc[plan["n"] == n, "repeat"].nunique() 

        tqdm.write("\n" + "=" * 60)
        tqdm.write(f"{model.name} | sample size n = {n} | {n_repeats} repetitions")
        tqdm.write("=" * 60)

        for repeat in range(n_repeats):
            # Sample n labelled CTs (nested across n for a given seed)
            seed = config.random_seed + repeat
            subset = get_subset(pool, id_to_index, plan, n, repeat) 

            # Fit (training + hyperparameter search for supervised models) 
            model.fit(subset["train"], subset["validation"], seed=seed,
                      progress_label=f"n={n} | repeat {repeat + 1}/{n_repeats}")
            raw_proba_test, binary_test = model.construct_predictions(dataset["test"])

            # Recalibrate (validation subset for XGBoost, all n CTs for zero-shot models)
            calibrator = fit_platt(model, subset, label_name=config.label)
            calibrated_proba_test = calibrator.transform(raw_proba_test)

            # Evaluate before and after recalibration; binary decisions are unchanged by Platt scaling
            for calibrated, predicted_proba in [(False, raw_proba_test), (True, calibrated_proba_test)]:
                run_info = {"model": model.name, "n": n, "seed": seed, "repeat": repeat, "calibrated": calibrated}

                metrics.append({
                    **run_info,
                    "n_pos_train": int(get_labels(subset["train"], config.label).sum()),
                    "n_pos_validation": int(get_labels(subset["validation"], config.label).sum()),
                    "was_tuned": getattr(model, "was_tuned", None),
                    "platt_slope": calibrator.a,
                    **performance_evaluation(test_labels, predicted_proba, binary_test),
                })

                predictions.append(pd.DataFrame({
                    **run_info,
                    "nctId": test_ids,
                    "proba": predicted_proba,
                    "binary_prediction": binary_test,
                }))

    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def main():

    # loading the arguments
    parser = parse_args()

    # loading loading the dataset
    dataset = DatasetDict({
        "train": load_from_disk(os.path.join(END_POINT_HF_DATASET_PATH, "train")),
        "validation": load_from_disk(os.path.join(END_POINT_HF_DATASET_PATH, "validation")),
        "test": load_from_disk(os.path.join(END_POINT_HF_DATASET_PATH, "test")),
    })
    dataset = dataset.map(lambda example: {"nctId": example["metadata"]["nctId"]})
    
    # if in debbuging mode
    if parser.debug_n is not None:
        dataset = DatasetDict({
            split: data.shuffle(seed=parser.random_seed).select(range(min(parser.debug_n, len(data))))
            for split, data in dataset.items()
        })
        tqdm.write(f"DEBUG: {parser.debug_n} CTs per split.")

    # construct the model
    if parser.models == 'XGBoost':
        model = XGBoostModel(param=parser, dataset=dataset, n_trials=100)
    elif parser.models == 'qwen3.5-397b' or parser.models == 'jev' or parser.models == 'kev-0.8b' or parser.models=='kev-9b':
        model = ZeroShotModel(config=parser, dataset=dataset)
    else:
        raise NotImplementedError('Currently only: XGBoost, qwen3.5-397b, jev, kev-0.8b, kev-9b is supported.')

    # run the experiment
    plan = load_or_create_subsets(dataset=dataset, config=parser, sample_sizes=parser.sample_list, label=parser.label)
    metrics_df, predictions_df = run_experiment(model, dataset, parser, plan)

    # print the results
    print_summary(metrics_df)

    # save the results for future analysis
    save_experiment(model.name, parser, dataset, metrics_df, predictions_df)


if __name__ == "__main__":
    main()

    
import hashlib
import json
import os
import pandas as pd
from utils import sample_labelled_subset
from datasets import concatenate_datasets
from constants import SUBSETS_DIR


def _subsets_path(label, debug_n=None):
    suffix = f"_debug{debug_n}" if debug_n else ""
    return os.path.join(SUBSETS_DIR, f"subsets_{label}{suffix}.parquet")

def get_n_repeats(n: int, n_available: int, config) -> int:
    """Return the number of repetitions for sample size n."""
    if n >= n_available:
        return 1
    return config.n_repeats_small if n < config.small_sample_threshold else config.n_repeats_large


def _fingerprint(dataset, config, sample_sizes, id_column="nctId"):
    ids = sorted(dataset["train"][id_column]) + ["|"] + sorted(dataset["validation"][id_column])
    return {
        "sample_sizes": list(sample_sizes),
        "n_repeats_small": config.n_repeats_small,
        "n_repeats_large": config.n_repeats_large,
        "small_sample_threshold": config.small_sample_threshold,
        "train_fraction": config.train_fraction,
        "random_seed": config.random_seed,
        "labelled_ids_sha256": hashlib.sha256(",".join(ids).encode()).hexdigest(),
    }



def load_or_create_subsets(dataset, config, label, sample_sizes, id_column="nctId"):
    """Return a DataFrame (n, repeat, split, nctId) defining every labelled subset; created once, then reused."""
    path = _subsets_path(label, config.debug_n)
    meta_path = path.replace(".parquet", ".json")
    fingerprint = _fingerprint(dataset, config, sample_sizes, id_column)

    if os.path.exists(path):
        with open(meta_path) as f:
            if json.load(f) != fingerprint:
                raise ValueError(f"{path} was created with a different setup; delete it to regenerate.")
        return pd.read_parquet(path)

    n_available = len(dataset["train"]) + len(dataset["validation"])
    rows = []
    for n in sample_sizes:
        for repeat in range(get_n_repeats(n, n_available, config)):
            subset = sample_labelled_subset(dataset, n, seed=config.random_seed + repeat, config=config)
            for split in ("train", "validation"):
                rows += [{"n": n, "repeat": repeat, "split": split, id_column: i}
                         for i in subset[split][id_column]]

    plan = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plan.to_parquet(path, index=False)
    with open(meta_path, "w") as f:
        json.dump(fingerprint, f, indent=2)
    return plan


def build_id_index(dataset, id_column="nctId"):
    pool = concatenate_datasets([dataset["train"], dataset["validation"]])
    ids = pool[id_column]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs in labelled pool.")
    return pool, {nct_id: i for i, nct_id in enumerate(ids)}


def get_subset(pool, id_to_index, plan, n, repeat, id_column="nctId"):
    """Rebuild the labelled subset for (n, repeat) from the saved plan, in plan order."""
    rows = plan[(plan["n"] == n) & (plan["repeat"] == repeat)]
    subset = {}
    for split in ("train", "validation"):
        wanted = rows.loc[rows["split"] == split, id_column].tolist()
        subset[split] = pool.select([id_to_index[i] for i in wanted])
    if len(subset["train"]) + len(subset["validation"]) != len(rows):
        raise ValueError(f"Subset size mismatch for n={n}, repeat={repeat}.")
    return subset
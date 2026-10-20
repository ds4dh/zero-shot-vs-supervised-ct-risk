import json
import os

import numpy as np
import pandas as pd
from datasets import concatenate_datasets
from tqdm import tqdm

from constants import RESULTS_DIR, ZERO_SHOT_MODELS
from prompts import PROMPT_VERSION, QUESTIONS
from utils import get_labels, select_decision_threshold
from Qwen import build_qwen_predictor
from jev import build_jev_predictor
from kev_predictor import build_kev_predictor


SPLITS = ("train", "validation", "test")


class ZeroShotModel:
    """Interface between a zero-shot model (Jev, Kev, Qwen) and the experiments.

    Predictions for all CTs are computed once, cached in `{scores_dir}/{name}.parquet`, and
    looked up afterwards. A fingerprint (model, revision, prompt, question) is stored next to
    them; loading fails if it differs, so a changed setup never silently reuses old predictions.
    """

    calibration_split = "all"

    def __init__(self, dataset, config, batch_size=32, id_column="nctId"):
        if config.models not in ZERO_SHOT_MODELS:
            raise ValueError(f"Unknown zero-shot model '{config.models}'. "
                             f"Available: {list(ZERO_SHOT_MODELS)}")
        self.name = config.models
        self.label_name = config.label
        self.id_column = id_column
        self.threshold = None

        # logs paths
        scores_dir = os.path.join(RESULTS_DIR, self.name)
        self.scores_path = os.path.join(scores_dir, f"{self.name}.parquet")
        self.fingerprint_path = os.path.join(scores_dir, f"{self.name}.json")
        self.failed_path = os.path.join(scores_dir, f"{self.name}_failed.json")

        self.fallback_ids = []

        self.all_predictions = self.load_or_compute_predictions(dataset, config, batch_size)

    # ------------------------------------------------------------------ experiment interface
    def fit(self, train, validation, seed=None, progress_label=""):
        """No training: only selects the decision threshold on all labelled CTs."""
        labelled = concatenate_datasets([train, validation])
        self.threshold = select_decision_threshold(
            get_labels(labelled, self.label_name), self.predict_proba(labelled)
        )

    def predict_proba(self, data):
        ids = pd.Index(data[self.id_column])
        missing = ids.difference(self.all_predictions.index)
        if len(missing) > 0:
            raise KeyError(f"{len(missing)} CTs have no {self.name} prediction (e.g. {missing[0]}).")
        return self.all_predictions.loc[ids].to_numpy(dtype=float)

    def construct_predictions(self, data):
        if self.threshold is None:
            raise RuntimeError("Call fit() before construct_predictions().")
        probabilities = self.predict_proba(data)
        return probabilities, (probabilities >= self.threshold).astype(int)

    # ------------------------------------------------------------------ predictions and caching
    def load_or_compute_predictions(self, dataset, config, batch_size):
        """Load the predictions from disk; compute (and save) those that are missing.

        CTs the predictor could not score (returned None, e.g. input over the token limit or
        GPU out of memory) are recorded in {name}_failed.json and not retried; delete that file
        to retry them. If the model spec has "fallback": "train_prevalence", they get the
        training-split prevalence (in memory only, never cached); otherwise the run stops.
        """
        fingerprint = self._build_fingerprint(config)
        predictions = self._load_predictions(fingerprint)

        all_ids = [i for split in SPLITS for i in dataset[split][self.id_column]]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Duplicate CT ids across splits.")

        known_failures = self._load_failed_ids(fingerprint)
        missing = set(all_ids) - set(predictions.index) - known_failures
        if missing:
            tqdm.write(f"{self.name}: {len(predictions)} predictions loaded, {len(missing)} to compute "
                       f"({len(known_failures)} known failures skipped).")
            predictions = self._compute_missing_predictions(
                dataset, missing, predictions, config, fingerprint, batch_size
            )
        else:
            tqdm.write(f"{self.name}: {len(predictions)} predictions loaded from {self.scores_path}.")

        still_missing = sorted(set(all_ids) - set(predictions.index))
        if still_missing:
            self._save_failed_ids(still_missing, fingerprint)
            if ZERO_SHOT_MODELS[self.name].get("fallback") != "train_prevalence":
                raise RuntimeError(
                    f"{self.name}: {len(still_missing)} CTs could not be scored (e.g. {still_missing[:5]}). "
                    f"Delete {self.failed_path} and rerun to retry them."
                )
            fallback = float(np.mean(get_labels(dataset["train"], self.label_name)))
            tqdm.write(f"{self.name}: {len(still_missing)} CTs unscored; assigned the training "
                       f"prevalence {fallback:.4f} (listed in {self.failed_path}).")
            filler = pd.Series(fallback, index=pd.Index(still_missing, name=self.id_column), name="proba")
            predictions = pd.concat([predictions, filler])      # in memory only, never cached
            self.fallback_ids = still_missing

        self._validate_predictions(predictions)
        return predictions.loc[all_ids]

    def _load_failed_ids(self, fingerprint):
        """IDs that failed in an earlier run with the same setup (empty if none or setup changed)."""
        if not os.path.exists(self.failed_path):
            return set()
        with open(self.failed_path) as f:
            stored = json.load(f)
        return set(stored["failed"]) if stored["fingerprint"] == fingerprint else set()

    def _save_failed_ids(self, failed, fingerprint):
        os.makedirs(os.path.dirname(self.failed_path), exist_ok=True)
        with open(self.failed_path, "w") as f:
            json.dump({"fingerprint": fingerprint, "failed": list(failed)}, f, indent=2)

    def _build_fingerprint(self, config):
        """Everything that defines the predictions (not the URL or API key)."""
        return {
            **ZERO_SHOT_MODELS[self.name]["fingerprint"],
            "label": config.label,
            "question": QUESTIONS[config.label],
            "prompt_version": PROMPT_VERSION,
        }

    def _load_predictions(self, fingerprint):
        if not os.path.exists(self.scores_path):
            return pd.Series(dtype=float, name="proba").rename_axis(self.id_column)

        if not os.path.exists(self.fingerprint_path):
            raise FileNotFoundError(f"{self.fingerprint_path} missing; cannot verify cached predictions.")
        with open(self.fingerprint_path) as f:
            stored = json.load(f)
        if stored != fingerprint:
            raise ValueError(
                f"Cached predictions for {self.name} were produced with a different setup.\n"
                f"Stored:  {stored}\nCurrent: {fingerprint}\n"
                f"Delete {self.scores_path} or change the model name."
            )
        return pd.read_parquet(self.scores_path).set_index(self.id_column)["proba"]

    
    def _compute_missing_predictions(self, dataset, missing, predictions, config, fingerprint, batch_size):
        """Compute the predictions of the CTs not cached yet, checkpointing after every batch.

        CTs for which the predictor returns None (e.g. out of memory after retries) are left
        unscored; they stay 'missing' and are retried by the next run.
        """
        predict_batch = self._build_predictor(config)

        os.makedirs(os.path.dirname(self.scores_path), exist_ok=True)
        with open(self.fingerprint_path, "w") as f:
            json.dump(fingerprint, f, indent=2)

        pool = concatenate_datasets([dataset[split] for split in SPLITS])
        to_predict = pool.select([k for k, i in enumerate(pool[self.id_column]) if i in missing])

        n_failed = 0
        for start in tqdm(range(0, len(to_predict), batch_size), desc=f"Predicting {self.name}"):
            batch = to_predict.select(range(start, min(start + batch_size, len(to_predict))))
            probabilities = predict_batch(batch)
            if len(probabilities) != len(batch):
                raise ValueError(f"{self.name} returned {len(probabilities)} values for {len(batch)} CTs.")

            scored = [(i, p) for i, p in zip(batch[self.id_column], probabilities) if p is not None]
            n_failed += len(batch) - len(scored)
            if scored:
                ids, probs = zip(*scored)
                new = pd.Series(probs, name="proba", dtype=float,
                                index=pd.Index(ids, name=self.id_column))
                predictions = pd.concat([predictions, new])
                self._write_predictions(predictions)

        if n_failed:
            tqdm.write(f"{self.name}: {n_failed} CTs could not be scored in this run.")
        return predictions

    def _build_predictor(self, config):
        """Dispatch to the model-specific function: predict_batch(cts) -> list of P(yes)."""
        spec = ZERO_SHOT_MODELS[self.name]
        if self.name == "qwen3.8-flash-next":
            return build_qwen_predictor(spec, config.label)
        elif self.name == 'jev':
            return build_jev_predictor(spec, config.label)
        if self.name in ("kev-0.8b", "kev-9b"):
            return build_kev_predictor(spec, config.label)
            
        raise NotImplementedError(f"No predictor implemented for {self.name}.")

    def _write_predictions(self, predictions):
        """Save all predictions computed so far (checkpoint); atomic, so an interruption never corrupts the file."""
        tmp_path = self.scores_path + ".tmp"
        predictions.rename_axis(self.id_column).reset_index().to_parquet(tmp_path, index=False)
        os.replace(tmp_path, self.scores_path)

    def _validate_predictions(self, predictions):
        """Fail loudly on duplicated CTs, missing values or probabilities outside [0, 1]."""
        if predictions.index.duplicated().any():
            raise ValueError(f"Duplicate ids in {self.scores_path}.")
        values = predictions.to_numpy(dtype=float)
        if np.isnan(values).any() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"{self.name} predictions contain NaN or values outside [0, 1].")


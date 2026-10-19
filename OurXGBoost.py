import numpy as np
import optuna
import pandas as pd
from datasets import DatasetDict
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from utils import construct_xgboost_hyperparameter_search

optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostModel:
    """XGBoost trained on structured features, with hyperparameters optimized by
    Optuna on the sampled validation subset.

    Features are prepared once for all splits at initialization and indexed by
    CT identifier; `fit` and `predict_proba` only look up the requested CTs.

    Two degenerate cases, frequent at small sample sizes, are handled explicitly:
      - no positive CT in the training subset: the model predicts the observed
        prevalence of all sampled CTs;
      - a single class in the validation subset: AUROC is undefined, so default
        hyperparameters are used.
    """

    name = "xgboost"
    calibration_split = "validation"  # Platt scaling is fitted on the validation subset

    def __init__(self, param, dataset: DatasetDict, n_trials: int, id_column: str = "nctId"):
        """
        Args:
            param: General config; must provide `label` (label column name).
            dataset: Full DatasetDict, used once to build the feature table.
            n_trials: Number of Optuna trials.
            id_column: Name of the CT identifier column.
        """
        self.param = param
        self.n_trials = n_trials
        self.id_column = id_column

        # Structured features + label for all CTs, indexed by CT identifier
        self.feature_table = prepare_xgboost_features(dataset, param.label, id_column)

        self.model = None
        self.constant_prediction = None
        self.decision_threshold = None
        self.was_tuned = False  # logged to report how often tuning was possible

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def fit(self, train, validation, seed: int, progress_label: str = "") -> None:
        """Optimize hyperparameters on `validation`, then train on `train`.

        Args:
            train: Sampled training subset.
            validation: Sampled validation subset.
            seed: Repetition-specific seed (Optuna sampler and XGBoost).
            progress_label: Text shown in the progress bar (e.g. "n=400 | repeat 1/10").
        """
        X_train, y_train = self._extract_features_and_labels(train)
        X_val, y_val = self._extract_features_and_labels(validation)

        # Degenerate case 1: no positive CT in the training subset
        if y_train.sum() == 0:
            self.model = None
            self.was_tuned = False
            self.constant_prediction = (y_train.sum() + y_val.sum()) / (len(y_train) + len(y_val))
            tqdm.write(f"  {progress_label}: no positive CT in training subset -> constant prediction")
            return
        self.constant_prediction = None

        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

        # Degenerate case 2: AUROC undefined on a single-class validation subset
        if len(np.unique(y_val)) < 2:
            best_params = add_fixed_params({"scale_pos_weight": scale_pos_weight}, seed)
            self.was_tuned = False
            tqdm.write(f"  {progress_label}: single class in validation subset -> default hyperparameters")
        else:
            best_params = self._optimize_hyperparameters(
                X_train, y_train, X_val, y_val, scale_pos_weight, seed, progress_label
            )
            self.was_tuned = True

        self.model = XGBClassifier(**best_params)
        self.model.fit(X_train, y_train)

        # optimize decision threshold
        self.decision_threshold = self._select_decision_threshold(X_val, y_val)

    def _select_decision_threshold(self, X_val, y_val) -> float:
        """Select the decision threshold that maximizes the F1 score on the validation subset."""
        y_probs = self.model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0, 1, 101)
        f1_scores = [f1_score(y_val, y_probs >= t, zero_division=0) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        return best_threshold

    def construct_predictions(self, data):
        """ Return the predicted and predicted labels for the given data. """
        y_probs = self.predict_proba(data)
        if self.decision_threshold is not None:
            y_preds = (y_probs >= self.decision_threshold).astype(int)
        else:
            y_preds = y_probs
            print('Warning: Decision threshold is not set. Returning probabilities as predictions.')
        return y_probs, y_preds

    def predict_proba(self, data) -> np.ndarray:
        """Return the predicted probability of the positive class."""
        X, _ = self._extract_features_and_labels(data)
        if self.model is None:
            return np.full(len(X), self.constant_prediction)
        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _extract_features_and_labels(self, data):
        """Look up the structured features and labels of the CTs in `data`."""
        rows = self.feature_table.loc[data[self.id_column]]
        features = rows.drop(columns=[self.param.label])
        labels = rows[self.param.label].to_numpy(dtype=int)
        return features, labels

    def _optimize_hyperparameters(self, X_train, y_train, X_val, y_val,
                                  scale_pos_weight, seed, progress_label: str = "") -> dict:
        """Run Optuna (TPE sampler) to maximize validation AUROC."""

        def objective(trial):
            params = construct_xgboost_hyperparameter_search(self.param, trial, scale_pos_weight)
            params["random_state"] = seed  # repetition-specific seed
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        # One progress bar per search, kept on screen after completion
        progress = tqdm(total=self.n_trials, desc=f"  {progress_label}", leave=True)

        def report(study, trial):
            progress.update(1)
            progress.set_postfix(best_val_auroc=f"{study.best_value:.3f}")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=self.n_trials, callbacks=[report])
        progress.close()

        return add_fixed_params(study.best_params, seed)


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #
def prepare_xgboost_features(dataset: DatasetDict, label: str, id_column: str) -> pd.DataFrame:
    """Build one numeric table (structured features + label) for all splits.

    Text features are dropped and missing values are kept as NaN, as XGBoost
    handles them natively. No rows are removed, so all models are evaluated
    on the same CTs. Rows are indexed by CT identifier.
    """
    frames = []
    for split in dataset.values():
        features = pd.DataFrame(split["features"])
        text_columns = [name for name, feature in split.features["features"].items()
                        if feature.dtype == "string"]
        features = features.drop(columns=text_columns)
        features[label] = pd.DataFrame(split["labels"])[label].to_numpy()
        features.index = split[id_column]
        frames.append(features)

    table = pd.concat(frames)

    # Boolean features with missing values are stored as `object` (True/False/None).
    # Convert them to float (1.0/0.0/NaN) so XGBoost can handle missing values natively.
    object_columns = table.select_dtypes(include="object").columns.drop(label, errors="ignore")
    table[object_columns] = table[object_columns].astype(float)

    if not table.index.is_unique:
        raise ValueError(f"Duplicate identifiers found in column '{id_column}'.")
    return table


def add_fixed_params(params: dict, seed: int) -> dict:
    """Add the fixed binary-task settings and the repetition-specific seed."""
    return {**params, "objective": "binary:logistic", "eval_metric": "aucpr", "random_state": seed}
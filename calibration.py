
import numpy as np
from datasets import concatenate_datasets
from scipy.optimize import minimize
from scipy.special import expit, logit


class PlattCalibrator:
    """Platt scaling: p_calibrated = sigmoid(a * logit(p) + b).

    Uses Platt's smoothed targets (Platt, 1999), which keep the fit stable
    when the calibration set contains few or no positive examples.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps          # clipping to avoid infinite logits
        self.a, self.b = 1.0, 0.0

    def _to_logit(self, probabilities) -> np.ndarray:
        return logit(np.clip(probabilities, self.eps, 1 - self.eps))

    def fit(self, probabilities, labels) -> "PlattCalibrator":
        x = self._to_logit(np.asarray(probabilities, dtype=float))
        y = np.asarray(labels, dtype=int)

        # Smoothed targets
        n_pos, n_neg = y.sum(), len(y) - y.sum()
        targets = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

        def negative_log_likelihood(params):
            p = np.clip(expit(params[0] * x + params[1]), self.eps, 1 - self.eps)
            return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))

        result = minimize(negative_log_likelihood, x0=[1.0, 0.0], method="L-BFGS-B", bounds=[(0, None), (None, None)])
        self.a, self.b = result.x
        return self

    def transform(self, probabilities) -> np.ndarray:
        x = self._to_logit(np.asarray(probabilities, dtype=float))
        return expit(self.a * x + self.b)


def get_labels(data, label_name: str) -> np.ndarray:
    """Return the binary labels of a Dataset (labels are stored in the 'labels' field)."""
    return np.array([labels[label_name] for labels in data["labels"]], dtype=int)


def fit_platt(model, subset, label_name: str) -> PlattCalibrator:
    """Fit Platt scaling on the CTs designated by `model.calibration_split`.

    - "validation": the sampled validation subset (supervised models);
    - "all": all n sampled CTs, i.e. training and validation (zero-shot models).
    """
    if model.calibration_split == "validation":
        calibration_data = subset["validation"]
    elif model.calibration_split == "all":
        calibration_data = concatenate_datasets([subset["train"], subset["validation"]])
    else:
        raise ValueError(f"Unknown calibration split: {model.calibration_split}")

    probabilities = model.predict_proba(calibration_data)
    labels = get_labels(calibration_data, label_name)
    return PlattCalibrator().fit(probabilities, labels)


"""Load experiment runs (metrics.csv, predictions.parquet, test_labels.csv) and plot AUROC / Brier vs n."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def load_run(run_dir):
    """Return config, metrics, predictions (with labels merged) for one run folder."""
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    metrics = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    predictions = pd.read_parquet(os.path.join(run_dir, "predictions.parquet"))

    labels = pd.read_csv(os.path.join(run_dir, "test_labels.csv"))
    label_column = [c for c in labels.columns if c != "nctId"][0]
    labels = labels.rename(columns={label_column: "label"})
    predictions = predictions.merge(labels[["nctId", "label"]], on="nctId", how="left", validate="many_to_one")
    if predictions["label"].isna().any():
        raise ValueError(f"{run_dir}: some test predictions have no label.")
    return config, metrics, predictions


def check_run(name, config, metrics, predictions):
    """Print completeness checks: grid, repeats, test-set size, missing scores."""
    print(f"=== {name} ===")
    print(f"label={config.get('label')}  debug_n={config.get('debug_n')}  commit={config.get('git_commit', '?')[:8]}")
    grid = metrics[~metrics["calibrated"]].groupby("n")["repeat"].nunique()
    print("repeats per n:", grid.to_dict())
    per_run = predictions.groupby(["n", "repeat", "calibrated"])["nctId"].nunique()
    print(f"test CTs per run: min={per_run.min()} max={per_run.max()}  "
          f"prevalence={predictions.drop_duplicates('nctId')['label'].mean():.4f}")
    print(f"NaN probabilities: {predictions['proba'].isna().sum()}\n")


def metrics_from_predictions(predictions):
    """Recompute AUROC and Brier per (n, repeat, calibrated) from the saved predictions."""
    rows = []
    for (n, repeat, calibrated), group in predictions.groupby(["n", "repeat", "calibrated"]):
        group = group.dropna(subset=["proba"])
        rows.append({"n": n, "repeat": repeat, "calibrated": calibrated,
                     "auroc": roc_auc_score(group["label"], group["proba"]),
                     "brier": brier_score_loss(group["label"], group["proba"])})
    return pd.DataFrame(rows)


def summarise(scores, metric, calibrated, interval=(25, 75)):
    """Median and percentile interval (default IQR) across repeats, per n."""
    subset = scores[scores["calibrated"] == calibrated]
    return subset.groupby("n")[metric].agg(
        median="median",
        low=lambda x: np.percentile(x, interval[0]),
        high=lambda x: np.percentile(x, interval[1]),
        repeats="count",
    ).reset_index()


def is_constant_in_n(scores, metric="auroc", calibrated=False, tol=1e-9):
    """True for models whose metric does not depend on n (zero-shot raw scores)."""
    values = scores.loc[scores["calibrated"] == calibrated, metric]
    return values.max() - values.min() < tol


def compare_to_zero_shot(scores_by_model, supervised="XGBoost", metric="auroc", calibrated=False):
    """Per n and zero-shot model: median supervised-minus-zero-shot difference, and the share of
    repeats in which the supervised model is better (higher AUROC / lower Brier). No interpolation."""
    sup = scores_by_model[supervised]
    sup = sup[sup["calibrated"] == calibrated][["n", "repeat", metric]]
    rows = []
    for name, scores in scores_by_model.items():
        if name == supervised:
            continue
        zs = scores[scores["calibrated"] == calibrated][["n", "repeat", metric]]
        paired = sup.merge(zs, on=["n", "repeat"], suffixes=("_sup", "_zs"))
        diff = paired[f"{metric}_sup"] - paired[f"{metric}_zs"]
        better = diff > 0 if metric == "auroc" else diff < 0
        paired = paired.assign(diff=diff, better=better)
        for n, g in paired.groupby("n"):
            rows.append({"zero_shot": name, "n": n, "median_diff": g["diff"].median(),
                         "share_supervised_better": g["better"].mean(), "repeats": len(g)})
    return pd.DataFrame(rows).pivot(index="n", columns="zero_shot",
                                    values=["share_supervised_better", "median_diff"])


def bootstrap_auroc(predictions, n_boot=1000, seed=0):
    """AUROC and 95% percentile CI over test CTs, for the raw scores of a zero-shot model (for tables)."""
    run = predictions[~predictions["calibrated"]]
    first = run[["n", "repeat"]].iloc[0]
    run = run[(run["n"] == first["n"]) & (run["repeat"] == first["repeat"])].dropna(subset=["proba"])
    y, p = run["label"].to_numpy(), run["proba"].to_numpy()
    rng = np.random.default_rng(seed)
    boot = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
            if y[i].min() != y[i].max()]
    return roc_auc_score(y, p), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# ---------------------------------------------------------------------------------------------
# Figures: monochrome + one accent. The supervised model (the subject of the paper) is the only
# coloured element; zero-shot models are neutral grey and identified by direct labels.
# ---------------------------------------------------------------------------------------------
ACCENT = "#2a78d6"
INK, GREY, LIGHT, GRID = "#1f1f1e", "#8a8984", "#c9c8c2", "#ecebe7"

PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": LIGHT, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": GREY, "ytick.color": GREY, "text.color": INK,
    "xtick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 0, "xtick.minor.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titleweight": "regular", "axes.titlelocation": "left", "axes.titlepad": 6,
    "pdf.fonttype": 42, "svg.fonttype": "none",  # editable text in PDF/SVG
}


def _format_n(value):
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def _spread(positions, min_gap):
    """Nudge label y-positions apart so none are closer than min_gap (keeps order)."""
    order = np.argsort(positions)
    placed = np.array(positions, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    result = np.empty_like(placed)
    result[order] = placed - (placed.mean() - np.mean(positions))
    return result


def _right_labels(ax, labels, min_gap_fraction=0.07):
    """labels: list of (text, y, colour). Drawn just right of the axes, de-overlapped."""
    texts, ys, colours = zip(*labels)
    low, high = ax.get_ylim()
    for text, y, colour in zip(texts, _spread(ys, min_gap_fraction * (high - low)), colours):
        ax.annotate(text, xy=(1, y), xycoords=("axes fraction", "data"), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=colour,
                    annotation_clip=False)


def _style_n_axis(ax, all_n, tick_step=2):
    ax.set_xscale("log")
    ticks = all_n[::tick_step] if len(all_n) > 6 else all_n
    ax.set_xticks(ticks, [_format_n(n) for n in ticks])
    ax.set_xlim(all_n[0] / 1.3, all_n[-1] * 1.3)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def _draw_supervised(ax, scores, metric, calibrated, show_repeats):
    """Accent median line + IQR band; optionally every repeat as a faint dot."""
    curve = summarise(scores, metric, calibrated)
    if show_repeats:
        pts = scores[scores["calibrated"] == calibrated]
        jitter = np.exp(np.random.default_rng(0).uniform(-0.06, 0.06, len(pts)))  # small horizontal jitter
        ax.scatter(pts["n"] * jitter, pts[metric], s=4, color=ACCENT, alpha=0.18, lw=0, zorder=2)
    ax.fill_between(curve["n"], curve["low"], curve["high"], color=ACCENT, alpha=0.16, lw=0, zorder=1)
    ax.plot(curve["n"], curve["median"], color=ACCENT, lw=1.8, zorder=4,
            marker="o", ms=3, mfc=ACCENT, mec="white", mew=0.6)
    return curve


def _save(fig, path):
    if path:
        fig.savefig(f"{path}.pdf", bbox_inches="tight")
        fig.savefig(f"{path}.png", dpi=800, bbox_inches="tight")


def plot_discrimination(scores_by_model, supervised="XGBoost", path=None, figsize=(4.4, 2.8), show_repeats=True):
    """AUROC vs n. Supervised model: median + IQR across repeats (accent).
    Zero-shot models: their AUROC does not depend on n, so a flat grey reference line each, labelled."""
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        all_n = sorted(scores_by_model[supervised]["n"].unique())
        labels = []
        for name, scores in scores_by_model.items():
            if name == supervised:
                continue
            if not is_constant_in_n(scores):
                raise ValueError(f"{name}: AUROC varies with n; it is not a zero-shot run.")
            auroc = scores.loc[~scores["calibrated"], "auroc"].iloc[0]
            ax.axhline(auroc, color=GREY, lw=0.9, zorder=3)
            labels.append((f"{name} {auroc:.2f}", auroc, GREY))

        curve = _draw_supervised(ax, scores_by_model[supervised], "auroc", False, show_repeats)
        labels.append((supervised, curve["median"].iloc[-1], ACCENT))

        ax.axhline(0.5, color=GREY, ls=(0, (1, 2)), lw=0.8, zorder=1)
        _style_n_axis(ax, all_n)
        ax.set_xlabel("Labelled CTs (n)")
        ax.set_ylabel("AUROC")
        ax.margins(y=0.06)
        _right_labels(ax, labels)
        fig.tight_layout()
        _save(fig, path)
    return fig


def plot_calibration(scores_by_model, prevalence, supervised="XGBoost", path=None,
                     panel_size=(1.75, 2.0), skill=True, show_repeats=False):
    """Small multiples, one panel per zero-shot model: supervised (accent) vs that zero-shot model (grey),
    both after Platt scaling, median + IQR across repeats. Shared y axis.
    skill=True plots the Brier skill score 1 - Brier / (p(1-p)); 0 = predicting the prevalence."""
    reference = prevalence * (1 - prevalence)
    metric_name = "Brier skill score" if skill else "Brier score"

    def transform(scores):
        scores = scores.copy()
        if skill:
            scores["brier"] = 1 - scores["brier"] / reference
        return scores

    zero_shot = [m for m in scores_by_model if m != supervised]
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, len(zero_shot), sharey=True, squeeze=False,
                                 figsize=(panel_size[0] * len(zero_shot), panel_size[1]))
        all_n = sorted(scores_by_model[supervised]["n"].unique())
        sup = transform(scores_by_model[supervised])
        for ax, name in zip(axes[0], zero_shot):
            zs = summarise(transform(scores_by_model[name]), "brier", calibrated=True)
            ax.fill_between(zs["n"], zs["low"], zs["high"], color=GREY, alpha=0.18, lw=0, zorder=1)
            ax.plot(zs["n"], zs["median"], color=GREY, lw=1.2, zorder=3)
            _draw_supervised(ax, sup, "brier", True, show_repeats)
            ax.axhline(0 if skill else reference, color=GREY, ls=(0, (1, 2)), lw=0.8, zorder=0)
            _style_n_axis(ax, all_n, tick_step=4)
            ax.set_title(name)
        axes[0][0].set_ylabel(f"{metric_name}")
        fig.supxlabel("Labelled CTs (n)", fontsize=8, y=0.06)

        # Identity once, as direct labels at the line ends of the last panel
        last, last_zs = axes[0][-1], summarise(transform(scores_by_model[zero_shot[-1]]), "brier", True)
        sup_curve = summarise(sup, "brier", True)
        _right_labels(last, [(supervised, sup_curve["median"].iloc[-1], ACCENT),
                             ("Zero-shot", last_zs["median"].iloc[-1], GREY)], min_gap_fraction=0.12)
        fig.tight_layout(w_pad=1.2)
        _save(fig, path)
    return fig


# ---------------------------------------------------------------------------------------------
# Fallback (unscored CTs) reporting: CTs listed in Results/<model>/<model>_failed.json received
# the training prevalence instead of a model score. Needed for Methods and a sensitivity check.
# ---------------------------------------------------------------------------------------------
def load_failed_ids(cache_dir):
    """IDs that received the prevalence fallback (empty set if the model had none)."""
    name = os.path.basename(os.path.normpath(cache_dir))
    path = os.path.join(cache_dir, f"{name}_failed.json")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(json.load(f)["failed"])


def fallback_report(name, predictions, failed_ids):
    """Counts of fallback CTs in the test set, their label rate, and raw AUROC with/without them."""
    test = predictions[~predictions["calibrated"]].drop_duplicates("nctId")
    is_fallback = test["nctId"].isin(failed_ids)
    kept = test[~is_fallback]
    return {
        "model": name,
        "fallback_total": len(failed_ids),
        "fallback_test": int(is_fallback.sum()),
        "positive_rate_fallback_test": test.loc[is_fallback, "label"].mean() if is_fallback.any() else np.nan,
        "auroc_all": roc_auc_score(test["label"], test["proba"]),
        "auroc_without_fallback": roc_auc_score(kept["label"], kept["proba"]),
    }


def latest_run(prefix, results_dir="."):
    """Most recent run folder named <prefix>_YYYY-mm-dd_HHMMSS (timestamps sort as text)."""
    import glob
    runs = sorted(glob.glob(os.path.join(results_dir, f"{prefix}_20*")))
    if not runs:
        raise FileNotFoundError(f"No run folder matching {prefix}_20* in {os.path.abspath(results_dir)}")
    return runs[-1]
"""Task 6 — validate the Matchup Win Score against PFF pressure labels.

This module answers the project's credibility question: does the geometric
``win_score`` actually track real pass-protection outcomes? It provides:

* :func:`calibrate_weights` — fit an interpretable logistic model predicting
  ``pressure_allowed`` from the control features, returning raw-feature weights.
* :func:`validation_report` — numeric-only AUC / Pearson-correlation / count report.
* :func:`plot_validation` — an optional ROC + score-separation figure for the demo.

Sign convention (critical)
---------------------------
A higher ``win_score`` means the blocker WON the rep, which means LESS pressure.
Therefore:

* The ROC/AUC treats ``-win_score`` as the predictor of ``pressure_allowed`` (so a
  larger predictor => more pressure), giving an AUC in ``[0, 1]`` where ``> 0.5``
  means the score separates pressures from clean reps in the correct direction.
* The Pearson correlation ``corr(win_score, pressure_allowed)`` is expected to be
  NEGATIVE for a working score.

All functions return dicts / figures; nothing is printed outside ``__main__``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from matchup.config import _make_synthetic_scored

#: Control-feature columns fed to the logistic model. These are the ``features``
#: keys minus the id/label columns — the signals that describe the rep geometry.
DEFAULT_FEATURE_KEYS: list[str] = [
    "ground_given_up",
    "min_rusher_to_qb_dist",
    "penetration_past_los",
    "sep_mean",
    "sep_min",
    "betweenness_mean",
    "betweenness_end",
    "mirroring_mean",
    "rusher_speed_late",
    "rusher_accel_late",
    "rep_duration_sec",
]

_LABEL_COL = "pressure_allowed"


def _usable_feature_frame(
    df: pd.DataFrame, feature_keys: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Return rows usable for fitting plus the feature keys actually present.

    Filters to ``pairing_ok == True`` when that column exists, drops rows with any
    non-finite feature value or a missing label, and keeps counts honest by only
    removing rows that genuinely cannot be modelled.
    """
    keys = [k for k in feature_keys if k in df.columns]
    work = df.copy()
    if "pairing_ok" in work.columns:
        work = work[work["pairing_ok"].astype(bool)]
    if _LABEL_COL in work.columns:
        work = work[work[_LABEL_COL].notna()]
    if keys:
        finite_mask = np.isfinite(work[keys].to_numpy(dtype=float)).all(axis=1)
        work = work[finite_mask]
    return work, keys


def calibrate_weights(scored_or_features: pd.DataFrame) -> dict:
    """Fit a logistic model of ``pressure_allowed`` and return raw-feature weights.

    A :class:`~sklearn.preprocessing.StandardScaler` is fit on the control features
    and a balanced :class:`~sklearn.linear_model.LogisticRegression` is trained to
    predict ``pressure_allowed``. The scaler is then folded algebraically into the
    coefficients so the returned weights apply directly to RAW (unscaled) feature
    values::

        logit(pressure) = intercept_raw + sum_k w_raw[k] * x_raw[k]
        w_raw[k]        = coef[k] / scale[k]
        intercept_raw   = intercept - sum_k coef[k] * mean[k] / scale[k]

    Note
    ----
    These weights predict PRESSURE (a blocker *loss*). To drive ``score.win_score``
    (higher = blocker won) the caller must negate the linear predictor, because a
    higher probability of pressure corresponds to a lower win score.

    Parameters
    ----------
    scored_or_features:
        A ``scored`` or ``features``+label frame. Must contain ``pressure_allowed``
        and at least one of :data:`DEFAULT_FEATURE_KEYS`. ``pairing_ok`` is honoured
        when present; non-finite rows and missing labels are dropped.

    Returns
    -------
    dict
        ``{feature_key: weight, ..., "intercept": intercept_raw}``. Keys match the
        feature set actually present in the input. Returns an all-zero mapping with
        a nan intercept if the data is degenerate (single class or too few rows).
    """
    work, keys = _usable_feature_frame(scored_or_features, DEFAULT_FEATURE_KEYS)
    weights: dict = {k: 0.0 for k in keys}
    weights["intercept"] = float("nan")

    if not keys or len(work) < 2 or _LABEL_COL not in work.columns:
        return weights
    y = work[_LABEL_COL].to_numpy(dtype=int)
    if np.unique(y).size < 2:
        # Only one class present: cannot fit a discriminative model.
        return weights

    x = work[keys].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(x_std, y)

    coef = model.coef_.ravel()
    scale = scaler.scale_
    mean = scaler.mean_
    w_raw = coef / scale
    intercept_raw = float(model.intercept_[0] - np.sum(coef * mean / scale))

    weights = {k: float(w_raw[i]) for i, k in enumerate(keys)}
    weights["intercept"] = intercept_raw
    return weights


def validation_report(scored: pd.DataFrame) -> dict:
    """Report how well ``win_score`` separates pressures from clean reps.

    Computes, on finite rows that have both ``win_score`` and ``pressure_allowed``
    (and ``pairing_ok == True`` when that column exists):

    * ``auc`` — ``roc_auc_score(pressure_allowed, -win_score)``. The score is negated
      because a higher ``win_score`` means the blocker won (less pressure); negating
      makes a larger predictor correspond to more pressure, so ``auc > 0.5`` means
      the score orders reps correctly.
    * ``corr`` — Pearson ``corr(win_score, pressure_allowed)``, expected NEGATIVE.
    * ``n`` — number of reps used.

    Degenerate inputs (fewer than 2 rows, or only one label class) yield ``nan``
    metrics plus a ``note`` key explaining why. Numeric-only: no plotting here.

    Parameters
    ----------
    scored:
        A ``scored``-schema frame.

    Returns
    -------
    dict
        ``{"auc": float, "corr": float, "n": int}`` and, when degenerate, ``"note"``.
    """
    work = scored
    if "pairing_ok" in work.columns:
        work = work[work["pairing_ok"].astype(bool)]
    needed = ["win_score", _LABEL_COL]
    if not all(c in work.columns for c in needed):
        return {
            "auc": float("nan"),
            "corr": float("nan"),
            "n": 0,
            "note": "missing win_score or pressure_allowed column",
        }

    ws = pd.to_numeric(work["win_score"], errors="coerce")
    y = pd.to_numeric(work[_LABEL_COL], errors="coerce")
    mask = np.isfinite(ws) & np.isfinite(y)
    ws = ws[mask].to_numpy(dtype=float)
    y = y[mask].to_numpy(dtype=int)
    n = int(ws.size)

    if n < 2:
        return {"auc": float("nan"), "corr": float("nan"), "n": n, "note": "fewer than 2 usable rows"}
    if np.unique(y).size < 2:
        return {
            "auc": float("nan"),
            "corr": float("nan"),
            "n": n,
            "note": "only one pressure_allowed class present",
        }

    auc = float(roc_auc_score(y, -ws))  # negate: higher win_score => less pressure
    # Pearson correlation; guard a zero-variance predictor.
    if np.std(ws) == 0.0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(ws, y)[0, 1])
    return {"auc": auc, "corr": corr, "n": n}


def plot_validation(scored: pd.DataFrame) -> matplotlib.figure.Figure:
    """Build a two-panel validation figure for the demo (ROC + score separation).

    Left panel: ROC curve of ``-win_score`` predicting ``pressure_allowed`` with the
    AUC annotated. Right panel: ``win_score`` boxplots split by ``pressure_allowed``
    (clean vs pressure) so the separation is visible. Returns the Figure without
    showing it. Degenerate inputs produce a figure carrying an explanatory message.

    Parameters
    ----------
    scored:
        A ``scored``-schema frame.

    Returns
    -------
    matplotlib.figure.Figure
    """
    report = validation_report(scored)
    fig, (ax_roc, ax_box) = plt.subplots(1, 2, figsize=(11, 4.5))

    if not np.isfinite(report.get("auc", float("nan"))):
        msg = report.get("note", "insufficient data for validation")
        for ax in (ax_roc, ax_box):
            ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Win Score validation (degenerate input)")
        fig.tight_layout()
        return fig

    work = scored
    if "pairing_ok" in work.columns:
        work = work[work["pairing_ok"].astype(bool)]
    ws = pd.to_numeric(work["win_score"], errors="coerce")
    y = pd.to_numeric(work[_LABEL_COL], errors="coerce")
    mask = np.isfinite(ws) & np.isfinite(y)
    ws = ws[mask].to_numpy(dtype=float)
    y = y[mask].to_numpy(dtype=int)

    # ROC panel.
    fpr, tpr, _ = roc_curve(y, -ws)
    ax_roc.plot(fpr, tpr, color="C0", lw=2, label=f"AUC = {report['auc']:.3f}")
    ax_roc.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="chance")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC: -win_score predicts pressure_allowed")
    ax_roc.legend(loc="lower right")

    # Separation panel.
    clean = ws[y == 0]
    pressure = ws[y == 1]
    ax_box.boxplot([clean, pressure], tick_labels=["clean (0)", "pressure (1)"])
    ax_box.set_ylabel("win_score")
    ax_box.set_title(f"win_score by outcome (corr = {report['corr']:.3f})")

    fig.suptitle("Matchup Win Score validation vs PFF labels")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    scored = _make_synthetic_scored(n_reps=400, seed=0)

    print("validate.py demo")
    print("  scored shape:", scored.shape)

    weights = calibrate_weights(scored)
    print("  calibrated weights (raw-feature, predict pressure):")
    for k, v in weights.items():
        print(f"    {k:24s} {v:+.4f}")

    report = validation_report(scored)
    print("  validation_report:", report)
    assert report["auc"] > 0.5, f"AUC should exceed 0.5, got {report['auc']}"
    assert report["corr"] < 0.0, f"corr should be negative, got {report['corr']}"
    print(f"  OK: AUC={report['auc']:.3f} (>0.5), corr={report['corr']:.3f} (<0)")

    fig = plot_validation(scored)
    from matchup.config import CACHE_DIR

    png = CACHE_DIR / "validation.png"
    fig.savefig(png, dpi=120)
    print("  saved:", png)

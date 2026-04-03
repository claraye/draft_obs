"""
classifier.py — XGBoost ensemble classifier and reclassification methods
Paper: Han & Qin (2026), SSRN 4452964

Implements:
  1. assign_return_classes()     — label each stock-month with return decile 1-10
  2. train_xgb_ensemble()        — train N_RUNS XGBoost classifiers, return avg probabilities
  3. reclassify_DPR()            — reclassify by probability difference (Eq. 11)
  4. reclassify_RET()            — reclassify by predicted expected return (Eq. 9-10, 12)
  5. reclassify_SRP()            — reclassify by predicted Sharpe ratio (Eq. 13-14)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ─── Return class labelling ───────────────────────────────────────────────────

def assign_return_classes(ret_series: pd.Series, n_classes: int = 10) -> pd.Series:
    """
    Assign cross-sectional return decile labels 0..9 (0=worst, 9=best).
    Returns NaN for stocks with missing forward returns.
    """
    valid = ret_series.dropna()
    if len(valid) < n_classes:
        return pd.Series(np.nan, index=ret_series.index)
    labels = pd.qcut(valid, q=n_classes, labels=False, duplicates="drop")
    return labels.reindex(ret_series.index)


# ─── XGBoost ensemble ─────────────────────────────────────────────────────────

def train_xgb_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    n_runs:  int,
    xgb_params: dict,
    early_stopping_rounds: int = 20,
) -> np.ndarray:
    """
    Train `n_runs` XGBoost classifiers with different random seeds.
    Average the predicted class probabilities across runs.

    Returns
    -------
    np.ndarray  shape (len(X_train) + len(X_val) equivalent at predict time, n_classes)
        This function returns the predicted probabilities on X_val only,
        for diagnostic purposes. The caller should pass the predict set separately.

    Note: This function is called once per training date. X_train/y_train is
    the training set; X_val/y_val is used for early stopping only.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("Install xgboost: pip install xgboost")

    n_classes = len(np.unique(y_train))
    all_proba = []

    for run in range(n_runs):
        params = dict(xgb_params)
        params["random_state"] = run  # different seed per run

        clf = XGBClassifier(**params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # Predict on training+val combined is not needed here;
        # we return the model to predict on the inference set later.
        all_proba.append(clf)

    return all_proba   # list of fitted models


def predict_ensemble(models: list, X: np.ndarray) -> np.ndarray:
    """
    Average predicted class probabilities across all ensemble models.

    Returns
    -------
    np.ndarray  shape (N, n_classes)  averaged probabilities
    """
    prob_sum = None
    for clf in models:
        p = clf.predict_proba(X)
        if prob_sum is None:
            prob_sum = p.copy()
        else:
            prob_sum += p
    return prob_sum / len(models)


# ─── Reclassification ─────────────────────────────────────────────────────────

def reclassify_DPR(proba: np.ndarray) -> np.ndarray:
    """
    Reclassification on probability difference (DPR), Eq. 11.
    Does not require class mean returns.

    Score_i = Σ_{k=1}^{10/2} (ŷ_{i,k} − ŷ_{i,11-k}) × (10+1−k)/2 + (1-k)
    (proportional to expected return under linear class means)

    Parameters
    ----------
    proba : np.ndarray  shape (N, 10)  class probabilities (0-indexed: col 0 = class 1)

    Returns
    -------
    np.ndarray  shape (N,)  DPR scores (higher = more likely winner)
    """
    N, K = proba.shape
    scores = np.zeros(N)
    half = K // 2
    for k in range(1, half + 1):  # k = 1,...,5
        # ŷ_{k} − ŷ_{11-k}, weight = (10+1-k)/2
        weight  = (K + 1 - k) / 2.0
        scores += (proba[:, k - 1] - proba[:, K - k]) * weight
    return scores


def reclassify_RET(proba: np.ndarray, class_means: np.ndarray) -> np.ndarray:
    """
    Reclassification on expected return (RET), Eq. 9–10, 12.

    µ̂_i = Σ_k ŷ_{i,k} × µ̂^k

    Parameters
    ----------
    proba       : np.ndarray  shape (N, K)   class probabilities
    class_means : np.ndarray  shape (K,)     historical mean return of each class

    Returns
    -------
    np.ndarray  shape (N,)  predicted expected return
    """
    return proba @ class_means


def reclassify_SRP(proba: np.ndarray, class_means: np.ndarray, class_stds: np.ndarray) -> np.ndarray:
    """
    Reclassification on Sharpe ratio (SRP), Eq. 13–14.

    σ̂²_i = Σ_k ŷ_{i,k} × (σ̂²^k + µ̂^{k²}) − µ̂²_i
    SR_i  = µ̂_i / σ̂_i

    Parameters
    ----------
    proba       : np.ndarray  shape (N, K)
    class_means : np.ndarray  shape (K,)
    class_stds  : np.ndarray  shape (K,)

    Returns
    -------
    np.ndarray  shape (N,)  predicted Sharpe ratio
    """
    mu_i  = proba @ class_means
    var_i = proba @ (class_stds**2 + class_means**2) - mu_i**2
    var_i = np.maximum(var_i, 1e-12)  # numerical floor
    return mu_i / np.sqrt(var_i)


# ─── Class statistics ─────────────────────────────────────────────────────────

def estimate_class_statistics(
    feature_df: pd.DataFrame,
    lookback_years: int = 10,
    n_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the rolling mean (µ̂^k) and std (σ̂^k) of each return class
    over the past `lookback_years` of training data.

    Parameters
    ----------
    feature_df : pd.DataFrame  with columns ['date', 'ret_fwd', 'class_label']
    lookback_years : int       how many years of history to use

    Returns
    -------
    class_means : np.ndarray  shape (n_classes,)
    class_stds  : np.ndarray  shape (n_classes,)
    """
    class_means = np.zeros(n_classes)
    class_stds  = np.ones(n_classes)

    for k in range(n_classes):
        mask  = feature_df["class_label"] == k
        rets  = feature_df.loc[mask, "ret_fwd"].dropna()
        if len(rets) > 5:
            class_means[k] = rets.mean()
            class_stds[k]  = rets.std()

    return class_means, class_stds

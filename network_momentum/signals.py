"""
signals.py — Network momentum feature propagation, OLS regression, and trading signals.

Paper Section 3.2 and 4.1:
  Network momentum feature (Eq.7):  ũ_{i,t} = Σ_{j∈N(i)} Ã_{ij,t} · u_{j,t}
  OLS regression (Eq.8):             y_{i,t} = ũ_{i,t}^T β + b
  Position:                           x_{i,t} = sign(y_{i,t})
  MACD position (Eq.10):             x_{i,t} = (1/3) Σ_k φ(y_{i,t}(S_k,L_k))
                                     φ(y) = y·exp(-y²/4) / 0.89
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ─── Feature propagation ──────────────────────────────────────────────────────

def propagate_features(
    u_t: np.ndarray,
    A_tilde: np.ndarray,
) -> np.ndarray:
    """
    Network momentum feature propagation (Eq. 7):
      ũ_{i,t} = Σ_{j ∈ N_t(i)} Ã_{ij,t} · u_{j,t}

    Args:
      u_t:     (N, 8) individual momentum features for assets in graph
      A_tilde: (N, N) normalised adjacency matrix

    Returns:
      ũ_t:     (N, 8) network momentum features
    """
    return A_tilde @ u_t  # (N, 8)


# ─── OLS regression ───────────────────────────────────────────────────────────

def ols_train(
    X: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """
    Fit cross-sectional OLS with optional L2 regularisation.
    Paper uses standard OLS (Eq. 8): y_{i,t} = ũ_{i,t}^T β + b

    Args:
      X: (M, 8) stacked network momentum features (all assets × all training days)
      y: (M,)   stacked targets (vol-scaled next-day returns)
      ridge_lambda: small ridge penalty for numerical stability

    Returns:
      beta: (8,) coefficient vector
      intercept: float
    """
    # Add intercept column
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack([X, ones])

    # Ridge: (X^T X + λI)^{-1} X^T y
    lam = ridge_lambda * np.eye(X_aug.shape[1])
    lam[-1, -1] = 0  # don't regularise intercept
    try:
        coef = np.linalg.solve(X_aug.T @ X_aug + lam, X_aug.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(X_aug, y, rcond=None)[0]

    beta      = coef[:-1]  # (8,)
    intercept = coef[-1]
    return beta, float(intercept)


def ols_predict(
    X: np.ndarray,
    beta: np.ndarray,
    intercept: float,
) -> np.ndarray:
    """
    Predict y_{i,t} = ũ_{i,t}^T β + b.

    Args:
      X: (N, 8) network momentum features for N assets
    Returns:
      y: (N,) predicted signals
    """
    return X @ beta + intercept


# ─── Individual momentum OLS (LinReg baseline) ────────────────────────────────

def linreg_train(
    u_train: np.ndarray,
    y_train: np.ndarray,
    ridge_lambda: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Same as ols_train but for individual momentum features (LinReg baseline)."""
    return ols_train(u_train, y_train, ridge_lambda)


# ─── MACD signal (model-free baseline, Eq. 10) ───────────────────────────────

def response_function(y: np.ndarray) -> np.ndarray:
    """
    Position scaling function φ(y) = y · exp(-y²/4) / 0.89  (Baz et al.)
    Maps a momentum signal y to a bounded position in (-1, 1).
    """
    return y * np.exp(-y ** 2 / 4) / 0.89


def macd_signal(
    feature_df: pd.DataFrame,
    assets: list[str],
    date: pd.Timestamp,
) -> np.ndarray:
    """
    MACD signal (Eq. 10): x_{i,t} = (1/3) Σ_k φ(y(S_k, L_k))

    Returns:
      positions: (N,) array in approximately (-1, 1)
    """
    from config import MACD_PAIRS
    N = len(assets)
    signal_sum = np.zeros(N)
    n_valid = 0

    for (S, L) in MACD_PAIRS:
        feat_name = f"macd_{S}_{L}"
        if feat_name not in feature_df.columns.get_level_values(0):
            continue
        y_row = feature_df.loc[date, feat_name].reindex(assets).values
        y_row = np.where(np.isfinite(y_row), y_row, 0.0)
        signal_sum += response_function(y_row)
        n_valid += 1

    if n_valid == 0:
        return np.zeros(N)

    return signal_sum / n_valid  # average across 3 MACD speeds


# ─── Signal assembly ──────────────────────────────────────────────────────────

def build_training_data(
    feature_tensor: np.ndarray,      # (T, N, 8)
    returns_1d: np.ndarray,          # (T, N) vol-scaled next-day returns (target)
    sigma: np.ndarray,               # (T, N) daily vol
    A_tilde_series: dict,            # {t_idx: (N_valid, N_valid) adjacency, valid_idx: array}
    train_t_indices: list[int],
    mode: str = "gmom",              # "gmom" | "linreg"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack cross-sectional features and targets over the training period.

    Args:
      mode: "gmom" → use network momentum features;
            "linreg" → use individual momentum features

    Returns:
      X_train: (M, 8)
      y_train: (M,)
    """
    X_list, y_list = [], []

    for t in train_t_indices:
        if t + 1 >= feature_tensor.shape[0]:
            continue

        # Target: vol-scaled 1-day return at t+1
        target = returns_1d[t + 1, :]  # (N,)

        if mode == "gmom":
            if t not in A_tilde_series:
                continue
            A_tilde, valid_idx = A_tilde_series[t]
            if A_tilde is None or len(valid_idx) < 5:
                continue

            u_t = feature_tensor[t, valid_idx, :]   # (N_valid, 8)
            u_net = propagate_features(u_t, A_tilde) # (N_valid, 8)
            y_t = target[valid_idx]                  # (N_valid,)

            mask = np.isfinite(u_net).all(axis=1) & np.isfinite(y_t)
            if mask.sum() < 3:
                continue

            X_list.append(u_net[mask])
            y_list.append(y_t[mask])

        elif mode == "linreg":
            u_t = feature_tensor[t, :, :]  # (N, 8)
            mask = np.isfinite(u_t).all(axis=1) & np.isfinite(target)
            if mask.sum() < 3:
                continue
            X_list.append(u_t[mask])
            y_list.append(target[mask])

    if not X_list:
        return None, None

    return np.vstack(X_list), np.concatenate(y_list)

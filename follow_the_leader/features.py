"""
features.py — Volatility-scaled features and MACD oscillators

Paper: Section 2 — "Trend-Following Indicators"

Equations implemented:
  Δ̃_{t,m}  = Δ_{t,m} / σ^22_{t,m}                (vol-scaled return)
  P̃_m      = cumsum(Δ̃_{t,m})                      (vol-scaled cumulative price)
  R^k_{t,m} = EWMA(P̃_m, α_fast(k)) - EWMA(P̃_m, α_slow(k))  (oscillator)
  α_fast(k) = 1 / 2^k
  α_slow(k) = 1 / (M · 2^k)
  φ(x)     = x · exp(-x² / 4) / 0.89              (response function, Baz et al.)
"""
import numpy as np
import pandas as pd


def ewma_vol(returns: pd.DataFrame, span: int = 22) -> pd.DataFrame:
    """
    22-day EWMA standard deviation of log returns: σ^22_{t,m}.

    Args:
        returns: DataFrame(T, N) of log returns
        span:    EWMA span in trading days (default 22 ≈ 1 month)

    Returns:
        vol: DataFrame(T, N) of EWMA daily standard deviations
    """
    return returns.ewm(span=span, min_periods=span // 2).std()


def vol_scaled_delta(returns: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility-scaled return: Δ̃_{t,m} = r_{t,m} / σ^22_{t,m}.
    Removes cross-market scale differences before computing MACD.
    """
    return returns / (vol + 1e-10)


def vol_scaled_cumulative_price(
    returns: pd.DataFrame,
    vol: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vol-scaled cumulative price: P̃_m = cumsum(Δ̃_{t,m}).

    This is the series input to the MACD oscillator.
    Note: P̃ is NOT simply price/vol because σ^22 changes over time.
    """
    return vol_scaled_delta(returns, vol).cumsum()


def macd_oscillator(
    p_tilde: pd.DataFrame,
    k: int,
    M: int = 4,
) -> pd.DataFrame:
    """
    MACD-style oscillator at speed k on vol-scaled cumulative price:

      R^k_{t,m} = EWMA(P̃_m, α_fast(k)) - EWMA(P̃_m, α_slow(k))

    Smoothing factors:
      α_fast(k) = 1 / 2^k         → fast halflife ≈ 2^k trading days
      α_slow(k) = 1 / (M · 2^k)  → slow halflife ≈ M · 2^k trading days

    With M=4, k=1: fast=2d, slow=8d  (very short-term)
    With M=4, k=5: fast=32d, slow=128d (medium-term, ~6 months)

    Args:
        p_tilde: DataFrame(T, N) vol-scaled cumulative prices
        k:       speed parameter (integer ≥ 1)
        M:       slow/fast ratio (paper default 4)

    Returns:
        R_k: DataFrame(T, N) oscillator values
    """
    alpha_fast = 1.0 / (2 ** k)
    alpha_slow = 1.0 / (M * (2 ** k))
    fast = p_tilde.ewm(alpha=alpha_fast, adjust=False, min_periods=int(1.0 / alpha_slow)).mean()
    slow = p_tilde.ewm(alpha=alpha_slow, adjust=False, min_periods=int(1.0 / alpha_slow)).mean()
    return fast - slow


def build_oscillators(
    returns: pd.DataFrame,
    vol: pd.DataFrame,
    k_values: list = None,
    M: int = 4,
) -> dict:
    """
    Build all MACD oscillators at each speed k.

    Args:
        returns:  DataFrame(T, N) log returns
        vol:      DataFrame(T, N) EWMA daily vol
        k_values: list of integer speed parameters (default [1,2,3,4,5])
        M:        slow/fast ratio

    Returns:
        oscillators: dict {k: DataFrame(T, N)} — one oscillator per speed
    """
    if k_values is None:
        k_values = [1, 2, 3, 4, 5]
    p_tilde = vol_scaled_cumulative_price(returns, vol)
    return {k: macd_oscillator(p_tilde, k, M) for k in k_values}


def response_function(x: np.ndarray) -> np.ndarray:
    """
    Position response (squashing) function φ: ℝ → (-1, 1).

    Baz et al. (2015) definition used throughout the paper:
      φ(x) = x · exp(-x² / 4) / 0.89

    Properties:
      - φ(0) = 0: no position when signal is zero
      - |φ(x)| < 1: bounded positions (option-like payoff profile)
      - max |φ| ≈ 1 at |x| ≈ √2 ≈ 1.41; normalising by 0.89 ≈ max(x·exp(-x²/4))
      - For large |x|: φ → 0 (attenuates extreme signals for risk control)
      - For small |x|: φ ≈ x/0.89 (approximately linear near origin)

    Args:
        x: numpy array of raw oscillator values

    Returns:
        bounded signal values in (-1, 1)
    """
    return x * np.exp(-x ** 2 / 4.0) / 0.89


def oscillators_at_t(oscillators: dict, idx: int) -> dict:
    """Extract oscillator values at a single time index as dict {k: (N,) array}."""
    return {k: df.iloc[idx].values for k, df in oscillators.items()}


def oscillators_matrix_at_t(oscillators: dict, idx: int) -> np.ndarray:
    """
    Stack oscillators at time idx into a (N, K) matrix.
    Column k corresponds to speed k_values[k].
    """
    k_values = sorted(oscillators.keys())
    return np.column_stack([oscillators[k].iloc[idx].values for k in k_values])

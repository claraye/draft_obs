"""
features.py — Individual momentum feature construction
Paper Section 2.2: 8 features per asset per day.

Features:
  (1-5) Volatility-scaled returns: r_{t-delta:t} / (sigma_t * sqrt(delta)),  delta in {1,21,63,126,252}
  (6-8) Normalised MACD: y(S,L) for (S,L) in {(8,24),(16,48),(32,96)}

All features are winsorised at +/-5 EWMA std (252-day half-life).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from config import (
    VOL_RETURN_WINDOWS, MACD_PAIRS, VOL_SPAN,
    WINSOR_HALFLIFE, WINSOR_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─── Volatility-scaled returns ────────────────────────────────────────────────

def vol_scaled_returns(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    sigma: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """
    Compute r_{t-Δ:t} / (σ_t * √Δ) for each Δ in VOL_RETURN_WINDOWS.
    σ_t: daily EWMA vol (already computed), annualised = σ_t * sqrt(252).
    We use daily σ_t (not annualised) consistent with Eq.(1) of Baz et al.

    Returns dict: {Δ: DataFrame(index=Date, columns=assets)}
    """
    result = {}
    for delta in VOL_RETURN_WINDOWS:
        # Return over [t-Δ, t]: log(P_t / P_{t-Δ})
        ret_delta = np.log(prices / prices.shift(delta))
        # Scale by daily vol and horizon normalisation
        scaled = ret_delta / (sigma * np.sqrt(delta) + 1e-10)
        result[delta] = scaled
    return result


# ─── MACD features ────────────────────────────────────────────────────────────

def _ewma(series: pd.Series, alpha: float) -> pd.Series:
    """EWMA with explicit smoothing factor α (not span)."""
    return series.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def _ewma_df(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Column-wise EWMA with explicit smoothing factor α."""
    return df.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def normalised_macd(prices: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
    """
    Paper Eqs. (1–3):
      MACD(i,t,S,L) = m(i,t,S) - m(i,t,L)              [Eq.1]
        where m(i,t,J) = EWMA(prices_i, α=1/J)
      MACD_norm = MACD / std(prices_{t-63:t})            [Eq.2]
      y(S,L) = MACD_norm / std(MACD_norm_{t-252:t})      [Eq.3]

    Returns dict: {(S,L): DataFrame(index=Date, columns=assets)}
    """
    result = {}
    # 63-day rolling std of prices (for Eq.2 normalisation)
    price_std_63 = prices.rolling(window=63, min_periods=32).std()

    for (S, L) in MACD_PAIRS:
        alpha_fast = 1.0 / S
        alpha_slow = 1.0 / L

        fast_ma = _ewma_df(prices, alpha=alpha_fast)
        slow_ma = _ewma_df(prices, alpha=alpha_slow)

        macd = fast_ma - slow_ma                              # Eq.1
        macd_norm = macd / (price_std_63 + 1e-10)            # Eq.2

        # Eq.3: normalise by 252-day rolling std of macd_norm
        macd_norm_std = macd_norm.rolling(window=252, min_periods=126).std()
        y = macd_norm / (macd_norm_std + 1e-10)              # Eq.3

        result[(S, L)] = y

    return result


# ─── Winsorisation ────────────────────────────────────────────────────────────

def winsorise(
    features: pd.DataFrame,
    halflife: int = WINSOR_HALFLIFE,
    threshold: float = WINSOR_THRESHOLD,
) -> pd.DataFrame:
    """
    Cap and floor each feature to ±threshold × EWMA std (halflife-day half-life).
    Paper: 'each feature of each asset was capped and floored to fall within a range
    defined by five times its exponentially weighted moving standard deviations from
    its corresponding exponentially weighted moving average.'
    """
    ewm_kwargs = dict(halflife=halflife, adjust=False, min_periods=halflife // 2)
    mu  = features.ewm(**ewm_kwargs).mean()
    std = features.ewm(**ewm_kwargs).std()

    upper = mu + threshold * std
    lower = mu - threshold * std

    return features.clip(lower=lower, upper=upper)


# ─── Assemble feature matrix ──────────────────────────────────────────────────

def build_feature_matrix(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    sigma: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the full 8-feature matrix.

    Returns:
      DataFrame with MultiIndex columns: (feature_name, asset_ticker)
      Index: Date
      Shape: (T, 8*N) or equivalently stacked (T, N, 8) accessible via .values
    """
    feature_frames = {}

    # Vol-scaled returns
    vs_ret = vol_scaled_returns(prices, returns, sigma)
    for delta, df in vs_ret.items():
        feature_frames[f"vsr_{delta}d"] = df

    # Normalised MACD
    macd_dict = normalised_macd(prices)
    for (S, L), df in macd_dict.items():
        feature_frames[f"macd_{S}_{L}"] = df

    # Stack into MultiIndex columns: (feature, asset)
    combined = pd.concat(feature_frames, axis=1)  # columns: (feat, asset)

    # Winsorise each feature independently across all assets
    winsorised_parts = {}
    for feat_name in feature_frames:
        winsorised_parts[feat_name] = winsorise(combined[feat_name])

    result = pd.concat(winsorised_parts, axis=1)
    return result  # MultiIndex: (feature_name, asset)


def get_feature_tensor(
    feature_df: pd.DataFrame,
    assets: list[str],
    date_idx: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Extract feature matrix U_t ∈ R^(N × 8) for each date in date_idx.

    Returns:
      np.ndarray of shape (T, N, 8) where T=len(date_idx), N=len(assets)
    """
    feature_names = [f"vsr_{d}d" for d in VOL_RETURN_WINDOWS] + \
                    [f"macd_{S}_{L}" for S, L in MACD_PAIRS]

    T = len(date_idx)
    N = len(assets)
    K = len(feature_names)

    tensor = np.full((T, N, K), np.nan)
    for k, feat in enumerate(feature_names):
        if feat in feature_df.columns.get_level_values(0):
            sub = feature_df[feat].reindex(index=date_idx, columns=assets)
            tensor[:, :, k] = sub.values

    return tensor  # (T, N, 8)


def stack_lookback(
    tensor: np.ndarray,
    t_idx: int,
    delta: int,
    assets_available: np.ndarray,
) -> np.ndarray | None:
    """
    Stack feature tensors over a lookback window of δ days ending at t_idx.
    Returns V_t ∈ R^(N_avail × 8δ) or None if insufficient history.

    Only includes assets that have no NaN over the full lookback window.
    """
    if t_idx < delta:
        return None, assets_available

    window = tensor[t_idx - delta: t_idx, :, :]  # (δ, N, 8)

    # Find assets with complete data in this window
    has_full = ~np.isnan(window).any(axis=(0, 2))  # (N,) bool
    valid_idx = np.where(has_full & assets_available)[0]

    if len(valid_idx) < 5:  # Need at least 5 assets
        return None, valid_idx

    sub = window[:, valid_idx, :]           # (δ, N_valid, 8)
    V = sub.transpose(1, 0, 2).reshape(len(valid_idx), -1)  # (N_valid, 8δ)
    return V, valid_idx

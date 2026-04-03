"""
features.py — Feature construction for Deep Momentum replication
Paper: Han & Qin (2026), SSRN 4452964

Constructs the 16-feature set per Section 3.3.1:
  - 5 normalized m-month momentum features: zMOM^m_i   (Eq. 6–8)
  - 5 cross-sectional means:                mMOM^m
  - 5 cross-sectional standard deviations:  sMOM^m
  - 1 size decile:                          SIZE_i  (categorical 1–10)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

MOM_WINDOWS = [1, 3, 6, 9, 12]   # months (paper Eq. 6–7)


def compute_momentum(returns: pd.DataFrame, m: int) -> pd.DataFrame:
    """
    Compute m-month momentum for each stock-month.

    For m == 1: MOM^1_i = r_{i,t}  (previous one-month return)
    For m >  1: MOM^m_i = ∏_{j=t-m+1}^{t-1} (1 + r_{i,j}) − 1
                         i.e. cumulative return over months [t-m+1, t-1]

    The paper uses a 1-month lag on the formation period, so the most recent
    monthly return (month t) is excluded for m > 1.

    Parameters
    ----------
    returns : pd.DataFrame  (T, N)  monthly returns
    m       : int           lookback in months

    Returns
    -------
    pd.DataFrame  (T, N)  m-month momentum, NaN during burn-in
    """
    if m == 1:
        # Short-term reversal feature: previous one-month return
        return returns.copy()
    else:
        # Compound return over [t-m+1 ... t-1]  (skip most-recent month = t)
        log_ret = np.log1p(returns.fillna(0))
        rolled  = log_ret.shift(1).rolling(m - 1).sum()   # skip month t, sum m-1 months
        return np.expm1(rolled)


def build_features(returns: pd.DataFrame, size_proxy: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full 16-feature matrix for all stock-months.

    Returns
    -------
    pd.DataFrame  shape (T * N, 16 + 2)
        Multi-indexed by (Date, Ticker) or stacked as (T*N, 18) with columns:
          zMOM1m, zMOM3m, zMOM6m, zMOM9m, zMOM12m,
          mMOM1m, mMOM3m, mMOM6m, mMOM9m, mMOM12m,
          sMOM1m, sMOM3m, sMOM6m, sMOM9m, sMOM12m,
          SIZE,
          ret_fwd  (one-month forward return — target)
          date, ticker (index)
    """
    records = []

    tickers = returns.columns.tolist()
    dates   = returns.index

    # Pre-compute raw momentum matrices for all windows
    mom_raw = {m: compute_momentum(returns, m) for m in MOM_WINDOWS}

    for t_idx, date in enumerate(dates):
        if t_idx == 0:
            continue  # need at least one lagged observation

        # ── Raw momentum values at time t ────────────────────────────────
        mom_t = {}
        for m in MOM_WINDOWS:
            mom_t[m] = mom_raw[m].iloc[t_idx]   # Series (N,)

        # ── Cross-sectional standardization (Eq. 8) ──────────────────────
        zmom_t = {}
        mmom_t = {}
        smom_t = {}
        for m in MOM_WINDOWS:
            vals = mom_t[m].dropna()
            if len(vals) < 5:
                continue
            mu  = vals.mean()
            std = vals.std()
            mmom_t[m] = mu
            smom_t[m] = std
            if std > 1e-8:
                zmom_t[m] = (mom_t[m] - mu) / std
            else:
                zmom_t[m] = mom_t[m] * 0.0

        if len(zmom_t) < len(MOM_WINDOWS):
            continue  # skip if any window has insufficient data

        # ── Size decile ───────────────────────────────────────────────────
        mcap = size_proxy.iloc[t_idx]
        valid_mcap = mcap.dropna()
        if len(valid_mcap) < 10:
            size_decile = pd.Series(np.nan, index=tickers)
        else:
            # Assign decile 1 (small) to 10 (large) based on cross-sectional rank
            size_decile = valid_mcap.rank(pct=True)
            size_decile = np.ceil(size_decile * 10).clip(1, 10)
            size_decile = size_decile.reindex(tickers)

        # ── Forward return (target) ───────────────────────────────────────
        if t_idx + 1 < len(dates):
            ret_fwd = returns.iloc[t_idx + 1]
        else:
            ret_fwd = pd.Series(np.nan, index=tickers)

        # ── Assemble per-stock rows ───────────────────────────────────────
        for ticker in tickers:
            row = {"date": date, "ticker": ticker}
            for m in MOM_WINDOWS:
                row[f"zMOM{m}m"] = zmom_t[m].get(ticker, np.nan)
                row[f"mMOM{m}m"] = mmom_t[m]
                row[f"sMOM{m}m"] = smom_t[m]
            row["SIZE"]    = size_decile.get(ticker, np.nan)
            row["ret_fwd"] = ret_fwd.get(ticker, np.nan)
            records.append(row)

    df = pd.DataFrame(records)
    df = df.dropna(subset=[f"zMOM{m}m" for m in MOM_WINDOWS] + ["SIZE"])
    return df


FEATURE_COLS = (
    [f"zMOM{m}m" for m in MOM_WINDOWS]
    + [f"mMOM{m}m" for m in MOM_WINDOWS]
    + [f"sMOM{m}m" for m in MOM_WINDOWS]
    + ["SIZE"]
)
assert len(FEATURE_COLS) == 16, f"Expected 16 features, got {len(FEATURE_COLS)}"

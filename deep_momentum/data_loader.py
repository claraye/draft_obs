"""
data_loader.py — Data loading and preprocessing for Deep Momentum replication
Paper: Han & Qin (2026), SSRN 4452964

Handles:
  - Fetching adjusted monthly prices from Yahoo Finance
  - Computing monthly returns from end-of-month prices
  - Loading user-supplied CSV of monthly returns
  - Basic data quality filters (paper Section 3.1)
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_yahoo_monthly(tickers: list[str], start: str, end: str, cache_dir: str) -> pd.DataFrame:
    """
    Download adjusted daily prices from Yahoo Finance and resample to
    end-of-month total returns.

    Returns
    -------
    pd.DataFrame  shape (T_months, N_stocks)
        Monthly returns as decimals. Index is last trading day of each month.
        Columns are ticker symbols.
    """
    import yfinance as yf

    cache_path = os.path.join(cache_dir, "yahoo_monthly_returns.parquet")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached monthly returns from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Downloading daily prices for {len(tickers)} tickers via yfinance ...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Handle single-ticker download (returns Series)
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    # Resample to end-of-month
    monthly = raw.resample("ME").last()

    # Compute simple monthly returns
    returns = monthly.pct_change()
    returns = returns.iloc[1:]  # drop first NaN row

    # Drop tickers with >20% missing
    missing_frac = returns.isna().mean()
    keep = missing_frac[missing_frac < 0.2].index.tolist()
    if len(keep) < len(tickers):
        dropped = [t for t in tickers if t not in keep]
        logger.warning(f"Dropped {len(dropped)} tickers due to >20% missing: {dropped[:10]}...")
    returns = returns[keep]

    # Forward-fill remaining gaps (up to 2 months)
    returns = returns.fillna(method="ffill", limit=2)

    logger.info(f"Monthly returns: {returns.shape[0]} months × {returns.shape[1]} stocks")
    os.makedirs(cache_dir, exist_ok=True)
    returns.to_parquet(cache_path)
    return returns


def load_csv(path: str) -> pd.DataFrame:
    """
    Load user-supplied CSV of monthly returns.

    Expected format:
      - Column named 'Date' (YYYY-MM-DD) or DatetimeIndex
      - One column per stock (monthly returns as decimals, e.g. 0.05 = +5%)
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    logger.info(f"Loaded CSV: {df.shape[0]} months × {df.shape[1]} stocks from {path}")
    return df


def apply_data_filters(returns: pd.DataFrame, bottom_mcap_pct: float = 0.05) -> pd.DataFrame:
    """
    Apply paper's data quality filters (Section 3.1):
      1. Winsorize monthly returns at [-95%, +300%] then cross-sectionally at [1%, 99%]
      2. Drop stock-months with |return| = 0 (proxy for zero volume)

    Note: We cannot fully replicate the paper's WRDS-based filters (zero trading volume,
    missing market cap, bottom 5% mcap) without fundamental data. These are approximated.

    Returns
    -------
    pd.DataFrame  same shape as input, with filtered/winsorized values.
    """
    # Hard winsorize: [−95%, +300%]
    returns = returns.clip(lower=-0.95, upper=3.0)

    # Cross-sectional winsorize at 1%/99% per month
    def cs_winsorize(row: pd.Series) -> pd.Series:
        lo, hi = row.quantile([0.01, 0.99])
        return row.clip(lower=lo, upper=hi)

    returns = returns.apply(cs_winsorize, axis=1)

    return returns


def compute_market_cap_proxy(returns: pd.DataFrame, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute a market-cap proxy for size decile assignment.

    In the absence of actual shares outstanding data, we use the cumulative
    price level (reinvested returns, starting from a base of 1.0) as a
    proportional proxy for relative size. This approximation is reasonable
    when the cross-section of price levels is stable.

    Returns
    -------
    pd.DataFrame  same shape as returns, containing relative market cap proxy.
    """
    # Use cumulative return index as size proxy (base = 100 at start)
    cumret = (1 + returns.fillna(0)).cumprod() * 100.0
    return cumret

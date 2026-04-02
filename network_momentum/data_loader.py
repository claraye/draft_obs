"""
data_loader.py — Fetch and cache price data
Supports: Yahoo Finance (default) or CSV files with a 'Date' index column.
"""
from __future__ import annotations

import os
import pickle
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Yahoo Finance loader ─────────────────────────────────────────────────────

def fetch_yahoo(tickers: list[str], start: str, end: str, cache_dir: str) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.
    Returns DataFrame: index=Date, columns=tickers (only tickers with sufficient data).
    Results are cached to cache_dir as a pickle.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"prices_{start}_{end}.pkl")

    if os.path.exists(cache_file):
        logger.info(f"Loading prices from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    logger.info(f"Downloading {len(tickers)} tickers from Yahoo Finance ({start} → {end})")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    # Drop tickers with >20% missing
    thresh = int(len(prices) * 0.8)
    prices = prices.dropna(axis=1, thresh=thresh)
    prices = prices.ffill().bfill()

    logger.info(f"Retained {prices.shape[1]} tickers after filtering")

    with open(cache_file, "wb") as f:
        pickle.dump(prices, f)

    return prices


def load_csv(path: str) -> pd.DataFrame:
    """
    Load price data from a CSV file.
    Expected format: first column = Date (YYYY-MM-DD), remaining columns = asset prices.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.ffill().bfill()
    logger.info(f"Loaded {df.shape[1]} assets from {path}")
    return df


# ─── Returns & vol ───────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices / prices.shift(1))


def ewm_vol(returns: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    """
    EWMA daily volatility with given span.
    σ_t = EWMstd(returns, span=span)
    Annualised: σ_t * sqrt(252)
    """
    return returns.ewm(span=span, min_periods=span // 2, adjust=False).std()


def rolling_std(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling standard deviation of prices (used in MACD normalisation)."""
    return prices.rolling(window=window, min_periods=window // 2).std()


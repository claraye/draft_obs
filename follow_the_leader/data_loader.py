"""
data_loader.py — Price data loading and preprocessing for Follow The Leader
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_yahoo(
    tickers: list,
    start: str,
    end: str,
    cache_dir: str,
    min_days: int = 252,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices from Yahoo Finance with disk caching.

    Args:
        tickers:   list of Yahoo Finance ticker symbols
        start:     start date string "YYYY-MM-DD"
        end:       end date string "YYYY-MM-DD"
        cache_dir: directory for pickle cache
        min_days:  minimum non-NaN days required to keep a ticker

    Returns:
        prices: DataFrame(T, N) of adjusted close prices, date index
    """
    import yfinance as yf

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"prices_{start}_{end}.pkl")

    if os.path.exists(cache_path):
        logger.info(f"Loading prices from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(f"Fetching {len(tickers)} tickers from Yahoo Finance ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if len(tickers) == 1 else raw

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Forward-fill gaps up to 5 days (weekends, public holidays)
    prices = prices.ffill(limit=5)

    # Drop tickers with insufficient history
    valid = prices.notna().sum() >= min_days
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} tickers with < {min_days} days: "
                       f"{list(valid[~valid].index)}")
    prices = prices.loc[:, valid]

    # Drop rows where all prices are NaN (early dates)
    prices = prices.dropna(how="all")

    with open(cache_path, "wb") as f:
        pickle.dump(prices, f)
    logger.info(
        f"Cached {prices.shape[1]} assets, {len(prices)} trading days "
        f"({prices.index[0].date()} to {prices.index[-1].date()})"
    )
    return prices


def load_csv(path: str) -> pd.DataFrame:
    """
    Load prices from a CSV file.
    Expected format: date in first column (index), asset prices in remaining columns.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index().ffill(limit=5).dropna(how="all")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns: r_t = log(P_t / P_{t-1})"""
    return np.log(prices / prices.shift(1))


def align_to_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    backtest_start: str,
    backtest_end: str,
) -> tuple:
    """
    Remove assets that still have NaN at the backtest start date.
    Returns (prices, returns) aligned to the valid asset set.
    """
    start_dt = pd.Timestamp(backtest_start)
    end_dt   = pd.Timestamp(backtest_end)

    # Find assets with valid prices at backtest start
    prices_oos = prices.loc[start_dt:end_dt]
    valid = prices_oos.notna().all(axis=0)
    if (~valid).any():
        dropped = list(valid[~valid].index)
        logger.info(f"Removed {len(dropped)} assets missing data at backtest start: {dropped}")

    prices_clean  = prices.loc[:end_dt, valid]
    returns_clean = returns.loc[:end_dt, valid]
    return prices_clean, returns_clean

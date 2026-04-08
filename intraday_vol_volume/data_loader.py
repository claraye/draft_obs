"""
data_loader.py — Download and prepare 5-minute futures data for the
Volume-Driven ToD Intraday Volatility replication.

Key functions
─────────────
fetch_intraday(tickers, period, cache_dir, nocache, interval)
    → dict[ticker → pd.DataFrame with columns: open, high, low, close, volume]
      indexed by timezone-aware datetime (US/Eastern)

prepare_intraday(df, interval_labels, log_var_clip)
    → pd.DataFrame with columns:
        date        : calendar date
        interval    : intraday interval label (HHMM of bar open)
        log_r2      : log(return²), clipped at log_var_clip
        log_vol     : log(volume)
        bar_ret     : open-to-close simple return for the bar
        bar_idx     : 0-based position within trading day (0 = first bar)
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_START = dt.time(18, 0)
SESSION_BREAK_START = dt.time(17, 0)
BAR_FREQ = "5min"


def _configure_yfinance_cache(yf, cache_dir: str) -> None:
    """Keep yfinance's optional sqlite caches from breaking local downloads."""
    try:
        import yfinance.cache as yf_cache

        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(cache_dir)

        yf_cache._TzCacheManager._tz_cache = yf_cache._TzCacheDummy()
        yf_cache._CookieCacheManager._Cookie_cache = yf_cache._CookieCacheDummy()
        if hasattr(yf_cache, "_ISINCacheManager") and hasattr(yf_cache, "_ISINCacheDummy"):
            yf_cache._ISINCacheManager._isin_cache = yf_cache._ISINCacheDummy()
    except Exception as exc:
        logger.debug(f"[data] Unable to reconfigure yfinance cache: {exc}")


def _assign_trade_date(index: pd.DatetimeIndex) -> pd.Index:
    """Map timestamps to the futures trade date used in the paper."""
    return pd.Index([
        (ts + pd.Timedelta(days=1)).date() if ts.time() >= SESSION_START else ts.date()
        for ts in index
    ])


def _session_timestamps(trade_date: dt.date) -> pd.DatetimeIndex:
    """Expected 5-minute timestamps for one 23-hour futures session."""
    session_open = pd.Timestamp.combine(
        trade_date - dt.timedelta(days=1), SESSION_START
    ).tz_localize("US/Eastern")
    return pd.date_range(session_open, periods=276, freq=BAR_FREQ)


def _standardize_futures_session(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the daily maintenance break and fill missing intraday intervals.

    Following the paper, missing intervals within the trading session are filled
    with the last observed price and zero volume.
    """
    raw = raw.sort_index().copy()
    raw = raw[
        (raw.index.time < SESSION_BREAK_START) | (raw.index.time >= SESSION_START)
    ].copy()
    raw["trade_date"] = _assign_trade_date(raw.index)

    sessions: list[pd.DataFrame] = []
    for trade_date, grp in raw.groupby("trade_date"):
        expected_index = _session_timestamps(trade_date)
        session = grp.reindex(expected_index)

        prev_close = session["close"].ffill()
        session["close"] = prev_close
        for col in ("open", "high", "low"):
            session[col] = session[col].where(session[col].notna(), prev_close)
        session["volume"] = session["volume"].fillna(0.0)
        session = session.dropna(subset=["open", "high", "low", "close"])
        session["trade_date"] = trade_date
        sessions.append(session)

    if not sessions:
        return raw.iloc[0:0][["open", "high", "low", "close", "volume"]]

    result = pd.concat(sessions).sort_index()
    return result[["open", "high", "low", "close", "volume"]]


def fetch_intraday(
    tickers:   list[str],
    period:    str  = "60d",
    cache_dir: str  = "cache",
    interval:  str  = "5m",
    nocache:   bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download intraday OHLCV data via yfinance.

    Returns dict {ticker: DataFrame(datetime_index, [open,high,low,close,volume])}.
    Data is in US/Eastern timezone and standardized to the paper's futures
    session: 18:00 ET to 17:00 ET with the 17:00–18:00 break removed.
    """
    import yfinance as yf

    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(BASE_DIR, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    tz_cache_dir = os.path.join(cache_dir, "yfinance_tz_cache")
    os.makedirs(tz_cache_dir, exist_ok=True)
    _configure_yfinance_cache(yf, tz_cache_dir)

    cache_path = os.path.join(cache_dir, f"intraday_{interval}_{period}.pkl")

    if not nocache and os.path.exists(cache_path):
        logger.info(f"[data] Loading intraday data from cache: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning(f"[data] Cache load failed; rebuilding cache. Reason: {exc}")

    logger.info(
        f"[data] Downloading {interval} intraday data for {len(tickers)} tickers..."
    )
    result: dict[str, pd.DataFrame] = {}

    for tk in tickers:
        try:
            raw = yf.download(
                tk, period=period, interval=interval,
                auto_adjust=True, progress=False
            )
            if raw.empty:
                logger.warning(f"  {tk}: no data returned")
                continue

            # Flatten multi-level columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0).str.lower()
            else:
                raw.columns = raw.columns.str.lower()

            # Localize to US/Eastern
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC").tz_convert("US/Eastern")
            else:
                raw.index = raw.index.tz_convert("US/Eastern")

            raw = _standardize_futures_session(raw)

            if raw.empty:
                logger.warning(f"  {tk}: no bars in futures session")
                continue

            result[tk] = raw[["open", "high", "low", "close", "volume"]]
            n_sessions = len(pd.Index(_assign_trade_date(raw.index)).unique())
            logger.info(f"  {tk}: {len(raw)} {interval} bars across {n_sessions} sessions")

        except Exception as e:
            logger.warning(f"  {tk}: download failed — {e}")

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"[data] Cached to {cache_path}")

    return result


def fetch_hourly(
    tickers:   list[str],
    period:    str  = "60d",
    cache_dir: str  = "cache",
    nocache:   bool = False,
) -> dict[str, pd.DataFrame]:
    """Backward-compatible alias for older callers."""
    return fetch_intraday(
        tickers=tickers,
        period=period,
        cache_dir=cache_dir,
        interval="5m",
        nocache=nocache,
    )


def prepare_intraday(
    df:             pd.DataFrame,
    interval_labels: list[int] | None = None,
    log_var_clip:   float = -20.0,
) -> pd.DataFrame:
    """
    Convert raw intraday OHLCV DataFrame to the panel format needed by the model.

    Parameters
    ----------
    df              : raw intraday DataFrame (datetime index, US/Eastern)
    interval_labels : list of HHMM labels (e.g. [930, 1030, ...]) in order.
                      If None, labels are inferred from the data.
    log_var_clip    : floor for log(r²); prevents -inf from zero-return bars

    Returns
    -------
    DataFrame with columns:
        timestamp, date, interval (HHMM int), bar_idx (int), log_r2 (float),
        log_vol (float), bar_ret (float)
    Sorted by (date, bar_idx).
    """
    df = df.copy()

    # Compute bar return from open-to-close of each bar
    df["ret"] = (df["close"] - df["open"]) / df["open"]

    # log(r²), clipped
    r2 = df["ret"] ** 2
    df["log_r2"] = np.log(r2.clip(lower=np.exp(log_var_clip)))

    # log(volume), clip negatives
    df["log_vol"] = np.log(df["volume"].clip(lower=1.0))

    # Intraday interval label from bar open time (HHMM integer)
    df["interval"] = df.index.hour * 100 + df.index.minute
    df["date"]     = _assign_trade_date(df.index)

    # bar_idx: rank of bar within the trading day (0-based)
    df["bar_idx"] = df.groupby("date").cumcount()

    # Drop rows with NaN returns
    df = df.dropna(subset=["ret", "log_r2", "log_vol"])

    # If interval_labels provided, filter to only those intervals
    if interval_labels is not None:
        df = df[df["interval"].isin(interval_labels)]

    result = df[["date", "interval", "bar_idx", "log_r2", "log_vol", "ret"]].copy()
    result.insert(0, "timestamp", df.index)
    result = result.rename(columns={"ret": "bar_ret"})
    result = result.sort_values(["date", "bar_idx"]).reset_index(drop=True)

    return result


def build_volume_deviation(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the standardized volume deviation from the time-of-day mean.

    v_t = (log_vol_t - mean(log_vol | interval_k)) / std(log_vol | interval_k)

    This is the v_{t-1} term in tod_t^v = v_{t-1} * I_t' * beta^v.

    Parameter
    ---------
    panel : output of prepare_intraday(), covering the full estimation window

    Returns
    -------
    Same panel with additional column 'vol_dev' (standardized volume deviation).
    """
    panel = panel.copy()
    grp = panel.groupby("interval")["log_vol"]
    mu  = grp.transform("mean")
    sig = grp.transform("std")

    # Avoid division by zero for intervals with constant volume
    sig = sig.replace(0, np.nan).fillna(1.0)

    panel["vol_dev"] = (panel["log_vol"] - mu) / sig
    return panel


def daily_returns(intraday_raw: pd.DataFrame) -> pd.Series:
    """
    Compute daily open-to-close returns from intraday data.

    Uses first bar open and last bar close of each trading day.

    Returns pd.Series indexed by date (Python date).
    """
    daily: dict = {}
    trade_dates = _assign_trade_date(intraday_raw.index)
    for date, grp in intraday_raw.groupby(trade_dates):
        grp_sorted = grp.sort_index()
        if len(grp_sorted) == 0:
            continue
        day_open  = grp_sorted["open"].iloc[0]
        day_close = grp_sorted["close"].iloc[-1]
        if day_open > 0:
            daily[date] = (day_close - day_open) / day_open
    return pd.Series(daily, name="daily_ret")

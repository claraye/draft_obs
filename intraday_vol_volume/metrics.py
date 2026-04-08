"""
metrics.py — Performance metrics for intraday vol-managed portfolio replication.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def annualized_return(rets: np.ndarray) -> float:
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return np.nan
    total = np.prod(1 + rets)
    n_yrs = len(rets) / TRADING_DAYS_PER_YEAR
    if n_yrs <= 0:
        return np.nan
    return float(total ** (1 / n_yrs) - 1)


def annualized_vol(rets: np.ndarray) -> float:
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return np.nan
    return float(rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(rets: np.ndarray, rf_annual: float = 0.04) -> float:
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return np.nan
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    excess   = rets - rf_daily
    if excess.std() == 0:
        return np.nan
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(rets: np.ndarray) -> float:
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return np.nan
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd   = cum / peak - 1
    return float(dd.min())


def calmar_ratio(rets: np.ndarray) -> float:
    mdd = max_drawdown(rets)
    ann = annualized_return(rets)
    if np.isnan(mdd) or mdd == 0 or np.isnan(ann):
        return np.nan
    return float(ann / abs(mdd))


def print_summary(
    results: dict,   # {ticker: BacktestResult}
    rf_annual: float = 0.04,
) -> None:
    """
    Print performance table for all tickers and strategies.

    Parameters
    ----------
    results   : dict[ticker → BacktestResult]
    rf_annual : annualized risk-free rate
    """
    rows = []
    for ticker, res in results.items():
        if not res.dates:
            continue
        n_days = len(res.dates)
        for strat, rets in res.returns.items():
            rets = np.array(rets, dtype=float)
            valid = rets[np.isfinite(rets)]
            rows.append({
                "Ticker":         ticker,
                "Strategy":       strat,
                "Ann. Return (%)": round(annualized_return(valid) * 100, 2),
                "Ann. Vol (%)":   round(annualized_vol(valid) * 100, 2),
                "Sharpe":         round(sharpe_ratio(valid, rf_annual), 3),
                "Max DD (%)":     round(max_drawdown(valid) * 100, 2),
                "Calmar":         round(calmar_ratio(valid), 3),
                "N Days":         len(valid),
            })
        # Append forecast quality rows
        rows.append({
            "Ticker":         ticker,
            "Strategy":       "── MZ R² (train)",
            "Ann. Return (%)": round(res.mz_r2_train * 100, 2) if np.isfinite(res.mz_r2_train) else np.nan,
            "Ann. Vol (%)":   "—", "Sharpe": "—", "Max DD (%)": "—",
            "Calmar": "—", "N Days": "—",
        })
        rows.append({
            "Ticker":         ticker,
            "Strategy":       "── MZ R² (OOS)",
            "Ann. Return (%)": round(res.mz_r2_oos * 100, 2) if np.isfinite(res.mz_r2_oos) else np.nan,
            "Ann. Vol (%)":   "—", "Sharpe": "—", "Max DD (%)": "—",
            "Calmar": "—", "N Days": "—",
        })

    df = pd.DataFrame(rows).set_index(["Ticker", "Strategy"])

    print("\n" + "=" * 90)
    print("  VOLUME-DRIVEN TOD INTRADAY VOLATILITY — REPLICATION — PERFORMANCE SUMMARY")
    print("=" * 90)
    print(df.to_string())
    print("=" * 90 + "\n")

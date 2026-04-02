"""
metrics.py — Portfolio performance metrics
Matches Table 1 in the paper: return, vol, Sharpe, downside dev, MDD,
MDD duration, Sortino, Calmar, hit rate, Avg.P/Avg.L
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualised_return(daily_rets: np.ndarray, periods: int = 252) -> float:
    """Expected annualised return."""
    return float(np.nanmean(daily_rets) * periods)


def annualised_vol(daily_rets: np.ndarray, periods: int = 252) -> float:
    """Annualised standard deviation."""
    return float(np.nanstd(daily_rets, ddof=1) * np.sqrt(periods))


def sharpe_ratio(daily_rets: np.ndarray, periods: int = 252) -> float:
    """Sharpe ratio (zero risk-free rate assumption, consistent with paper)."""
    mu  = np.nanmean(daily_rets)
    std = np.nanstd(daily_rets, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(mu / std * np.sqrt(periods))


def downside_deviation(daily_rets: np.ndarray, periods: int = 252) -> float:
    """Annualised downside deviation (semi-std of negative returns)."""
    neg = daily_rets[daily_rets < 0]
    if len(neg) == 0:
        return 0.0
    return float(np.std(neg, ddof=1) * np.sqrt(periods))


def sortino_ratio(daily_rets: np.ndarray, periods: int = 252) -> float:
    """Sortino ratio = Ann.Return / Downside Deviation."""
    r = annualised_return(daily_rets, periods)
    dd = downside_deviation(daily_rets, periods)
    if dd < 1e-12:
        return 0.0
    return r / dd


def max_drawdown(daily_rets: np.ndarray) -> tuple[float, float]:
    """
    Maximum drawdown and MDD duration (fraction of days in drawdown).
    Returns (mdd, mdd_duration_fraction).
    """
    cum = np.cumprod(1 + daily_rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = float(dd.min())
    mdd_duration = float((dd < 0).mean())
    return mdd, mdd_duration


def calmar_ratio(daily_rets: np.ndarray, periods: int = 252) -> float:
    """Calmar = Ann.Return / |Max Drawdown|."""
    r = annualised_return(daily_rets, periods)
    mdd, _ = max_drawdown(daily_rets)
    if abs(mdd) < 1e-12:
        return 0.0
    return r / abs(mdd)


def hit_rate(daily_rets: np.ndarray) -> float:
    """Fraction of days with positive returns."""
    return float((daily_rets > 0).mean())


def avg_profit_loss_ratio(daily_rets: np.ndarray) -> float:
    """Average profit / average loss (absolute)."""
    profits = daily_rets[daily_rets > 0]
    losses  = daily_rets[daily_rets < 0]
    avg_p = profits.mean() if len(profits) > 0 else 0.0
    avg_l = abs(losses.mean()) if len(losses) > 0 else 1e-10
    return float(avg_p / avg_l)


def cost_adjusted_returns(
    daily_rets: np.ndarray,
    turnovers: np.ndarray,
    cost_bps: float,
) -> np.ndarray:
    """
    Subtract transaction costs from daily returns.
    cost_bps: cost per unit of turnover in basis points.
    """
    cost = (cost_bps / 10_000) * turnovers
    return daily_rets - cost


def compute_turnover(
    positions_t: np.ndarray,
    positions_tm1: np.ndarray,
    sigma_t: np.ndarray,
    sigma_tm1: np.ndarray,
    vol_target: float = 0.15,
) -> float:
    """
    Daily portfolio turnover ζ (Eq. 13):
      ζ_{i,t} = σ_tgt * |x_{i,t}/σ_{i,t} - x_{i,t-1}/σ_{i,t-1}|
    Returns average turnover across all assets.
    """
    scaled_t   = positions_t   / (sigma_t   * np.sqrt(252) + 1e-10)
    scaled_tm1 = positions_tm1 / (sigma_tm1 * np.sqrt(252) + 1e-10)
    turnover_i = vol_target * np.abs(scaled_t - scaled_tm1)
    return float(np.nanmean(turnover_i))


def summary_table(daily_rets: np.ndarray, label: str = "", periods: int = 252) -> dict:
    """Compute all metrics from Table 1 and return as a dict."""
    rets = np.array(daily_rets)
    rets = rets[np.isfinite(rets)]

    mdd, mdd_dur = max_drawdown(rets)
    return {
        "Strategy":     label,
        "Return":       round(annualised_return(rets, periods), 4),
        "Vol":          round(annualised_vol(rets, periods), 4),
        "Sharpe":       round(sharpe_ratio(rets, periods), 4),
        "DownsideDev":  round(downside_deviation(rets, periods), 4),
        "MDD":          round(abs(mdd), 4),
        "MDD_Duration": round(mdd_dur, 4),
        "Sortino":      round(sortino_ratio(rets, periods), 4),
        "Calmar":       round(calmar_ratio(rets, periods), 4),
        "HitRate":      round(hit_rate(rets), 4),
        "AvgP_AvgL":    round(avg_profit_loss_ratio(rets), 4),
    }


def print_summary(results: dict[str, np.ndarray], periods: int = 252) -> pd.DataFrame:
    """Print a formatted performance table and return as DataFrame."""
    rows = []
    for label, rets in results.items():
        rows.append(summary_table(rets, label, periods))
    df = pd.DataFrame(rows).set_index("Strategy")
    print("\n" + "=" * 80)
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80 + "\n")
    return df

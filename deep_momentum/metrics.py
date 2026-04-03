"""
metrics.py — Performance metrics for Deep Momentum replication
Paper: Han & Qin (2026), SSRN 4452964
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

MONTHS_PER_YEAR = 12


def annualized_return(rets: np.ndarray) -> float:
    """Compound annualized return from monthly returns."""
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return np.nan
    total = np.prod(1 + rets)
    n_years = len(rets) / MONTHS_PER_YEAR
    return total ** (1 / n_years) - 1


def sharpe_ratio(rets: np.ndarray, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio (no risk-free subtraction by default, as in the paper)."""
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return np.nan
    excess = rets - rf / MONTHS_PER_YEAR
    return (excess.mean() / excess.std()) * np.sqrt(MONTHS_PER_YEAR)


def max_drawdown(rets: np.ndarray) -> float:
    """Maximum drawdown from monthly returns."""
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return np.nan
    cumulative = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    return drawdowns.min()


def crash_rate(rets: np.ndarray, mdd_threshold: float = -0.20) -> float:
    """
    Fraction of years in which the MDD exceeds the threshold (paper: 20%).
    Computed 12 times with 1-month shifts and averaged.
    """
    rets = rets[np.isfinite(rets)]
    n = len(rets)
    if n < 12:
        return np.nan

    crash_years = []
    for shift in range(12):
        shifted = rets[shift:]
        n_full_years = len(shifted) // 12
        if n_full_years == 0:
            continue
        for yr in range(n_full_years):
            yr_rets = shifted[yr * 12 : (yr + 1) * 12]
            mdd = max_drawdown(yr_rets)
            crash_years.append(int(mdd < mdd_threshold))

    return np.mean(crash_years) if crash_years else np.nan


def breakeven_tc(rets: np.ndarray, turnovers: np.ndarray, step_bps: float = 1.0) -> float:
    """
    Find the flat transaction cost (bps) at which net monthly return = 0.
    Increment by step_bps until cumulative return ≤ 0.

    turnovers: monthly portfolio turnover (fraction, e.g. 2.48 = 248% monthly)
    """
    for cost_bps in np.arange(0, 500, step_bps):
        cost_frac = cost_bps / 10_000
        net_rets  = rets - turnovers * cost_frac
        if annualized_return(net_rets) <= 0:
            return cost_bps
    return np.nan


def print_summary(
    returns_dict: dict[str, np.ndarray],
    turnovers_dict: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Print and return a performance summary table.
    """
    rows = []
    for strat, rets in returns_dict.items():
        rets = np.array(rets, dtype=float)
        valid = rets[np.isfinite(rets)]
        row = {
            "Strategy":       strat,
            "Ann. Return (%)": round(annualized_return(rets) * 100, 2),
            "Sharpe":         round(sharpe_ratio(rets), 3),
            "Max Drawdown (%)": round(max_drawdown(rets) * 100, 2),
            "Crash Rate":     round(crash_rate(rets), 3),
            "N Months":       len(valid),
        }
        if turnovers_dict and strat in turnovers_dict:
            tvr = np.array(turnovers_dict[strat], dtype=float)
            row["Avg Monthly Turnover (%)"] = round(np.nanmean(tvr) * 100, 1)
            row["Breakeven TC (bps)"] = round(breakeven_tc(rets, tvr), 1)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Strategy")

    print("\n" + "=" * 70)
    print("  PERFORMANCE SUMMARY")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70 + "\n")

    return df

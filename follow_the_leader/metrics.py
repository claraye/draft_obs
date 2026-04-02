"""
metrics.py — Portfolio performance metrics

Matches paper Table 1 metrics: Net Sharpe, Sortino, Skewness, MDD.
Also includes: Ann. Return, Ann. Vol, Calmar, Hit Rate, Avg Profit/Loss.
"""
import numpy as np
import pandas as pd


def annualised_return(daily_rets: np.ndarray, tdpy: int = 252) -> float:
    return float(np.nanmean(daily_rets) * tdpy)


def annualised_vol(daily_rets: np.ndarray, tdpy: int = 252) -> float:
    return float(np.nanstd(daily_rets, ddof=1) * np.sqrt(tdpy))


def sharpe_ratio(daily_rets: np.ndarray, tdpy: int = 252) -> float:
    """Sharpe ratio (zero risk-free rate, consistent with paper)."""
    mu  = np.nanmean(daily_rets)
    std = np.nanstd(daily_rets, ddof=1)
    return float(mu / std * np.sqrt(tdpy)) if std > 1e-12 else 0.0


def downside_deviation(daily_rets: np.ndarray, tdpy: int = 252) -> float:
    """Annualised downside deviation (semi-std of negative returns)."""
    neg = daily_rets[daily_rets < 0]
    return float(np.std(neg, ddof=1) * np.sqrt(tdpy)) if len(neg) > 1 else 0.0


def sortino_ratio(daily_rets: np.ndarray, tdpy: int = 252) -> float:
    """Sortino = Ann. Return / Downside Deviation."""
    r  = annualised_return(daily_rets, tdpy)
    dd = downside_deviation(daily_rets, tdpy)
    return r / dd if dd > 1e-12 else 0.0


def max_drawdown(daily_rets: np.ndarray) -> tuple:
    """(MDD, MDD_duration_fraction) — fraction of time in drawdown."""
    cum  = np.cumprod(1.0 + daily_rets)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / (peak + 1e-12)
    return float(dd.min()), float((dd < 0).mean())


def calmar_ratio(daily_rets: np.ndarray, tdpy: int = 252) -> float:
    r   = annualised_return(daily_rets, tdpy)
    mdd, _ = max_drawdown(daily_rets)
    return r / abs(mdd) if abs(mdd) > 1e-12 else 0.0


def skewness(daily_rets: np.ndarray) -> float:
    """Sample skewness of daily returns."""
    from scipy.stats import skew
    return float(skew(daily_rets, nan_policy="omit"))


def hit_rate(daily_rets: np.ndarray) -> float:
    return float((daily_rets > 0).mean())


def avg_profit_loss(daily_rets: np.ndarray) -> float:
    profits = daily_rets[daily_rets > 0]
    losses  = daily_rets[daily_rets < 0]
    avg_p = profits.mean() if len(profits) > 0 else 0.0
    avg_l = abs(losses.mean()) if len(losses) > 0 else 1e-10
    return float(avg_p / avg_l)


def summary_table(daily_rets: np.ndarray, label: str = "", tdpy: int = 252) -> dict:
    """All metrics from paper Table 1."""
    rets = np.asarray(daily_rets)
    rets = rets[np.isfinite(rets)]
    mdd, mdd_dur = max_drawdown(rets)
    return {
        "Strategy":     label,
        "Return":       round(annualised_return(rets, tdpy), 4),
        "Vol":          round(annualised_vol(rets, tdpy), 4),
        "Sharpe":       round(sharpe_ratio(rets, tdpy), 4),
        "Sortino":      round(sortino_ratio(rets, tdpy), 4),
        "Skewness":     round(skewness(rets), 4),
        "MDD":          round(abs(mdd), 4),
        "MDD_Duration": round(mdd_dur, 4),
        "Calmar":       round(calmar_ratio(rets, tdpy), 4),
        "HitRate":      round(hit_rate(rets), 4),
        "AvgP_AvgL":    round(avg_profit_loss(rets), 4),
    }


def print_summary(results: dict, tdpy: int = 252) -> pd.DataFrame:
    """Print formatted performance table and return as DataFrame."""
    rows = [summary_table(np.asarray(v), label=k, tdpy=tdpy) for k, v in results.items()]
    df   = pd.DataFrame(rows).set_index("Strategy")
    sep  = "=" * 85
    print(f"\n{sep}")
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print(sep)
    print(df.to_string())
    print(f"{sep}\n")
    return df

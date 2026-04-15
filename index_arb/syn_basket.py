"""
syn_basket.py — Self-contained S&P500 Synthetic Basket Stat-Arb grid search.

Recorded metrics per (score_type, spread, lookback, entry, exit) combination:
  sharpe        — annualised Sharpe ratio
  calmar        — annualised return / |max drawdown|
  trade_on_pct  — fraction of OOS days with a non-zero position
  turnover_yr   — position changes per year (entries + exits + flips)

Usage:
    python syn_basket.py              # full grid, ~600 combos
    python syn_basket.py --quick      # small grid for smoke-test
    python syn_basket.py --spread MKFA  # one spread only
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

TRADING_DAYS = 252
SPREAD_LABELS = ["MKFA", "Rolling OLS", "Constant OLS"]


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_prices(
    tickers: List[str],
    start: str,
    end: str,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Download adjusted close prices.
    Drops tickers with more than 5% missing data; forward-fills remaining gaps.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=progress)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])
    threshold = 0.05 * len(raw)
    clean = raw.dropna(axis=1, thresh=int(len(raw) - threshold))
    return clean.ffill().dropna()


def get_log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return natural log of price DataFrame."""
    return np.log(prices)


# ═══════════════════════════════════════════════════════════════════════════════
# Cointegration
# ═══════════════════════════════════════════════════════════════════════════════

def adf_is_stationary(series: pd.Series, pvalue_threshold: float = 0.05) -> bool:
    """Return True if ADF test rejects unit root at given significance level."""
    result = adfuller(series.dropna(), autolag="AIC")
    return result[1] < pvalue_threshold


def ols_hedge_ratio(y: pd.Series, X: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    """
    Fit OLS: y = X @ beta (no intercept unless a constant column is already in X).
    Returns (beta array, residuals Series).
    """
    model = OLS(y, X).fit()
    return model.params.values, model.resid


def screen_individual_cointegration(
    log_spy: pd.Series,
    log_stocks: pd.DataFrame,
    pvalue_threshold: float = 0.05,
) -> List[str]:
    """
    For each stock S_i, run log(SPY) = γ·log(S_i) + ε (no intercept).
    Return tickers whose OLS residuals are stationary (ADF p < threshold).
    """
    candidates = []
    for ticker in log_stocks.columns:
        _, resid = ols_hedge_ratio(log_spy, log_stocks[[ticker]])
        if adf_is_stationary(resid, pvalue_threshold):
            candidates.append(ticker)
    return candidates


def build_synthetic_basket(
    log_spy: pd.Series,
    log_candidates: pd.DataFrame,
    pvalue_threshold: float = 0.05,
) -> Tuple[Optional[List[str]], Optional[np.ndarray], Optional[pd.Series]]:
    """
    Joint OLS: log(SPY) = γ·Σ_i log(S_i) + ε.
    Returns (tickers, [γ], residuals) if stationary, else (None, None, None).
    """
    if log_candidates.empty or len(log_candidates.columns) == 0:
        return None, None, None
    basket_logsum = log_candidates.sum(axis=1).rename("basket")
    beta, resid = ols_hedge_ratio(log_spy, basket_logsum.to_frame())
    if adf_is_stationary(resid, pvalue_threshold):
        return list(log_candidates.columns), beta, resid
    return None, None, None


def individual_cointegration_pvalues(
    log_spy: pd.Series,
    log_stocks: pd.DataFrame,
) -> pd.Series:
    """
    For each ticker, run OLS log(SPY) = γ·log(S_i) + ε and return ADF p-value.
    Returns a Series {ticker -> p-value}; NaN if the test fails.
    """
    pvals: Dict[str, float] = {}
    for ticker in log_stocks.columns:
        try:
            _, resid = ols_hedge_ratio(log_spy, log_stocks[[ticker]])
            pvals[ticker] = float(adfuller(resid.dropna(), autolag="AIC")[1])
        except Exception:
            pvals[ticker] = float("nan")
    return pd.Series(pvals, name="pvalue")


# ═══════════════════════════════════════════════════════════════════════════════
# Hedge ratios
# ═══════════════════════════════════════════════════════════════════════════════

def spread_constant_ols(
    log_spy: pd.Series,
    log_basket: pd.DataFrame,
    gamma: np.ndarray,
) -> pd.Series:
    """Constant-gamma spread: ε_t = log(SPY_t) − γ·Σ_i log(S_i_t)."""
    return log_spy - gamma[0] * log_basket.sum(axis=1)


def spread_rolling_ols(
    log_spy: pd.Series,
    log_basket: pd.DataFrame,
    window: int = 60,
) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling OLS hedge ratio.
    At each t, refit OLS on trailing `window` observations.
    Returns (spread Series, gamma_t Series).
    """
    n = len(log_spy)
    spread  = pd.Series(np.nan, index=log_spy.index)
    gamma_t = pd.Series(np.nan, index=log_spy.index)
    basket_sum = log_basket.sum(axis=1)

    for t in range(window, n):
        y_win = log_spy.iloc[t - window: t]
        X_win = basket_sum.iloc[t - window: t].values.reshape(-1, 1)
        model = OLS(y_win, X_win).fit()
        g = model.params.iloc[0]
        gamma_t.iloc[t] = g
        spread.iloc[t]  = log_spy.iloc[t] - g * basket_sum.iloc[t]

    return spread, gamma_t


def fit_kalman_init(
    log_spy: pd.Series,
    log_basket: pd.DataFrame,
) -> np.ndarray:
    """
    Fit initial KF state y_0 = [gamma_1..N, mu] by OLS on in-sample data.
    OLS: log(SPY) = Σ_i gamma_i·log(S_i) + mu
    """
    X = add_constant(log_basket, prepend=False)   # stock cols, then const
    model = OLS(log_spy, X).fit()
    gamma = model.params[log_basket.columns].values
    mu    = model.params["const"]
    return np.append(gamma, mu)


def spread_kalman_filter(
    log_spy: pd.Series,
    log_basket: pd.DataFrame,
    gamma_init: np.ndarray,
    V_w: float = 1e-4,
    V_e: float = 1e-3,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Multivariate Kalman Filter hedge ratio (Drakos 2016, Eqs 12-19).

    State  y_t = [gamma_1..N, mu]^T       shape (N+1,)
    Obs    x_t = [log S_1..N,  1  ]       shape (N+1,)
    Transition: y_t = y_{t-1} + w_t       (random walk)
    Measurement: z_t = x_t @ y_t
    Spread: ε_t = log(SPY_t) − x_t @ y_{t|t}
    """
    stocks    = list(log_basket.columns)
    N         = len(stocks)
    state_dim = N + 1

    y        = gamma_init.copy().astype(float)
    P        = np.eye(state_dim) * V_w
    V_w_mat  = np.eye(state_dim) * V_w
    V_e_sc   = V_e

    n      = len(log_spy)
    spread = pd.Series(np.nan, index=log_spy.index)
    gammas = pd.DataFrame(np.nan, index=log_spy.index,
                          columns=stocks + ["mu"])

    for t in range(n):
        x = np.append(log_basket.iloc[t].values, 1.0)

        # Predict
        y_pred = y.copy()
        P_pred = P + V_w_mat

        # Measure
        z_hat = x @ y_pred
        eps   = log_spy.iloc[t] - z_hat
        S     = x @ P_pred @ x + V_e_sc
        K     = (P_pred @ x) / S

        # Update
        y = y_pred + K * eps
        P = P_pred - np.outer(K, K) * S

        spread.iloc[t] = log_spy.iloc[t] - x @ y
        gammas.iloc[t] = y

    return spread, gammas


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy signals
# ═══════════════════════════════════════════════════════════════════════════════

def compute_zscore(
    spread: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    """Rolling z-score: z_t = (ε_t − mean) / std  over `lookback` days."""
    roll_mean = spread.rolling(lookback).mean()
    roll_std  = spread.rolling(lookback).std()
    return (spread - roll_mean) / roll_std


def compute_sscore(
    spread: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    """
    OU S-score (loop version, no look-ahead bias).

    Fits ΔX_t = a + b·X_{t-1} + ε over a rolling window of length `lookback`.
    Recovery:
        mu       = −a / b          (long-run mean)
        sigma_eq = sqrt(var(ε) / −2b)   (equilibrium std)
    S_t = (X_t − mu) / sigma_eq

    Windows with b ≥ 0 (not mean-reverting) or b ≤ −2 (unstable) return NaN.
    """
    delta_x = spread.diff()
    x_lag   = spread.shift(1)
    n       = len(spread)
    sscore  = pd.Series(np.nan, index=spread.index)

    for t in range(lookback, n):
        dx = delta_x.iloc[t - lookback: t].values
        xl = x_lag.iloc[t - lookback: t].values

        mask = np.isfinite(dx) & np.isfinite(xl)
        if mask.sum() < lookback // 2:
            continue

        dx_w, xl_w = dx[mask], xl[mask]
        X_mat = np.column_stack([np.ones(len(dx_w)), xl_w])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_mat, dx_w, rcond=None)
        except np.linalg.LinAlgError:
            continue

        a, b = coeffs
        if b >= 0 or b <= -2:
            continue

        mu    = -a / b
        resid = dx_w - X_mat @ coeffs
        var_e = np.var(resid, ddof=2)
        if var_e <= 0:
            continue

        sigma_eq    = np.sqrt(var_e / (-2.0 * b))
        sscore.iloc[t] = (spread.iloc[t] - mu) / sigma_eq

    return sscore


def compute_sscore_fast(
    spread: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    """
    Vectorized OU S-score — identical semantics to compute_sscore, ~100× faster.

    At position t the regression window covers [t-lookback, t-1] (no look-ahead):
        dy = spread.diff().shift(1)   →  ΔX_{t-1} at position t
        xl = spread.shift(2)          →  X_lag at t-1 = spread[t-2]

    OLS normal-equation identities:
        b       = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
        a       = (Σy − b·Σx) / n
        SSE     = Σy² − a·Σy − b·Σxy
        var_eps = SSE / (n − 2)
        sigma_eq = sqrt(var_eps / −2b)
        S_t     = (spread_t − mu) / sigma_eq
    """
    dy = spread.diff().shift(1)   # ΔX_{t-1} seen at t
    xl = spread.shift(2)          # X_lag at t-1

    n = lookback

    sum_y  = dy.rolling(n).sum()
    sum_x  = xl.rolling(n).sum()
    sum_xy = (dy * xl).rolling(n).sum()
    sum_x2 = (xl ** 2).rolling(n).sum()
    sum_y2 = (dy ** 2).rolling(n).sum()

    denom = n * sum_x2 - sum_x ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        b      = (n * sum_xy - sum_x * sum_y) / denom
        a      = (sum_y - b * sum_x) / n
        mu     = -a / b
        sse    = sum_y2 - a * sum_y - b * sum_xy
        var_e  = sse / (n - 2)
        sig_eq = np.sqrt(var_e / (-2.0 * b))
        s      = (spread - mu) / sig_eq

    invalid = (denom.abs() < 1e-10) | (b >= 0) | (b <= -2) | (var_e <= 0)
    return s.where(~invalid)


def generate_signals(
    zscore: pd.Series,
    entry: float = 2.0,
    exit_: float = 0.5,
) -> pd.Series:
    """
    Stateful trading rules:
      Open long   if z < −entry   (spread too low  → buy SPY, sell basket)
      Open short  if z >  entry   (spread too high → sell SPY, buy basket)
      Exit long   if z > −exit_
      Exit short  if z <  exit_

    Returns Series of positions: {−1, 0, +1}
    """
    position = pd.Series(0, index=zscore.index, dtype=int)
    current  = 0

    for t, z in zscore.items():
        if np.isnan(z):
            position[t] = 0
            continue

        if current == 0:
            if z < -entry:
                current = 1
            elif z > entry:
                current = -1
        elif current == 1:
            if z > -exit_:
                current = 0
        elif current == -1:
            if z < exit_:
                current = 0

        position[t] = current

    return position


def simulate_pnl(
    spread: pd.Series,
    positions: pd.Series,
    lag: int = 0,
) -> pd.Series:
    """
    Daily P&L = position_{t-1-lag} × (spread_t − spread_{t-1})
    Spread is in log-price space, so this approximates log-returns.
    """
    spread_return = spread.diff()
    pnl = positions.shift(1 + lag) * spread_return
    return pnl.fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Performance metrics
# ═══════════════════════════════════════════════════════════════════════════════

def sharpe_ratio(pnl: pd.Series) -> float:
    """Annualised Sharpe (0% risk-free rate)."""
    std = pnl.std()
    return 0.0 if std == 0 else (pnl.mean() / std) * np.sqrt(TRADING_DAYS)


def annualised_return(pnl: pd.Series) -> float:
    """Annualised return (%) from daily log-returns."""
    n_years = len(pnl) / TRADING_DAYS
    return ((np.exp(pnl.sum() / n_years) - 1) * 100) if n_years > 0 else 0.0


def cumulative_return_pct(pnl: pd.Series) -> float:
    """Total cumulative return (%)."""
    return (np.exp(pnl.sum()) - 1) * 100


def maximum_drawdown(pnl: pd.Series) -> Dict[str, float]:
    """Returns mdd_pct (negative %) and mdd_duration (trading days)."""
    cum      = pnl.cumsum()
    roll_max = cum.cummax()
    drawdown = cum - roll_max
    mdd_pct  = (np.exp(drawdown.min()) - 1) * 100

    max_dur = cur_dur = 0
    for v in drawdown < 0:
        if v:
            cur_dur += 1
            max_dur  = max(max_dur, cur_dur)
        else:
            cur_dur  = 0

    return {"mdd_pct": mdd_pct, "mdd_duration": max_dur}


def beta_to_index(strategy_pnl: pd.Series, index_pnl: pd.Series) -> float:
    """OLS beta of strategy vs index daily returns."""
    s, i = strategy_pnl.align(index_pnl, join="inner")
    s, i = s.dropna(), i.dropna()
    if len(s) < 10:
        return np.nan
    cov = np.cov(s, i)
    return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan


def full_metrics(
    strategy_pnl: pd.Series,
    index_pnl: pd.Series,
    label: str = "Strategy",
) -> pd.Series:
    """All metrics matching Drakos (2016) tables."""
    mdd = maximum_drawdown(strategy_pnl)
    return pd.Series({
        "Label":               label,
        "Cumulative Return %": round(cumulative_return_pct(strategy_pnl), 3),
        "Annual Return %":     round(annualised_return(strategy_pnl), 3),
        "Sharpe Ratio":        round(sharpe_ratio(strategy_pnl), 3),
        "Beta":                round(beta_to_index(strategy_pnl, index_pnl), 3),
        "Max Drawdown %":      round(mdd["mdd_pct"], 3),
        "MDD Duration (days)": mdd["mdd_duration"],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Grid-search helpers
# ═══════════════════════════════════════════════════════════════════════════════

def signal_metrics(pnl: pd.Series, positions: pd.Series) -> dict:
    """
    Sharpe, Calmar, trade_on_pct, turnover_yr from a stitched daily P&L.
    pnl is in log-return space.
    """
    pnl = pnl.fillna(0.0)
    n   = len(pnl)
    std = pnl.std()

    if n == 0 or std == 0:
        return dict(sharpe=np.nan, calmar=np.nan, trade_on_pct=np.nan,
                    turnover_yr=np.nan, n_days=n, n_trades=0)

    sr         = pnl.mean() / std * np.sqrt(TRADING_DAYS)
    n_years    = n / TRADING_DAYS
    ann_ret    = np.expm1(pnl.sum() / n_years)
    cum        = pnl.cumsum().apply(np.exp)
    max_dd     = float((cum / cum.cummax() - 1).min())
    calmar     = ann_ret / abs(max_dd) if max_dd < -1e-6 else np.nan

    trade_on_pct = float((positions != 0).mean())
    n_trades     = int((positions.diff().fillna(0) != 0).sum())
    turnover_yr  = n_trades / n_years if n_years > 0 else np.nan

    return dict(
        sharpe=sr, calmar=calmar,
        trade_on_pct=trade_on_pct, turnover_yr=turnover_yr,
        n_days=n, n_trades=n_trades,
    )


def compute_period_spreads(
    period: dict,
    index_ticker: str,
    fallback_ticker: str,
    sp100_tickers: List[str],
    adf_threshold: float,
    rolling_window: int,
    kf_v_w: float,
    kf_v_e: float,
    max_warmup: int,
) -> Optional[dict]:
    """
    Download prices, select basket (with fallback), compute three spreads.
    The extended slice includes `max_warmup` in-sample days so all lookback
    values in the grid have enough history on the first OOS day.

    Returns dict with keys:
        trade_year   : str
        spy_pnl      : pd.Series (OOS daily log-returns of SPY)
        spreads      : dict[label -> (spread_ext, spread_oos)]
    or None if the period must be skipped.
    """
    ins_start = period["insample_start"]
    ins_end   = period["insample_end"]
    trade_end = period["trade_end"]

    prices = fetch_prices(
        [index_ticker, fallback_ticker] + sp100_tickers,
        start=ins_start, end=trade_end,
    )
    if index_ticker not in prices.columns:
        return None

    log_prices = get_log_prices(prices)
    log_spy    = log_prices[index_ticker]
    log_stocks = log_prices.drop(columns=[index_ticker])

    insample_mask  = log_prices.index <= ins_end
    log_spy_ins    = log_spy[insample_mask]
    log_stocks_ins = log_stocks[insample_mask]

    # ── Basket selection ──────────────────────────────────────────────────────
    screen     = log_stocks_ins.drop(columns=[fallback_ticker], errors="ignore")
    candidates = screen_individual_cointegration(
        log_spy_ins, screen, adf_threshold
    )

    basket_tickers = gamma_const = None
    if candidates:
        basket_tickers, gamma_const, _ = build_synthetic_basket(
            log_spy_ins, log_stocks_ins[candidates], adf_threshold
        )

    if basket_tickers is None:
        if fallback_ticker not in log_stocks_ins.columns:
            return None
        basket_tickers = [fallback_ticker]
        oef_sum        = log_stocks_ins[[fallback_ticker]].sum(axis=1)
        gamma_const, _ = ols_hedge_ratio(log_spy_ins, oef_sum.to_frame())

    # ── OOS and extended slices ───────────────────────────────────────────────
    oos_mask       = log_prices.index > ins_end
    log_spy_oos    = log_spy[oos_mask]
    log_basket_oos = log_stocks[oos_mask][basket_tickers]

    lb             = max(0, insample_mask.sum() - max_warmup)
    ext_idx        = log_prices.index[lb:]
    log_spy_ext    = log_spy.loc[ext_idx]
    log_basket_ext = log_stocks.loc[ext_idx, basket_tickers]

    # ── Three spreads ─────────────────────────────────────────────────────────
    sp_const_ext = spread_constant_ols(log_spy_ext, log_basket_ext, gamma_const)
    sp_const_oos = sp_const_ext.loc[log_spy_oos.index]

    sp_roll_ext, _ = spread_rolling_ols(log_spy_ext, log_basket_ext, rolling_window)
    sp_roll_oos    = sp_roll_ext.loc[log_spy_oos.index]

    gamma0        = fit_kalman_init(log_spy_ins, log_stocks_ins[basket_tickers])
    sp_kf_ext, _  = spread_kalman_filter(
        log_spy_ext, log_basket_ext, gamma0, kf_v_w, kf_v_e
    )
    sp_kf_oos = sp_kf_ext.loc[log_spy_oos.index]

    return {
        "trade_year": period["trade_start"][:4],
        "spy_pnl":    log_spy_oos.diff().fillna(0),
        "spreads": {
            "MKFA":         (sp_kf_ext,    sp_kf_oos),
            "Rolling OLS":  (sp_roll_ext,  sp_roll_oos),
            "Constant OLS": (sp_const_ext, sp_const_oos),
        },
    }


def eval_combo(
    periods_cache: List[dict],
    spread_label: str,
    score_fn,
    lookback: int,
    entry: float,
    exit_: float,
    lag: int = 0,
) -> dict:
    """
    Apply one (score_fn, lookback, entry, exit_) combo to all cached periods,
    stitch the P&L, and return signal_metrics.
    """
    all_pnl: List[pd.Series] = []
    all_pos: List[pd.Series] = []

    for cache in periods_cache:
        spread_ext, spread_oos = cache["spreads"][spread_label]
        s       = spread_ext.dropna()
        score   = score_fn(s, lookback)
        pos_ext = generate_signals(score, entry, exit_)
        pos     = pos_ext.reindex(spread_oos.index).fillna(0).astype(int)
        pnl     = simulate_pnl(spread_oos, pos, lag=lag)
        all_pnl.append(pnl)
        all_pos.append(pos)

    stitched_pnl = pd.concat(all_pnl).sort_index()
    stitched_pos = pd.concat(all_pos).sort_index()
    stitched_pnl = stitched_pnl[~stitched_pnl.index.duplicated(keep="first")]
    stitched_pos = stitched_pos[~stitched_pos.index.duplicated(keep="first")]

    return signal_metrics(stitched_pnl, stitched_pos)


def run_grid(
    periods_cache: List[dict],
    spread_labels: List[str],
    lookbacks: List[int],
    entry_z_vals: List[float],
    exit_z_vals: List[float],
    entry_s_vals: List[float],
    exit_s_vals: List[float],
    lag: int = 0,
    quick: bool = False,
) -> pd.DataFrame:
    """
    Sweep all (score_type, spread, lookback, entry, exit) combinations.
    Returns a DataFrame of metrics for each combination.
    """
    if quick:
        lookbacks    = [60]
        entry_z_vals = [2.0]
        exit_z_vals  = [0.5]
        entry_s_vals = [1.25]
        exit_s_vals  = [0.5]

    rows: List[dict] = []

    for score_type, entry_vals, exit_vals, score_fn in [
        ("zscore", entry_z_vals, exit_z_vals, compute_zscore),
        ("sscore", entry_s_vals, exit_s_vals, compute_sscore_fast),
    ]:
        combos = [
            (sl, lb, en, ex)
            for sl, lb, en, ex in itertools.product(
                spread_labels, lookbacks, entry_vals, exit_vals
            )
            if ex < en
        ]
        total = len(combos)
        print(f"  {score_type}: {total} combinations "
              f"({len(spread_labels)} spreads × {len(lookbacks)} lookbacks)")

        for i, (spread_label, lookback, entry, exit_) in enumerate(combos):
            m = eval_combo(periods_cache, spread_label, score_fn,
                           lookback, entry, exit_, lag=lag)
            rows.append({
                "score_type": score_type,
                "spread":     spread_label,
                "lookback":   lookback,
                "entry":      entry,
                "exit":       exit_,
                **m,
            })
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"    {i+1}/{total} done")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main(
    *,
    index_ticker: str,
    fallback_ticker: str,
    sp100_tickers: List[str],
    backtest_periods: List[dict],
    adf_threshold: float,
    rolling_window: int,
    kf_v_w: float,
    kf_v_e: float,
    lag: int,
    lookbacks: List[int],
    entry_z_vals: List[float],
    exit_z_vals: List[float],
    entry_s_vals: List[float],
    exit_s_vals: List[float],
    spread_labels: List[str],
    output_dir: Path,
    quick: bool = False,
) -> None:
    output_dir.mkdir(exist_ok=True)
    max_warmup = max(lookbacks)

    # ── Step 1: cache spreads ────────────────────────────────────────────────
    print(f"Computing spreads for {len(backtest_periods)} walk-forward periods ...")
    periods_cache: List[dict] = []

    for i, period in enumerate(backtest_periods):
        label = (f"{period['insample_start'][:4]} IS "
                 f"/ {period['trade_start'][:4]} OOS")
        print(f"  [{i+1:02d}/{len(backtest_periods)}] {label}", end="  ", flush=True)
        data = compute_period_spreads(
            period,
            index_ticker=index_ticker,
            fallback_ticker=fallback_ticker,
            sp100_tickers=sp100_tickers,
            adf_threshold=adf_threshold,
            rolling_window=rolling_window,
            kf_v_w=kf_v_w,
            kf_v_e=kf_v_e,
            max_warmup=max_warmup,
        )
        if data is None:
            print("SKIP")
            continue
        periods_cache.append(data)
        print(f"OK  ({data['trade_year']} trade year)")

    print(f"\nCached {len(periods_cache)} active periods.\n")

    # ── Step 2: grid search ──────────────────────────────────────────────────
    print("Running grid search ...")
    results = run_grid(
        periods_cache=periods_cache,
        spread_labels=spread_labels,
        lookbacks=lookbacks,
        entry_z_vals=entry_z_vals,
        exit_z_vals=exit_z_vals,
        entry_s_vals=entry_s_vals,
        exit_s_vals=exit_s_vals,
        lag=lag,
        quick=quick,
    )

    # ── Step 3: save + summarise ─────────────────────────────────────────────
    out_path = output_dir / "tune_results.csv"
    results.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nSaved {len(results)} rows → {out_path}")

    cols_show = ["score_type", "spread", "lookback", "entry", "exit",
                 "sharpe", "calmar", "trade_on_pct", "turnover_yr"]

    for score_type in ("zscore", "sscore"):
        sub = results[results.score_type == score_type]
        if sub.empty:
            continue
        print(f"\n── Top 10 by Sharpe  [{score_type}] ──")
        print(sub.nlargest(10, "sharpe")[cols_show].to_string(index=False))

    print("\n── Top 10 overall by Sharpe ──")
    print(results.nlargest(10, "sharpe")[cols_show].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — edit here, then run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import datetime

    # ── CLI flags ─────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Synthetic basket stat-arb grid search (self-contained)."
    )
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test with a single param per axis.")
    parser.add_argument("--spread", choices=SPREAD_LABELS, default=None,
                        help="Restrict grid to one spread type.")
    args = parser.parse_args()

    # ── Universe ──────────────────────────────────────────────────────────────
    INDEX_TICKER    = "SPY"
    FALLBACK_TICKER = "OEF"        # S&P100 ETF — used when synthetic basket fails

    SP100_TICKERS = [
        "AAPL", "ABBV", "ABT",  "ACN",  "AIG",  "ALL",  "AMGN", "AMT",  "AMZN", "AXP",
        "BA",   "BAC",  "BAX",  "BK",   "BLK",  "BMY",  "BRK-B","C",    "CAT",  "CHTR",
        "CL",   "CMCSA","COF",  "COP",  "COST", "CRM",  "CVS",  "CVX",  "DD",   "DE",
        "DHR",  "DIS",  "DOW",  "DVN",  "EMR",  "EXC",  "F",    "FCX",  "FDX",  "GD",
        "GE",   "GILD", "GM",   "GOOG", "GS",   "HAL",  "HD",   "HON",  "IBM",  "INTC",
        "JNJ",  "JPM",  "KO",   "LLY",  "LMT",  "LOW",  "MA",   "MCD",  "MDLZ", "MDT",
        "MET",  "MMM",  "MO",   "MON",  "MRK",  "MS",   "MSFT", "NEE",  "NFLX", "NSC",
        "NVDA", "ORCL", "OXY",  "PEP",  "PFE",  "PG",   "PM",   "PYPL", "QCOM", "RTX",
        "SBUX", "SLB",  "SO",   "SPG",  "T",    "TGT",  "TXN",  "UNH",  "UNP",  "UPS",
        "USB",  "V",    "VZ",   "WBA",  "WFC",  "WMT",  "XOM",
    ]

    # ── Walk-forward periods ──────────────────────────────────────────────────
    # 1-year in-sample → 1-year OOS; in-sample 2006–2025, trade 2007–2026
    _TODAY = datetime.date.today().isoformat()

    BACKTEST_PERIODS = [
        {
            "insample_start": f"{y}-01-01",
            "insample_end":   f"{y}-12-31",
            "trade_start":    f"{y+1}-01-01",
            "trade_end":      f"{y+2}-01-01" if y + 1 < 2026 else _TODAY,
        }
        for y in range(2006, 2026)
    ]

    # ── Cointegration ─────────────────────────────────────────────────────────
    ADF_PVALUE_THRESHOLD = 0.05   # Reject unit root if p < this

    # ── Hedge ratio ───────────────────────────────────────────────────────────
    ROLLING_WINDOW = 60           # days for rolling OLS
    KF_V_W         = 1e-4         # KF process noise
    KF_V_E         = 1e-3         # KF measurement noise

    # ── Execution ─────────────────────────────────────────────────────────────
    LAG = 1                       # signal lag in days; 0 = same-day execution, 1 = next-day, etc.

    # ── Parameter grid ────────────────────────────────────────────────────────
    LOOKBACKS    = [30, 45, 60, 90, 120]
    ENTRY_Z_VALS = [1.25, 1.5, 2.0, 2.5, 3.0]
    EXIT_Z_VALS  = [0.0,  0.25, 0.5, 0.75, 1.0]
    ENTRY_S_VALS = [0.5,  0.75, 1.0, 1.25, 1.5, 2.0]
    EXIT_S_VALS  = [0.0,  0.25, 0.5, 0.75]

    # ── Output ────────────────────────────────────────────────────────────────
    OUTPUT_DIR = Path(__file__).resolve().parent / "output"

    # ── Run ───────────────────────────────────────────────────────────────────
    main(
        index_ticker     = INDEX_TICKER,
        fallback_ticker  = FALLBACK_TICKER,
        sp100_tickers    = SP100_TICKERS,
        backtest_periods = BACKTEST_PERIODS,
        adf_threshold    = ADF_PVALUE_THRESHOLD,
        rolling_window   = ROLLING_WINDOW,
        kf_v_w           = KF_V_W,
        kf_v_e           = KF_V_E,
        lag              = LAG,
        lookbacks        = LOOKBACKS,
        entry_z_vals     = ENTRY_Z_VALS,
        exit_z_vals      = EXIT_Z_VALS,
        entry_s_vals     = ENTRY_S_VALS,
        exit_s_vals      = EXIT_S_VALS,
        spread_labels    = [args.spread] if args.spread else SPREAD_LABELS,
        output_dir       = OUTPUT_DIR,
        quick            = args.quick,
    )

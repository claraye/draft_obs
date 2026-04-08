"""
backtest.py — Walk-forward vol-managed portfolio backtest for the
Volume-Driven Intraday Volatility replication.

Strategy:
  - Generate target weights on each 5-minute bar T from the bar-level forecast
  - Trade during bar T+1
  - Realize the resulting PnL on bar T+2
  - Compound bar-level PnL back to daily returns for reporting

Benchmarks:
  - Buy-and-hold (no vol scaling)
  - GARCH(1,1) vol-managed (simple recursion)
  - AR1-RV vol-managed (rolling 21-day realized vol as forecast)

Walk-forward design:
  - Fit model once on TRAIN_FRAC of the sample
  - Test on the remaining OOS period
  - No re-fitting (to match paper's full-sample estimation, and because
    yfinance 5-minute data is short ~60 days)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    dates:          list
    returns:        dict[str, np.ndarray] = field(default_factory=dict)
    mz_r2_train:   float = np.nan
    mz_r2_oos:     float = np.nan


# ── GARCH(1,1) baseline ───────────────────────────────────────────────────────

def _fit_garch_variance(rets: np.ndarray,
                        omega: float = 1e-6,
                        alpha: float = 0.10,
                        beta:  float = 0.85) -> np.ndarray:
    """
    Simple GARCH(1,1) recursion with fixed parameters.
    Returns array of one-step-ahead variance forecasts (same length as rets).
    """
    n  = len(rets)
    h  = np.full(n, np.var(rets[:21]) if n > 21 else 1e-4)
    for t in range(1, n):
        h[t] = omega + alpha * rets[t - 1] ** 2 + beta * h[t - 1]
    return h


# ── Rolling RV baseline ───────────────────────────────────────────────────────

def _rolling_rv_variance(rets: np.ndarray, window: int = 21) -> np.ndarray:
    """Rolling 21-day realized variance as vol forecast."""
    rv = pd.Series(rets ** 2).rolling(window, min_periods=5).mean().values
    rv = np.where(np.isnan(rv), np.nanmean(rets ** 2), rv)
    return rv


# ── Vol-scaled return helper ──────────────────────────────────────────────────

def _target_positions_from_vol(
    vol_forecast: np.ndarray,
    target_vol: float,
    bars_per_day: int,
) -> np.ndarray:
    """Convert per-bar vol forecasts to target annualized-risk weights."""
    ann_vol_forecast = vol_forecast * np.sqrt(252 * bars_per_day)
    ann_vol_forecast = np.clip(ann_vol_forecast, 0.01, 5.0)

    pos = target_vol / ann_vol_forecast
    return np.clip(pos, 0.0, 3.0)


def _apply_execution_lag(
    target_pos: np.ndarray,
    raw_rets: np.ndarray,
    signal_lag_bars: int,
    tc_bps: float,
) -> np.ndarray:
    """
    Apply the T -> T+2 realization lag to bar-level target positions.

    Target weight formed on bar T is traded during bar T+1 and therefore
    affects realized PnL on bar T+2.
    """
    n = len(raw_rets)
    realized = np.full(n, np.nan, dtype=float)
    exec_pos = np.full(n, np.nan, dtype=float)
    if signal_lag_bars < n:
        exec_pos[signal_lag_bars:] = target_pos[:-signal_lag_bars]

    tc_rate = tc_bps * 1e-4
    prev_pos = 0.0
    for i in range(n):
        if not np.isfinite(exec_pos[i]) or not np.isfinite(raw_rets[i]):
            continue
        delta_pos = abs(exec_pos[i] - prev_pos)
        realized[i] = exec_pos[i] * raw_rets[i] - delta_pos * tc_rate
        prev_pos = exec_pos[i]

    return realized


def _compound_to_daily(bar_rets: np.ndarray, dates: pd.Series) -> pd.Series:
    """Compound bar-level returns into daily simple returns."""
    df = pd.DataFrame({"date": dates, "ret": bar_rets})
    daily = (
        df.dropna(subset=["ret"])
        .groupby("date")["ret"]
        .apply(lambda x: float(np.prod(1.0 + x.values) - 1.0))
    )
    daily.name = "daily_ret"
    return daily


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    intraday_data: dict[str, pd.DataFrame],
    ticker:        str,
    interval_labels: list[int],
    train_frac:    float = 0.70,
    target_ann_vol: float = 0.15,
    tc_bps:        float = 1.0,
    log_var_clip:  float = -20.0,
    mcmc_n_iter:   int = 120,
    mcmc_burn_in:  int = 40,
    mcmc_thin:     int = 2,
    sunday_open_lags: int = 12,
    random_state:  int = 42,
) -> BacktestResult:
    """
    Run walk-forward vol-managed backtest for a single ticker.

    Parameters
    ----------
    intraday_data  : output of data_loader.fetch_intraday()
    ticker         : e.g. "SPY"
    interval_labels: list of HHMM interval labels (e.g. [930, 1030, ...])
    train_frac     : fraction of sample for training (model fit)
    target_ann_vol : annualized vol target for vol-managed portfolio

    Returns
    -------
    BacktestResult
    """
    from data_loader import prepare_intraday, build_volume_deviation
    from model import VolumeToD_Model

    if ticker not in intraday_data:
        logger.error(f"Ticker {ticker} not in intraday_data")
        return BacktestResult(dates=[])

    raw = intraday_data[ticker]

    # Prepare panel (no vol_dev yet — must split first to avoid look-ahead bias)
    panel = prepare_intraday(raw, interval_labels=interval_labels,
                             log_var_clip=log_var_clip)

    # ── Train / test split BEFORE volume normalisation ────────────────────────
    # The paper computes V̄_{k*} and σ_{V,k*} from the estimation sample only.
    # Calling build_volume_deviation() on the full panel would leak the test
    # period's volume distribution into v_{t-1}, inflating OOS performance.
    # Fix: compute vol_dev stats on training data, apply those stats to test.
    all_dates = sorted(panel["date"].unique())
    n_train   = int(len(all_dates) * train_frac)
    train_dates = set(all_dates[:n_train])
    test_dates  = set(all_dates[n_train:])

    panel_train_raw = panel[panel["date"].isin(train_dates)].copy()
    panel_test_raw  = panel[panel["date"].isin(test_dates)].copy()

    # Build vol_dev on training data using its own statistics
    panel_train = build_volume_deviation(panel_train_raw)

    # Apply training-sample vol stats to test panel (no look-ahead)
    train_vol_stats = (
        panel_train
        .groupby("interval")["log_vol"]
        .agg(vol_mean="mean", vol_std="std")
    )
    train_vol_stats["vol_std"] = train_vol_stats["vol_std"].clip(lower=1e-8)
    panel_test = panel_test_raw.copy()
    mu_test  = panel_test["interval"].map(train_vol_stats["vol_mean"])
    sig_test = panel_test["interval"].map(train_vol_stats["vol_std"]).clip(lower=1e-8)
    panel_test["vol_dev"] = (panel_test["log_vol"] - mu_test) / sig_test

    if len(panel_train) < 50:
        logger.error(f"  {ticker}: insufficient training data ({len(panel_train)} bars)")
        return BacktestResult(dates=[])

    # Fit model
    model = VolumeToD_Model(
        log_var_clip=log_var_clip,
        n_iter=mcmc_n_iter,
        burn_in=mcmc_burn_in,
        thin=mcmc_thin,
        sunday_open_lags=sunday_open_lags,
        random_state=random_state,
    )
    model.fit(panel_train)

    # In-sample diagnostics
    train_pred = model.predict(panel_train)
    mz_train   = model.mz_r2(train_pred)
    logger.info(f"  {ticker}: in-sample MZ R² = {mz_train:.4f}")

    # OOS forecast
    x0 = float(train_pred["x_t"].iloc[-1]) if len(train_pred) else 0.0
    oos_pred  = model.predict(panel_test, x0=x0)
    mz_oos    = model.mz_r2(oos_pred)
    logger.info(f"  {ticker}: OOS MZ R² = {mz_oos:.4f}")

    bars_per_day = max(len(interval_labels), 1)
    signal_lag_bars = 2
    bar_rets = oos_pred["bar_ret"].to_numpy(dtype=float)
    bar_dates = pd.Series(oos_pred["date"].values)

    if len(bar_rets) < signal_lag_bars + 10:
        logger.warning(f"  {ticker}: too few OOS bars ({len(bar_rets)})")
        return BacktestResult(dates=[])

    # ── Strategy returns ───────────────────────────────────────────────────────
    # 1. Buy-and-hold daily return from compounding 5-minute bars
    bh_daily = _compound_to_daily(bar_rets, bar_dates)

    # 2. Volume-ToD vol-managed at the bar level
    vtod_bar_vol = np.sqrt(np.exp(oos_pred["h_pred"].clip(log_var_clip, 0.0).to_numpy()))
    vtod_target = _target_positions_from_vol(vtod_bar_vol, target_ann_vol, bars_per_day)
    vtod_bar_rets = _apply_execution_lag(vtod_target, bar_rets, signal_lag_bars, tc_bps)
    vtod_daily = _compound_to_daily(vtod_bar_rets, bar_dates)

    # 3. GARCH vol-managed at the bar level
    garch_var = _fit_garch_variance(bar_rets)
    garch_vol = np.sqrt(garch_var)
    garch_target = _target_positions_from_vol(garch_vol, target_ann_vol, bars_per_day)
    garch_bar_rets = _apply_execution_lag(garch_target, bar_rets, signal_lag_bars, tc_bps)
    garch_daily = _compound_to_daily(garch_bar_rets, bar_dates)

    # 4. Rolling RV vol-managed at the bar level
    rv_var = _rolling_rv_variance(bar_rets)
    rv_vol = np.sqrt(rv_var)
    rv_target = _target_positions_from_vol(rv_vol, target_ann_vol, bars_per_day)
    rv_bar_rets = _apply_execution_lag(rv_target, bar_rets, signal_lag_bars, tc_bps)
    rv_daily = _compound_to_daily(rv_bar_rets, bar_dates)

    daily_df = pd.concat(
        [bh_daily, vtod_daily, garch_daily, rv_daily],
        axis=1,
        keys=["buy_hold", "vtod_vm", "garch_vm", "rv21_vm"],
    ).dropna(how="all")

    if len(daily_df) < 2:
        logger.warning(f"  {ticker}: too few OOS days ({len(daily_df)})")
        return BacktestResult(dates=[])

    return BacktestResult(
        dates=daily_df.index.tolist(),
        returns={
            "buy_hold":  daily_df["buy_hold"].to_numpy(dtype=float),
            "vtod_vm":   daily_df["vtod_vm"].to_numpy(dtype=float),
            "garch_vm":  daily_df["garch_vm"].to_numpy(dtype=float),
            "rv21_vm":   daily_df["rv21_vm"].to_numpy(dtype=float),
        },
        mz_r2_train=mz_train,
        mz_r2_oos=mz_oos,
    )

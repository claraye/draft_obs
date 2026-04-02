"""
backtest.py — Walk-forward backtest engine
Paper Section 4.1: retrain every 5 years; OOS from 2000 to 2022.

Training schedule (rolling expanding window):
  Train 1990-1999 → Test 2000-2004
  Train 1990-2004 → Test 2005-2009
  Train 1990-2009 → Test 2010-2014
  Train 1990-2014 → Test 2015-2019
  Train 1990-2019 → Test 2020-2022

For each period:
  1. Last 10% of training data → validation set for (α, β) grid search
  2. Fit graph ensemble on full training data with best (α, β)
  3. Fit OLS regression cross-sectionally
  4. Generate OOS signals and compute PnL

Strategies evaluated in parallel:
  - GMOM:    network momentum OLS
  - LinReg:  individual momentum OLS
  - MACD:    model-free (no fitting)
  - LongOnly: always long
"""
from __future__ import annotations

import os
import pickle
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    dates: pd.DatetimeIndex
    returns: dict = field(default_factory=dict)       # {strategy: np.array}
    positions: dict = field(default_factory=dict)     # {strategy: pd.DataFrame}
    turnovers: dict = field(default_factory=dict)     # {strategy: np.array}
    graphs: dict = field(default_factory=dict)        # {date: (A_tilde, valid_idx)}
    reg_coefs: dict = field(default_factory=dict)     # {period: beta_coef}


def run_backtest(
    prices: pd.DataFrame,
    feature_df: pd.DataFrame,    # MultiIndex columns: (feature_name, asset)
    feature_tensor: np.ndarray,  # (T, N, 8) aligned to prices.index
    returns_df: pd.DataFrame,    # (T, N) log returns
    sigma_df: pd.DataFrame,      # (T, N) daily EWMA vol
    assets: list[str],
    backtest_start: str,
    backtest_end: str,
    retrain_years: int,
    val_fraction: float,
    lookback_windows: list[int],
    alpha_grid: list[float],
    beta_grid: list[float],
    vol_target: float,
    graph_refit_freq: int,
    solver_prefs: list[str],
    cache_dir: str,
    fast_mode: bool = False,  # Skip grid search, use default α=0.01, β=0.01
) -> BacktestResult:
    """
    Main walk-forward backtest.

    Args:
      fast_mode: Skip hyperparameter search (use α=0.01, β=0.01).
                 Much faster but less accurate to the paper.
    """
    from graph_learning import ensemble_graph, normalise_graph, select_hyperparams
    from signals import (propagate_features, ols_train, ols_predict,
                         macd_signal, build_training_data)

    os.makedirs(cache_dir, exist_ok=True)

    all_dates = prices.index
    bt_mask   = (all_dates >= backtest_start) & (all_dates <= backtest_end)
    bt_dates  = all_dates[bt_mask]
    N         = len(assets)
    T         = len(all_dates)

    # Volatility-scaled 1-day return (target variable)
    # vsr_1d = log_return_1d / (sigma_t * sqrt(1))
    vsr_1d = returns_df.values / (sigma_df.values + 1e-10)  # (T, N)

    result = BacktestResult(dates=bt_dates)
    for strat in ["GMOM", "LinReg", "MACD", "LongOnly"]:
        result.returns[strat]   = np.full(len(bt_dates), np.nan)
        result.turnovers[strat] = np.full(len(bt_dates), np.nan)

    # Determine training periods
    periods = _build_training_periods(
        all_dates, backtest_start, backtest_end, retrain_years
    )
    logger.info(f"Backtest periods: {[(str(p[0])[:10], str(p[1])[:10], str(p[2])[:10]) for p in periods]}")

    prev_positions = {s: np.zeros(N) for s in ["GMOM", "LinReg", "MACD", "LongOnly"]}

    for train_start, train_end, test_end in periods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Period: train={str(train_start)[:10]}→{str(train_end)[:10]}, test={str(train_end)[:10]}→{str(test_end)[:10]}")

        # Index ranges
        train_mask  = (all_dates >= train_start) & (all_dates < train_end)
        test_mask   = (all_dates >= train_end) & (all_dates <= test_end)
        train_tvals = np.where(train_mask)[0]
        test_tvals  = np.where(test_mask)[0]

        if len(train_tvals) < 252 or len(test_tvals) == 0:
            logger.warning("Insufficient data for this period, skipping.")
            continue

        # Split train / validation
        n_val = max(1, int(len(train_tvals) * val_fraction))
        val_t   = train_tvals[-n_val:]
        fit_t   = train_tvals[:-n_val]

        # ── Step 1: Graph computation on training period ──────────────────
        cache_key = f"graphs_{str(train_end)[:10]}_rf{graph_refit_freq}"
        graphs_cache = os.path.join(cache_dir, f"{cache_key}.pkl")

        if os.path.exists(graphs_cache):
            logger.info(f"Loading cached graphs from {graphs_cache}")
            with open(graphs_cache, "rb") as f:
                A_tilde_by_t = pickle.load(f)
        else:
            logger.info("Computing graphs on training set...")
            A_tilde_by_t = _compute_graphs_for_period(
                feature_tensor, train_tvals, assets, lookback_windows,
                alpha=0.01, beta=0.01,
                refit_freq=graph_refit_freq,
                solver_prefs=solver_prefs,
            )

        # ── Step 2: Hyperparameter selection ────────────────────────────────
        if fast_mode:
            best_alpha, best_beta = 0.01, 0.01
            logger.info("Fast mode: using α=0.01, β=0.01")
        else:
            logger.info("Running hyperparameter grid search...")
            best_alpha, best_beta = select_hyperparams(
                feature_tensor, fit_t, val_t,
                np.arange(N), alpha_grid, beta_grid,
                lookback_windows, solver_prefs, subsample=10
            )
            # Recompute graphs with best params
            A_tilde_by_t = _compute_graphs_for_period(
                feature_tensor, train_tvals, assets, lookback_windows,
                best_alpha, best_beta, refit_freq=graph_refit_freq, solver_prefs=solver_prefs
            )
            with open(graphs_cache, "wb") as f:
                pickle.dump(A_tilde_by_t, f)

        # ── Step 3: Fit OLS on training data ────────────────────────────────
        X_gmom, y_gmom = build_training_data(
            feature_tensor, vsr_1d, sigma_df.values,
            A_tilde_by_t, list(train_tvals), mode="gmom"
        )
        X_linreg, y_linreg = build_training_data(
            feature_tensor, vsr_1d, sigma_df.values,
            A_tilde_by_t, list(train_tvals), mode="linreg"
        )

        beta_gmom, intercept_gmom = (None, 0.0)
        beta_linreg, intercept_linreg = (None, 0.0)

        if X_gmom is not None and len(X_gmom) > 20:
            beta_gmom, intercept_gmom = ols_train(X_gmom, y_gmom)
            logger.info(f"GMOM OLS fitted on {len(X_gmom)} cross-sectional obs")

        if X_linreg is not None and len(X_linreg) > 20:
            beta_linreg, intercept_linreg = ols_train(X_linreg, y_linreg)
            logger.info(f"LinReg OLS fitted on {len(X_linreg)} cross-sectional obs")

        result.reg_coefs[str(train_end)[:10]] = {
            "gmom":   (beta_gmom, intercept_gmom),
            "linreg": (beta_linreg, intercept_linreg),
            "alpha":  best_alpha,
            "beta":   best_beta,
        }

        # ── Step 4: Compute graphs for test period ───────────────────────────
        logger.info("Computing graphs on test period...")
        A_tilde_test = _compute_graphs_for_period(
            feature_tensor, test_tvals, assets, lookback_windows,
            best_alpha, best_beta, refit_freq=graph_refit_freq, solver_prefs=solver_prefs
        )

        # ── Step 5: Generate signals and compute returns ─────────────────────
        logger.info("Generating signals and computing PnL...")
        _generate_oos_returns(
            result, bt_dates, all_dates, test_tvals,
            feature_tensor, feature_df, vsr_1d, sigma_df,
            returns_df, assets, A_tilde_test,
            beta_gmom, intercept_gmom,
            beta_linreg, intercept_linreg,
            vol_target, prev_positions,
        )

    return result


# ─── Helper functions ─────────────────────────────────────────────────────────

def _build_training_periods(
    all_dates: pd.DatetimeIndex,
    backtest_start: str,
    backtest_end: str,
    retrain_years: int,
) -> list[tuple]:
    """
    Build list of (train_start, train_end, test_end) periods.
    Train always starts from the very beginning of all_dates.
    """
    bt_start = pd.Timestamp(backtest_start)
    bt_end   = pd.Timestamp(backtest_end)
    train_start = all_dates[0]

    periods = []
    test_begin = bt_start
    while test_begin < bt_end:
        test_end = min(
            test_begin + pd.DateOffset(years=retrain_years) - pd.Timedelta(days=1),
            bt_end
        )
        # Snap to actual trading dates
        train_end_idx = all_dates.searchsorted(test_begin, side="left")
        test_end_idx  = all_dates.searchsorted(test_end, side="right") - 1

        if train_end_idx > 0 and test_end_idx > train_end_idx:
            periods.append((
                train_start,
                all_dates[train_end_idx],
                all_dates[test_end_idx],
            ))
        test_begin = test_end + pd.Timedelta(days=1)

    return periods


def _compute_graphs_for_period(
    feature_tensor: np.ndarray,
    t_indices: np.ndarray,
    assets: list[str],
    lookback_windows: list[int],
    alpha: float,
    beta: float,
    refit_freq: int,
    solver_prefs: list[str],
) -> dict:
    """
    Compute A_tilde for each date in t_indices, refitting every refit_freq days.
    Returns dict: {t_idx: (A_tilde, valid_idx)}
    """
    from graph_learning import ensemble_graph, normalise_graph
    from features import stack_lookback

    N_full = feature_tensor.shape[1]
    assets_mask = np.ones(N_full, dtype=bool)

    A_tilde_by_t = {}
    last_A = {}   # {frozenset(valid_idx): A_tilde}

    for i, t in enumerate(t_indices):
        # Determine valid assets at t (those with full history for min lookback)
        min_delta = min(lookback_windows)
        V_min, valid_idx = stack_lookback(feature_tensor, t, min_delta, assets_mask)

        if V_min is None or len(valid_idx) < 5:
            A_tilde_by_t[t] = (None, np.array([]))
            continue

        key = tuple(valid_idx.tolist())

        # Only refit every refit_freq steps
        if i % refit_freq == 0 or key not in last_A:
            A_bar = ensemble_graph(
                feature_tensor, t, valid_idx, alpha, beta,
                lookback_windows, solver_prefs
            )
            if A_bar is None:
                A_tilde_by_t[t] = (None, valid_idx)
                continue
            A_tilde = normalise_graph(A_bar)
            last_A[key] = A_tilde

        A_tilde_by_t[t] = (last_A.get(key), valid_idx)

        if i % 50 == 0:
            logger.info(f"  Graph progress: {i+1}/{len(t_indices)} (N_valid={len(valid_idx)})")

    return A_tilde_by_t


def _generate_oos_returns(
    result: BacktestResult,
    bt_dates: pd.DatetimeIndex,
    all_dates: pd.DatetimeIndex,
    test_tvals: np.ndarray,
    feature_tensor: np.ndarray,
    feature_df: pd.DataFrame,
    vsr_1d: np.ndarray,
    sigma_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    assets: list[str],
    A_tilde_test: dict,
    beta_gmom: Optional[np.ndarray],
    intercept_gmom: float,
    beta_linreg: Optional[np.ndarray],
    intercept_linreg: float,
    vol_target: float,
    prev_positions: dict,
) -> None:
    """
    For each test day t, compute positions and PnL.
    Signal at t → trade at t+1 → PnL at t+2 (paper's conservative assumption).
    """
    from signals import propagate_features, ols_predict, macd_signal

    N = feature_tensor.shape[1]

    for i, t in enumerate(test_tvals):
        # Need t+2 for PnL
        if t + 2 >= len(all_dates):
            continue

        date_t = all_dates[t]
        if date_t not in bt_dates:
            continue
        bt_i = bt_dates.get_loc(date_t)

        sigma_t   = sigma_df.values[t, :]           # (N,)
        sigma_t2  = sigma_df.values[t + 2, :]
        ret_t2    = returns_df.values[t + 2, :]     # actual log return at t+2

        A_tilde, valid_idx = A_tilde_test.get(t, (None, np.array([])))

        # ── GMOM signal ──────────────────────────────────────────────────
        x_gmom = np.zeros(N)
        if A_tilde is not None and beta_gmom is not None and len(valid_idx) > 0:
            u_t = feature_tensor[t, valid_idx, :]          # (N_valid, 8)
            u_net = propagate_features(u_t, A_tilde)       # (N_valid, 8)
            valid_rows = np.isfinite(u_net).all(axis=1)
            y_pred = ols_predict(u_net, beta_gmom, intercept_gmom)
            x_gmom[valid_idx[valid_rows]] = np.sign(y_pred[valid_rows])

        # ── LinReg signal ─────────────────────────────────────────────────
        x_linreg = np.zeros(N)
        if beta_linreg is not None:
            u_t_all = feature_tensor[t, :, :]              # (N, 8)
            valid_rows = np.isfinite(u_t_all).all(axis=1)
            y_pred_lr = ols_predict(u_t_all, beta_linreg, intercept_linreg)
            x_linreg[valid_rows] = np.sign(y_pred_lr[valid_rows])

        # ── MACD signal ───────────────────────────────────────────────────
        x_macd = macd_signal(feature_df, assets, date_t)

        # ── Long Only ─────────────────────────────────────────────────────
        x_long = np.ones(N)

        # ── Portfolio returns (Eq. 9) ────────────────────────────────────
        # r^portfolio = (1/N) Σ x_{i,t} * (σ_tgt/σ_{i,t+2}) * r_{i,t+2}
        # Using t+2 return (paper: signal@t → trade@t+1 → PnL@t+2)
        def _port_ret(x: np.ndarray) -> float:
            sig_scale = vol_target / (sigma_t2 * np.sqrt(252) + 1e-10)
            contrib = x * sig_scale * ret_t2
            valid = np.isfinite(contrib)
            if valid.sum() == 0:
                return 0.0
            return float(np.mean(contrib[valid]))

        result.returns["GMOM"][bt_i]    = _port_ret(x_gmom)
        result.returns["LinReg"][bt_i]  = _port_ret(x_linreg)
        result.returns["MACD"][bt_i]    = _port_ret(x_macd)
        result.returns["LongOnly"][bt_i] = _port_ret(x_long)

        # ── Turnover ─────────────────────────────────────────────────────
        from metrics import compute_turnover
        for strat, x_new in [
            ("GMOM", x_gmom), ("LinReg", x_linreg),
            ("MACD", x_macd), ("LongOnly", x_long)
        ]:
            result.turnovers[strat][bt_i] = compute_turnover(
                x_new, prev_positions[strat], sigma_t, sigma_t, vol_target
            )
            prev_positions[strat] = x_new.copy()

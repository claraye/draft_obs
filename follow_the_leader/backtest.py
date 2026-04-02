"""
backtest.py — Walk-forward backtest engine for Follow The Leader

Pipeline per strategy at each time t:
  1. Compute lead-lag matrices from rolling δ-day windows (ensemble)
  2. Learn sparse graph A via FTL convex QP
  3. Normalise: Ã = D^{-1/2}(A+I)D^{-1/2}
  4. Propagate oscillators: R̃^k = Ã · R^k  (NMM) or use R^k  (MACD baseline)
  5. Compute positions: X = φ(R̃) · σ_tgt / (σ^22 · √252) / M
  6. PnL: r^portfolio_{t+2} = X_t · r_{t+2}  (2-day execution lag)

Strategies:
  MACD    — Univariate MACD baseline (no graph)
  NMM     — Network Momentum using DEFAULT_METHOD lead-lag
  NMM-E   — NMM with ensemble over S lookback windows (DEFAULT_METHOD)
  [optional per-method variants when RUN_ALL_METHODS=True]
"""
import os
import pickle
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    returns:   dict   # {strategy: np.ndarray} of daily net returns
    turnovers: dict   # {strategy: np.ndarray} of daily turnovers
    positions: dict   # {strategy: np.ndarray(T, N)} of positions
    dates:     pd.DatetimeIndex
    assets:    list
    graphs:    dict = field(default_factory=dict)  # {date: A_tilde}
    meta:      dict = field(default_factory=dict)  # hyperparams etc.


def run_backtest(
    prices: pd.DataFrame,
    returns_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    oscillators: dict,
    assets: list,
    backtest_start: str,
    backtest_end: str,
    lookback_windows: list,
    alpha_opt: float,
    beta_opt: float,
    method: str = "dtw",
    shape_window: int = 5,
    sigma_tgt: float = 0.10,
    tdpy: int = 252,
    graph_refit_freq: int = 22,
    cost_bps: float = 3.0,
    solver_prefs: list = None,
    add_self_loops: bool = True,
    cache_dir: str = None,
    store_graphs: bool = False,
) -> BacktestResult:
    """
    Walk-forward backtest: computes daily returns for MACD and NMM strategies.

    Graph re-estimation schedule:
      - Graph Ã_t is recomputed every `graph_refit_freq` trading days.
      - Between refits, the previous Ã is reused.
      - This is equivalent to daily recomputation when graph_refit_freq=1.

    Args:
        prices:           DataFrame(T_full, N) of close prices
        returns_df:       DataFrame(T_full, N) of log returns
        vol_df:           DataFrame(T_full, N) of EWMA daily vol
        oscillators:      dict {k: DataFrame(T_full, N)} — pre-computed oscillators
        assets:           list of N asset names
        backtest_start:   OOS start date
        backtest_end:     OOS end date
        lookback_windows: list of δ values for ensemble (e.g. [22,44,66,88,110,132])
        alpha_opt:        graph learning α (from hyperparameter search)
        beta_opt:         graph learning β
        method:           lead-lag detection method
        shape_window:     SDTW/SDDTW descriptor window size
        sigma_tgt:        annualised target volatility
        tdpy:             trading days per year
        graph_refit_freq: days between graph re-estimations
        cost_bps:         transaction cost in bps (for net returns)
        solver_prefs:     CVXPY solver preference list
        add_self_loops:   add identity before graph normalisation
        cache_dir:        directory for graph caching (None = no caching)
        store_graphs:     whether to store Ã_t in BacktestResult.graphs

    Returns:
        BacktestResult with MACD, NMM-single, NMM-E strategies
    """
    from lead_lag import compute_ensemble_lead_lag
    from graph_learning import ensemble_graph, normalise_graph
    from signals import build_signal_matrix
    from portfolio import compute_positions, compute_turnover, apply_transaction_costs

    if solver_prefs is None:
        solver_prefs = ["CLARABEL", "SCS"]

    N = len(assets)
    K = len(oscillators)

    # ── OOS date range ──────────────────────────────────────────────────────
    start_dt = pd.Timestamp(backtest_start)
    end_dt   = pd.Timestamp(backtest_end)
    oos_idx  = returns_df.index[(returns_df.index >= start_dt) & (returns_df.index <= end_dt)]
    logger.info(f"OOS period: {oos_idx[0].date()} → {oos_idx[-1].date()} ({len(oos_idx)} days)")

    # ── Pre-extract arrays for speed ────────────────────────────────────────
    returns_arr = returns_df.values          # (T_full, N)
    vol_arr     = vol_df.values              # (T_full, N)
    # Oscillator arrays: {k: (T_full, N)}
    osc_arr = {k: oscillators[k].values for k in sorted(oscillators.keys())}
    k_values = sorted(osc_arr.keys())

    # ── Position buffers ────────────────────────────────────────────────────
    T_oos = len(oos_idx)
    strats = ["MACD", "NMM-E"]
    gross_rets   = {s: np.zeros(T_oos) for s in strats}
    turnover_buf = {s: np.zeros(T_oos) for s in strats}
    positions    = {s: np.zeros((T_oos, N)) for s in strats}

    prev_pos  = {s: np.zeros(N) for s in strats}
    A_tilde   = None      # current normalised graph
    last_refit_t = -999   # global index of last graph refit

    # ── Optional: graph cache ───────────────────────────────────────────────
    graph_cache = {}
    graph_cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        graph_cache_path = os.path.join(
            cache_dir, f"graphs_{method}_{alpha_opt}_{beta_opt}_{graph_refit_freq}.pkl"
        )
        if os.path.exists(graph_cache_path):
            logger.info(f"Loading graph cache: {graph_cache_path}")
            with open(graph_cache_path, "rb") as f:
                graph_cache = pickle.load(f)

    graphs_out = {}

    # ── Main loop ───────────────────────────────────────────────────────────
    for ti, date in enumerate(oos_idx):
        # Global index in returns_df
        gi = returns_df.index.get_loc(date)

        # ── 1. Refit graph if due ──────────────────────────────────────────
        refit_due = (gi - last_refit_t) >= graph_refit_freq

        if refit_due:
            if gi in graph_cache:
                A_tilde = graph_cache[gi]
            else:
                V_list = compute_ensemble_lead_lag(
                    returns_arr, gi,
                    lookback_windows=lookback_windows,
                    method=method,
                    shape_window=shape_window,
                )
                A_bar   = ensemble_graph(V_list, alpha_opt, beta_opt, solver_prefs)
                A_tilde = normalise_graph(A_bar, add_self_loops=add_self_loops)
                graph_cache[gi] = A_tilde
                last_refit_t = gi

            if store_graphs:
                graphs_out[date] = A_tilde

        if A_tilde is None:
            # Not enough data yet — skip
            for s in strats:
                positions[s][ti] = prev_pos[s]
            continue

        # ── 2. Oscillators at time t ────────────────────────────────────────
        osc_t = {k: osc_arr[k][gi] for k in k_values}

        # ── 3. Signal matrix per strategy ───────────────────────────────────
        S_macd = build_signal_matrix(osc_t, assets, use_network=False)
        S_nmm  = build_signal_matrix(osc_t, assets, use_network=True, A_tilde=A_tilde)

        # ── 4. Vol at time t ────────────────────────────────────────────────
        vol_t = vol_arr[gi]
        vol_t = np.where(vol_t > 1e-10, vol_t, np.nanmedian(vol_arr[:gi + 1], axis=0))

        # ── 5. Positions ─────────────────────────────────────────────────────
        pos_macd = compute_positions(S_macd, vol_t, sigma_tgt, tdpy)
        pos_nmm  = compute_positions(S_nmm,  vol_t, sigma_tgt, tdpy)

        positions["MACD"][ti]  = pos_macd
        positions["NMM-E"][ti] = pos_nmm

        # ── 6. PnL at t+2 (signal@t, trade@t+1, PnL@t+2) ───────────────────
        t2 = gi + 2
        if t2 < len(returns_arr):
            r_t2 = returns_arr[t2]
            gross_rets["MACD"][ti]  = float(np.nansum(pos_macd * r_t2))
            gross_rets["NMM-E"][ti] = float(np.nansum(pos_nmm  * r_t2))

        # ── 7. Turnover ───────────────────────────────────────────────────────
        turnover_buf["MACD"][ti]  = compute_turnover(pos_macd, prev_pos["MACD"])
        turnover_buf["NMM-E"][ti] = compute_turnover(pos_nmm,  prev_pos["NMM-E"])

        prev_pos["MACD"]  = pos_macd
        prev_pos["NMM-E"] = pos_nmm

    # ── Save graph cache ──────────────────────────────────────────────────
    if cache_dir is not None and graph_cache_path is not None:
        with open(graph_cache_path, "wb") as f:
            pickle.dump(graph_cache, f)

    # ── Apply transaction costs ───────────────────────────────────────────
    net_rets = {
        s: apply_transaction_costs(gross_rets[s], turnover_buf[s], cost_bps)
        for s in strats
    }

    return BacktestResult(
        returns=net_rets,
        turnovers=turnover_buf,
        positions=positions,
        dates=oos_idx,
        assets=assets,
        graphs=graphs_out,
        meta={"alpha": alpha_opt, "beta": beta_opt, "method": method,
              "cost_bps": cost_bps, "sigma_tgt": sigma_tgt},
    )


def run_hyperparam_search(
    returns_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    oscillators: dict,
    assets: list,
    backtest_start: str,
    lookback_windows: list,
    alpha_grid: list,
    beta_grid: list,
    method: str = "dtw",
    shape_window: int = 5,
    train_years: int = 3,
    sigma_tgt: float = 0.10,
    tdpy: int = 252,
    solver_prefs: list = None,
    add_self_loops: bool = True,
    fast_mode: bool = False,
) -> tuple:
    """
    Find optimal (alpha, beta) using in-sample Sharpe on first `train_years` of OOS data.

    In fast_mode, skips grid search and returns mid-range defaults.

    Returns:
        (alpha_opt, beta_opt)
    """
    from lead_lag import compute_ensemble_lead_lag
    from graph_learning import grid_search_hyperparams

    if fast_mode:
        alpha_opt = alpha_grid[len(alpha_grid) // 2]
        beta_opt  = beta_grid[len(beta_grid) // 2]
        logger.info(f"Fast mode: skipping grid search. "
                    f"Using alpha={alpha_opt}, beta={beta_opt}")
        return alpha_opt, beta_opt

    # Use training window: [backtest_start, backtest_start + train_years]
    start_dt = pd.Timestamp(backtest_start)
    end_train = start_dt + pd.DateOffset(years=train_years)

    mask = (returns_df.index >= start_dt) & (returns_df.index < end_train)
    returns_train = returns_df.loc[mask].values
    vol_train     = vol_df.loc[mask].values
    osc_train     = {k: oscillators[k].loc[mask].values for k in oscillators}

    T_train = len(returns_train)
    if T_train < 2 * max(lookback_windows):
        logger.warning(f"Training period too short ({T_train} days). "
                       f"Using default hyperparameters.")
        return alpha_grid[len(alpha_grid) // 2], beta_grid[len(beta_grid) // 2]

    # Compute lead-lag matrices at the midpoint of the training period
    mid = T_train // 2
    V_list = compute_ensemble_lead_lag(
        returns_train, mid,
        lookback_windows=lookback_windows,
        method=method, shape_window=shape_window,
    )

    alpha_opt, beta_opt = grid_search_hyperparams(
        V_list=V_list,
        returns_val=returns_train,
        oscillators_val=osc_train,
        vol_val=vol_train,
        alpha_grid=alpha_grid,
        beta_grid=beta_grid,
        sigma_tgt=sigma_tgt,
        tdpy=tdpy,
        solver_prefs=solver_prefs,
        add_self_loops=add_self_loops,
    )
    return alpha_opt, beta_opt


def make_backtest_fn(
    returns_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    oscillators: dict,
    assets: list,
    backtest_start: str,
    backtest_end: str,
    lookback_windows: list,
    alpha_opt: float,
    beta_opt: float,
    method: str,
    shape_window: int,
    sigma_tgt: float,
    tdpy: int,
    graph_refit_freq: int,
    cost_bps: float,
    solver_prefs: list,
    add_self_loops: bool,
) -> callable:
    """
    Create a backtest function suitable for bootstrap validation.
    The returned function takes (returns_boot, prices_boot) and re-computes
    all features from scratch on the bootstrapped data.
    """
    def _backtest(boot_returns: np.ndarray, boot_prices: np.ndarray) -> dict:
        # Recompute vol and oscillators on bootstrapped returns
        boot_ret_df = pd.DataFrame(boot_returns, columns=assets)
        boot_vol_df = boot_ret_df.ewm(span=22, min_periods=11).std()

        from features import build_oscillators
        boot_prices_df = pd.DataFrame(boot_prices[1:], columns=assets)
        boot_osc = build_oscillators(
            boot_ret_df, boot_vol_df,
            k_values=sorted(oscillators.keys()),
            M=4,
        )
        # Dummy date index (bootstrap scrambles dates)
        T_boot = len(boot_returns)
        fake_idx = pd.date_range(backtest_start, periods=T_boot, freq="B")
        for k in boot_osc:
            boot_osc[k].index = fake_idx
        boot_ret_df.index = fake_idx
        boot_vol_df.index = fake_idx

        result = run_backtest(
            prices=boot_prices_df,
            returns_df=boot_ret_df,
            vol_df=boot_vol_df,
            oscillators=boot_osc,
            assets=assets,
            backtest_start=str(fake_idx[max(lookback_windows)].date()),
            backtest_end=str(fake_idx[-1].date()),
            lookback_windows=lookback_windows,
            alpha_opt=alpha_opt,
            beta_opt=beta_opt,
            method=method,
            shape_window=shape_window,
            sigma_tgt=sigma_tgt,
            tdpy=tdpy,
            graph_refit_freq=graph_refit_freq,
            cost_bps=cost_bps,
            solver_prefs=solver_prefs,
            add_self_loops=add_self_loops,
            cache_dir=None,
        )
        return result.returns

    return _backtest

"""
Replication draft: Detecting Lead-Lag Relationships in Stock Returns and Portfolio Strategies
Cartea, Cucuringu, Jin

Implements the paper note's core workflow:
  1. Build a rolling pairwise lead-lag matrix with three scores:
       - C1: max directional cross-correlation
       - C2: average directional cross-correlation
       - levy: discrete Levy-area on cumulative standardized returns
  2. Rank stocks by lead-lag matrix column means.
  3. Trade a SPY-hedged zero-cost portfolio:
       - if previous mean leader return >= 0, long equal-weighted followers / short market
       - otherwise short followers / long market
  4. Compare data-driven strategies with simple market-cap and turnover leader/follower benchmarks.
  5. Run a frequency sweep and transaction-cost sweep.

Data contract for real-data runs
--------------------------------
returns CSV:
    wide format, first column is date, remaining columns are ticker returns in decimal units.
    Example columns: date,AAPL,MSFT,SPY

market CSV (optional):
    wide format, first column is date, one market-return column (e.g. SPY) in decimal units.

characteristics CSV (optional, long format):
    columns = date,ticker,market_cap,turnover
    Used only for Market Cap / Turnover benchmark rankings.

If no CSV paths are supplied, the script runs a synthetic demo with a planted nonlinear
lead-lag structure so the end-to-end backtest is executable immediately.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestConfig:
    lookback: int = 60
    leader_frac: float = 0.2
    follower_frac: float = 0.2
    max_lag: int = 5
    corr_score_mode: str = "absolute_strength"
    trade_lag: int = 1
    rebalance_every: int = 1
    transaction_cost_bps: float = 0.0
    min_valid_obs: int = 40


def generate_synthetic_demo(
    n_days: int = 1800,
    n_assets: int = 80,
    n_leaders: int = 12,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create a runnable toy market with both linear and nonlinear lagged follower responses."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_days)
    tickers = [f"STK{i:03d}" for i in range(n_assets)]

    market = rng.normal(0.0002, 0.009, size=n_days)
    latent = rng.normal(0.0, 0.012, size=(n_days, n_leaders))
    returns = rng.normal(0.0, 0.014, size=(n_days, n_assets))

    # Leaders react contemporaneously to latent factors.
    returns[:, :n_leaders] += 0.6 * latent + 0.35 * market[:, None]

    # Followers react with one-day delay and a mild nonlinear term.
    for follower in range(n_leaders, n_assets):
        leader = follower % n_leaders
        lagged = np.roll(latent[:, leader], 1)
        lagged[0] = 0.0
        returns[:, follower] += 0.22 * lagged + 0.12 * np.sign(lagged) * (lagged**2) / 0.012
        returns[:, follower] += 0.25 * market

    returns_df = pd.DataFrame(returns, index=dates, columns=tickers)
    market_ret = pd.Series(market, index=dates, name="SPY")

    rows = []
    base_caps = np.linspace(8e9, 5e10, n_assets)
    base_turnover = np.linspace(0.015, 0.06, n_assets)[::-1]
    for d_idx, dt in enumerate(dates):
        cap_noise = 1.0 + 0.05 * np.sin(d_idx / 50.0) + rng.normal(0.0, 0.01, n_assets)
        tvr_noise = 1.0 + 0.10 * np.cos(d_idx / 40.0) + rng.normal(0.0, 0.03, n_assets)
        for i, ticker in enumerate(tickers):
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "market_cap": base_caps[i] * cap_noise[i],
                    "turnover": max(base_turnover[i] * tvr_noise[i], 1e-5),
                }
            )
    characteristics_df = pd.DataFrame(rows)
    return returns_df, market_ret, characteristics_df


def read_wide_returns_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def read_market_csv(path: Path) -> pd.Series:
    df = read_wide_returns_csv(path)
    if df.shape[1] != 1:
        raise ValueError("market CSV must contain exactly one return column besides date")
    return df.iloc[:, 0].rename(df.columns[0])


def read_characteristics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "ticker", "market_cap", "turnover"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"characteristics CSV is missing columns: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    return df.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"])


def standardize_window(window_df: pd.DataFrame) -> pd.DataFrame:
    mu = window_df.mean(axis=0, skipna=True)
    sigma = window_df.std(axis=0, ddof=0, skipna=True).replace(0.0, np.nan)
    return (window_df - mu) / sigma


def _corr_at_positive_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(x):
        return np.nan
    left = x[lag:]
    right = y[:-lag]
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return np.nan
    left = left[mask]
    right = right[mask]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _best_signed_by_abs(values: list[float]) -> float:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0.0
    return float(finite[np.argmax(np.abs(finite))])


def c1_components(row_asset: np.ndarray, col_asset: np.ndarray, max_lag: int) -> dict[str, float]:
    """
    Separate direction-of-leadership from sign-of-predictive-relation.

    Returns:
      strength: positive if col leads row, negative if row leads col
      forward_relation_sign: sign of the strongest col->row lagged correlation
      reverse_relation_sign: sign of the strongest row->col lagged correlation
    """
    forward = [
        _corr_at_positive_lag(row_asset, col_asset, lag)
        for lag in range(1, max_lag + 1)
    ]
    reverse = [
        _corr_at_positive_lag(col_asset, row_asset, lag)
        for lag in range(1, max_lag + 1)
    ]
    forward_best = _best_signed_by_abs(forward)
    reverse_best = _best_signed_by_abs(reverse)
    return {
        "strength": float(abs(forward_best) - abs(reverse_best)),
        "forward_relation_sign": float(np.sign(forward_best)),
        "reverse_relation_sign": float(np.sign(reverse_best)),
        "forward_best_corr": float(forward_best),
        "reverse_best_corr": float(reverse_best),
    }


def c1_score(
    row_asset: np.ndarray,
    col_asset: np.ndarray,
    max_lag: int,
    mode: str = "absolute_strength",
) -> float:
    """
    Positive score means the column asset leads the row asset.

    This scalar is now a pure leadership-strength score based on absolute
    lagged dependence. Momentum-vs-reversal sign is preserved separately in
    `c1_components`, rather than being mixed into the direction score.
    """
    if mode == "absolute_strength":
        return c1_components(row_asset, col_asset, max_lag=max_lag)["strength"]
    if mode == "signed_correlation":
        forward = [
            _corr_at_positive_lag(row_asset, col_asset, lag)
            for lag in range(1, max_lag + 1)
        ]
        reverse = [
            _corr_at_positive_lag(col_asset, row_asset, lag)
            for lag in range(1, max_lag + 1)
        ]
        forward_best = np.nanmax(forward) if np.isfinite(forward).any() else 0.0
        reverse_best = np.nanmax(reverse) if np.isfinite(reverse).any() else 0.0
        return float(forward_best - reverse_best)
    raise ValueError(f"Unknown corr_score_mode for c1: {mode}")


def c2_components(row_asset: np.ndarray, col_asset: np.ndarray, max_lag: int) -> dict[str, float]:
    """
    Average-directional version of the same separation.

    Returns:
      strength: positive if col leads row, negative if row leads col
      forward_relation_sign: sign of the average col->row lagged correlation
      reverse_relation_sign: sign of the average row->col lagged correlation
    """
    forward = [
        _corr_at_positive_lag(row_asset, col_asset, lag)
        for lag in range(1, max_lag + 1)
    ]
    reverse = [
        _corr_at_positive_lag(col_asset, row_asset, lag)
        for lag in range(1, max_lag + 1)
    ]
    forward_finite = np.asarray([v for v in forward if np.isfinite(v)], dtype=float)
    reverse_finite = np.asarray([v for v in reverse if np.isfinite(v)], dtype=float)

    forward_mean = float(np.mean(forward_finite)) if forward_finite.size else 0.0
    reverse_mean = float(np.mean(reverse_finite)) if reverse_finite.size else 0.0
    forward_strength = float(np.mean(np.abs(forward_finite))) if forward_finite.size else 0.0
    reverse_strength = float(np.mean(np.abs(reverse_finite))) if reverse_finite.size else 0.0
    return {
        "strength": float(forward_strength - reverse_strength),
        "forward_relation_sign": float(np.sign(forward_mean)),
        "reverse_relation_sign": float(np.sign(reverse_mean)),
        "forward_mean_corr": float(forward_mean),
        "reverse_mean_corr": float(reverse_mean),
    }


def c2_score(
    row_asset: np.ndarray,
    col_asset: np.ndarray,
    max_lag: int,
    mode: str = "absolute_strength",
) -> float:
    """
    Positive score means the column asset leads the row asset.

    This scalar uses average absolute lagged dependence for direction, while
    `c2_components` preserves whether the average relation is momentum-like or
    reversal-like.
    """
    if mode == "absolute_strength":
        return c2_components(row_asset, col_asset, max_lag=max_lag)["strength"]
    if mode == "signed_correlation":
        forward = [
            _corr_at_positive_lag(row_asset, col_asset, lag)
            for lag in range(1, max_lag + 1)
        ]
        reverse = [
            _corr_at_positive_lag(col_asset, row_asset, lag)
            for lag in range(1, max_lag + 1)
        ]
        forward_mean = np.nanmean(forward) if np.isfinite(forward).any() else 0.0
        reverse_mean = np.nanmean(reverse) if np.isfinite(reverse).any() else 0.0
        return float(forward_mean - reverse_mean)
    raise ValueError(f"Unknown corr_score_mode for c2: {mode}")


def levy_area_score(row_asset: np.ndarray, col_asset: np.ndarray) -> float:
    """
    Discrete Levy-area proxy on cumulative standardized-return paths.

    The sign convention is flipped so a positive score means the column asset leads
    the row asset, consistent with ranking by column means.
    """
    mask = np.isfinite(row_asset) & np.isfinite(col_asset)
    x = row_asset[mask]
    y = col_asset[mask]
    if len(x) < 3:
        return 0.0

    x_path = np.cumsum(x)
    y_path = np.cumsum(y)
    dx = np.diff(x_path, prepend=x_path[0])
    dy = np.diff(y_path, prepend=y_path[0])
    area = 0.5 * np.sum(x_path * dy - y_path * dx)
    return float(-area / len(x))


def follower_relation_signals(
    window_df: pd.DataFrame,
    leaders: list[str],
    followers: list[str],
    method: str,
    max_lag: int,
    corr_score_mode: str,
) -> dict[str, float]:
    """
    Estimate the momentum-vs-reversal sign from leaders to each follower.

    Returns a per-follower sign in {-1, +1}. For methods that do not support a
    clean sign separation (e.g. levy, or signed_correlation mode), this falls
    back to +1 for every follower so the original continuation-style trading
    logic is preserved.
    """
    if method not in {"c1", "c2"} or corr_score_mode != "absolute_strength":
        return {follower: 1.0 for follower in followers}

    signs: dict[str, float] = {}
    for follower in followers:
        relation_values = []
        row_asset = window_df[follower].to_numpy(dtype=float)
        for leader in leaders:
            col_asset = window_df[leader].to_numpy(dtype=float)
            if method == "c1":
                relation = c1_components(row_asset, col_asset, max_lag=max_lag)[
                    "forward_relation_sign"
                ]
            else:
                relation = c2_components(row_asset, col_asset, max_lag=max_lag)[
                    "forward_relation_sign"
                ]
            if np.isfinite(relation) and relation != 0:
                relation_values.append(float(relation))

        if relation_values:
            signs[follower] = float(np.sign(np.mean(relation_values)))
            if signs[follower] == 0:
                signs[follower] = 1.0
        else:
            signs[follower] = 1.0
    return signs


def build_lead_lag_matrix(
    window_df: pd.DataFrame,
    method: str,
    max_lag: int,
    min_valid_obs: int,
    corr_score_mode: str = "absolute_strength",
) -> pd.DataFrame:
    z = standardize_window(window_df)
    cols = z.columns.tolist()
    n_assets = len(cols)
    values = z.to_numpy(dtype=float)
    matrix = np.zeros((n_assets, n_assets), dtype=float)

    valid_counts = np.isfinite(values).sum(axis=0)
    active = valid_counts >= min_valid_obs

    for i in range(n_assets):
        if not active[i]:
            continue
        row_asset = values[:, i]
        for j in range(i + 1, n_assets):
            if not active[j]:
                continue
            col_asset = values[:, j]
            if method == "c1":
                score = c1_score(
                    row_asset,
                    col_asset,
                    max_lag=max_lag,
                    mode=corr_score_mode,
                )
            elif method == "c2":
                score = c2_score(
                    row_asset,
                    col_asset,
                    max_lag=max_lag,
                    mode=corr_score_mode,
                )
            elif method == "levy":
                score = levy_area_score(row_asset, col_asset)
            else:
                raise ValueError(f"Unknown method: {method}")
            matrix[i, j] = score
            matrix[j, i] = -score

    return pd.DataFrame(matrix, index=cols, columns=cols)


def pick_leaders_followers(
    score_matrix: pd.DataFrame,
    leader_frac: float,
    follower_frac: float,
) -> tuple[list[str], list[str], pd.Series]:
    column_means = score_matrix.mean(axis=0).sort_values(ascending=False)
    n_assets = len(column_means)
    n_leaders = max(1, int(np.floor(n_assets * leader_frac)))
    n_followers = max(1, int(np.floor(n_assets * follower_frac)))
    if n_leaders + n_followers > n_assets:
        raise ValueError("leader_frac + follower_frac must be <= 1.0")
    leaders = column_means.index[:n_leaders].tolist()
    followers = column_means.index[-n_followers:].tolist()
    return leaders, followers, column_means


def rank_by_characteristic(
    chars_df: pd.DataFrame,
    date: pd.Timestamp,
    universe: list[str],
    characteristic: str,
    leader_frac: float,
    follower_frac: float,
) -> tuple[list[str], list[str]]:
    day = chars_df.loc[
        (chars_df["date"] == date) & (chars_df["ticker"].isin(universe)),
        ["ticker", characteristic],
    ].dropna()
    if day.empty:
        return [], []
    scores = day.set_index("ticker")[characteristic].sort_values(ascending=False)
    n_assets = len(scores)
    n_leaders = max(1, int(np.floor(n_assets * leader_frac)))
    n_followers = max(1, int(np.floor(n_assets * follower_frac)))
    if n_leaders + n_followers > n_assets:
        return [], []
    return scores.index[:n_leaders].tolist(), scores.index[-n_followers:].tolist()


def weights_from_signal(
    followers: list[str],
    market_name: str,
    signal_sign: float,
    follower_signs: dict[str, float] | None = None,
) -> pd.Series:
    if not followers:
        return pd.Series(dtype=float)
    if follower_signs is None:
        follower_signs = {ticker: 1.0 for ticker in followers}
    follower_weight = 1.0 / len(followers)
    weights = {
        ticker: signal_sign * follower_signs.get(ticker, 1.0) * follower_weight
        for ticker in followers
    }
    weights[market_name] = -signal_sign
    return pd.Series(weights, dtype=float)


def run_strategy(
    returns_df: pd.DataFrame,
    market_ret: pd.Series,
    config: BacktestConfig,
    method: str,
    chars_df: pd.DataFrame | None = None,
    benchmark_characteristic: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    dates = returns_df.index.intersection(market_ret.index)
    returns_df = returns_df.loc[dates].copy()
    market_ret = market_ret.loc[dates].copy()
    market_name = market_ret.name or "SPY"

    portfolio_returns: list[tuple[pd.Timestamp, float]] = []
    diagnostics: list[dict] = []
    prev_weights = pd.Series(dtype=float)
    prev_followers: list[str] = []
    prev_leaders: list[str] = []
    prev_follower_signs: dict[str, float] = {}

    first_signal_idx = config.lookback - 1
    last_signal_idx = len(dates) - 2 - config.trade_lag

    for t_idx in range(first_signal_idx, last_signal_idx + 1):
        signal_date = dates[t_idx]
        return_date = dates[t_idx + 1 + config.trade_lag]
        do_rebalance = ((t_idx - first_signal_idx) % config.rebalance_every) == 0

        if do_rebalance:
            window_df = returns_df.iloc[t_idx - config.lookback + 1 : t_idx + 1]
            universe = window_df.columns[window_df.notna().sum(axis=0) >= config.min_valid_obs].tolist()

            if benchmark_characteristic is None:
                score_matrix = build_lead_lag_matrix(
                    window_df[universe],
                    method=method,
                    max_lag=config.max_lag,
                    min_valid_obs=config.min_valid_obs,
                    corr_score_mode=config.corr_score_mode,
                )
                prev_leaders, prev_followers, rank_scores = pick_leaders_followers(
                    score_matrix,
                    leader_frac=config.leader_frac,
                    follower_frac=config.follower_frac,
                )
                prev_follower_signs = follower_relation_signals(
                    window_df=window_df[universe],
                    leaders=prev_leaders,
                    followers=prev_followers,
                    method=method,
                    max_lag=config.max_lag,
                    corr_score_mode=config.corr_score_mode,
                )
                leader_score = float(rank_scores.loc[prev_leaders].mean()) if prev_leaders else np.nan
                follower_score = float(rank_scores.loc[prev_followers].mean()) if prev_followers else np.nan
            else:
                if chars_df is None:
                    raise ValueError("chars_df is required for characteristic benchmarks")
                prev_leaders, prev_followers = rank_by_characteristic(
                    chars_df=chars_df,
                    date=signal_date,
                    universe=universe,
                    characteristic=benchmark_characteristic,
                    leader_frac=config.leader_frac,
                    follower_frac=config.follower_frac,
                )
                leader_score = np.nan
                follower_score = np.nan
                prev_follower_signs = {follower: 1.0 for follower in prev_followers}

            leader_ret_prev = returns_df.loc[signal_date, prev_leaders].mean()
            signal_sign = 1.0 if pd.notna(leader_ret_prev) and leader_ret_prev >= 0.0 else -1.0
            current_weights = weights_from_signal(
                followers=prev_followers,
                market_name=market_name,
                signal_sign=signal_sign,
                follower_signs=prev_follower_signs,
            )
        else:
            current_weights = prev_weights.copy()
            leader_score = np.nan
            follower_score = np.nan
            leader_ret_prev = returns_df.loc[signal_date, prev_leaders].mean() if prev_leaders else np.nan

        asset_ret = returns_df.loc[return_date, current_weights.index.intersection(returns_df.columns)]
        pnl = float((current_weights.loc[asset_ret.index] * asset_ret.fillna(0.0)).sum())
        if market_name in current_weights.index:
            pnl += float(current_weights[market_name] * market_ret.loc[return_date])

        all_names = prev_weights.index.union(current_weights.index)
        turnover = (
            current_weights.reindex(all_names, fill_value=0.0)
            - prev_weights.reindex(all_names, fill_value=0.0)
        ).abs().sum()
        pnl -= config.transaction_cost_bps * 1e-4 * float(turnover)

        portfolio_returns.append((return_date, pnl))
        diagnostics.append(
            {
                "signal_date": signal_date,
                "return_date": return_date,
                "n_leaders": len(prev_leaders),
                "n_followers": len(prev_followers),
                "leader_return_prev": leader_ret_prev,
                "leader_score_mean": leader_score,
                "follower_score_mean": follower_score,
                "avg_follower_relation_sign": (
                    float(np.mean(list(prev_follower_signs.values())))
                    if prev_follower_signs
                    else np.nan
                ),
                "turnover": float(turnover),
                "rebalance": do_rebalance,
            }
        )
        prev_weights = current_weights

    ret_s = pd.Series(dict(portfolio_returns)).sort_index()
    ret_s.name = method if benchmark_characteristic is None else benchmark_characteristic
    diag_df = pd.DataFrame(diagnostics).set_index("return_date")
    return ret_s, diag_df


def performance_summary(ret_s: pd.Series) -> dict[str, float]:
    if ret_s.empty:
        return {
            "compound_return_pct": np.nan,
            "return_bps_day": np.nan,
            "daily_vol_pct": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
        }
    equity = (1.0 + ret_s.fillna(0.0)).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    n_days = len(ret_s)
    ann_return = equity.iloc[-1] ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0
    daily_vol = ret_s.std(ddof=0)
    sharpe = (
        np.sqrt(TRADING_DAYS_PER_YEAR) * ret_s.mean() / daily_vol
        if daily_vol > 0
        else np.nan
    )
    return {
        "compound_return_pct": 100.0 * ann_return,
        "return_bps_day": 1e4 * ret_s.mean(),
        "daily_vol_pct": 100.0 * daily_vol,
        "sharpe": float(sharpe),
        "max_drawdown_pct": 100.0 * float(drawdown.min()),
    }


def run_frequency_sweep(
    returns_df: pd.DataFrame,
    market_ret: pd.Series,
    base_config: BacktestConfig,
    method: str = "levy",
) -> pd.DataFrame:
    rows = []
    for step in [1, 2, 5, 10, 15, 21]:
        cfg = BacktestConfig(
            lookback=base_config.lookback,
            leader_frac=base_config.leader_frac,
            follower_frac=base_config.follower_frac,
            max_lag=base_config.max_lag,
            corr_score_mode=base_config.corr_score_mode,
            trade_lag=base_config.trade_lag,
            rebalance_every=step,
            transaction_cost_bps=base_config.transaction_cost_bps,
            min_valid_obs=base_config.min_valid_obs,
        )
        ret_s, _ = run_strategy(returns_df, market_ret, cfg, method=method)
        row = {"rebalance_every_days": step, **performance_summary(ret_s)}
        rows.append(row)
    return pd.DataFrame(rows)


def run_cost_sweep(
    returns_df: pd.DataFrame,
    market_ret: pd.Series,
    base_config: BacktestConfig,
    method: str = "levy",
) -> pd.DataFrame:
    rows = []
    for cost_bps in [0.0, 1.0, 3.0, 4.0, 7.0, 10.0]:
        cfg = BacktestConfig(
            lookback=base_config.lookback,
            leader_frac=base_config.leader_frac,
            follower_frac=base_config.follower_frac,
            max_lag=base_config.max_lag,
            corr_score_mode=base_config.corr_score_mode,
            trade_lag=base_config.trade_lag,
            rebalance_every=base_config.rebalance_every,
            transaction_cost_bps=cost_bps,
            min_valid_obs=base_config.min_valid_obs,
        )
        ret_s, _ = run_strategy(returns_df, market_ret, cfg, method=method)
        row = {"transaction_cost_bps": cost_bps, **performance_summary(ret_s)}
        rows.append(row)
    return pd.DataFrame(rows)


def run_all_backtests(
    returns_df: pd.DataFrame,
    market_ret: pd.Series,
    chars_df: pd.DataFrame | None,
    config: BacktestConfig,
    output_dir: Path,
    run_sweeps: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    diagnostics = {}
    for method in ["c1", "c2", "levy"]:
        ret_s, diag_df = run_strategy(
            returns_df=returns_df,
            market_ret=market_ret,
            config=config,
            method=method,
        )
        results[method] = ret_s
        diagnostics[method] = diag_df
        diag_df.to_csv(output_dir / f"diagnostics_{method}.csv")

    if chars_df is not None:
        for bench in ["market_cap", "turnover"]:
            ret_s, diag_df = run_strategy(
                returns_df=returns_df,
                market_ret=market_ret,
                config=config,
                method="levy",
                chars_df=chars_df,
                benchmark_characteristic=bench,
            )
            results[bench] = ret_s
            diagnostics[bench] = diag_df
            diag_df.to_csv(output_dir / f"diagnostics_{bench}.csv")

    returns_panel = pd.DataFrame(results).sort_index()
    perf_table = pd.DataFrame(
        {name: performance_summary(ret_s) for name, ret_s in results.items()}
    ).T

    returns_panel.to_csv(output_dir / "strategy_returns.csv")
    perf_table.to_csv(output_dir / "performance_table.csv")
    if run_sweeps:
        run_frequency_sweep(returns_df, market_ret, config, method="levy").to_csv(
            output_dir / "frequency_sweep_levy.csv",
            index=False,
        )
        run_cost_sweep(returns_df, market_ret, config, method="levy").to_csv(
            output_dir / "cost_sweep_levy.csv",
            index=False,
        )

    print("\nPerformance summary")
    print(perf_table.round(4).to_string())
    print(f"\nSaved outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate Cartea-Cucuringu-Jin lead-lag portfolios from returns data."
    )
    parser.add_argument("--returns-csv", type=Path, default=None)
    parser.add_argument("--market-csv", type=Path, default=None)
    parser.add_argument("--characteristics-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("detecting_lead_lag_output"))
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--leader-frac", type=float, default=0.2)
    parser.add_argument("--follower-frac", type=float, default=0.2)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument(
        "--corr-score-mode",
        type=str,
        default="absolute_strength",
        choices=["absolute_strength", "signed_correlation"],
        help="For C1/C2 only: use original signed correlation scoring or separated absolute-strength leadership scoring.",
    )
    parser.add_argument("--trade-lag", type=int, default=1)
    parser.add_argument("--rebalance-every", type=int, default=1)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--min-valid-obs", type=int, default=40)
    parser.add_argument(
        "--run-sweeps",
        action="store_true",
        help="Also run the frequency and transaction-cost sweeps. Slower, but closer to the paper tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BacktestConfig(
        lookback=args.lookback,
        leader_frac=args.leader_frac,
        follower_frac=args.follower_frac,
        max_lag=args.max_lag,
        corr_score_mode=args.corr_score_mode,
        trade_lag=args.trade_lag,
        rebalance_every=args.rebalance_every,
        transaction_cost_bps=args.transaction_cost_bps,
        min_valid_obs=args.min_valid_obs,
    )

    if args.returns_csv is None:
        print("No CSV inputs supplied; running synthetic demo.")
        returns_df, market_ret, chars_df = generate_synthetic_demo()
    else:
        returns_df = read_wide_returns_csv(args.returns_csv)
        if args.market_csv is None:
            if "SPY" not in returns_df.columns:
                raise ValueError("Provide --market-csv or include an SPY column in --returns-csv")
            market_ret = returns_df.pop("SPY").rename("SPY")
        else:
            market_ret = read_market_csv(args.market_csv)
        chars_df = (
            read_characteristics_csv(args.characteristics_csv)
            if args.characteristics_csv is not None
            else None
        )

    run_all_backtests(
        returns_df=returns_df,
        market_ret=market_ret,
        chars_df=chars_df,
        config=config,
        output_dir=args.output_dir,
        run_sweeps=args.run_sweeps,
    )


if __name__ == "__main__":
    main()

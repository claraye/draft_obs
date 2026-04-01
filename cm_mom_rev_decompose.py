"""
Backtest: Momentum and Reversal on the Short-Term Horizon
Evidence from Commodity Markets — Ding, Kang, Yu, Zhao (2026)

Replicates the core empirical framework from the paper:
  1. Return decomposition via weekly cross-sectional (Fama-MacBeth style) regression
       R_{i,t} = b0_t + b1_t * Q_{i,t} + eps_{i,t}
       Q_{i,t}     = b1_t * Q_{i,t}            (flow component)
       R_nonQ_{i,t} = R_{i,t} - Q_fitted       (orthogonal residual)

  2. Short-term momentum (long high R_nonQ) and short-term reversal (long low Q)

  3. Improved intermediate-term momentum: sort on cumulative R_nonQ
     vs. raw R over 13/26/39/52-week windows

  4. Predictive Fama-MacBeth regressions with Newey-West SEs

  5. Variance channel: momentum stronger when volatility is low
  6. Crowding channel: momentum stronger when positioning is not crowded

─────────────────────────────────────────────────────────────────────────────
DATA REQUIREMENTS (all weekly, Tuesday-to-Tuesday, same dates × tickers grid)
─────────────────────────────────────────────────────────────────────────────
returns_df  : pd.DataFrame  — weekly futures returns (front-month)
              index=DatetimeIndex, columns=ticker strings, values=float
cot_df      : pd.DataFrame  — speculators' net position CHANGE scaled by OI
              = (NetLong_t - NetLong_{t-1}) / OI_{t-1}
              (CFTC COT non-commercial net, large-spec, or TFF managed money)

Optional control DataFrames (same index / columns):
basis_df    : annualised basis (ln F1 - ln F2) / (T2-T1)
basismom_df : 52-wk basis momentum (Boons & Porras Prado 2019)
shp_df      : smoothed hedging pressure = 52-wk MA of commercial short-long / OI

If these are not provided, controls are omitted from Fama-MacBeth regressions.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 0.  SAMPLE DATA GENERATOR (replace with real data loading)
# ══════════════════════════════════════════════════════════════════════════════

def generate_sample_data(
    n_weeks: int = 1_664,   # ~32 years at weekly frequency (1993-2025)
    n_assets: int = 26,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Synthetic weekly data with structure that loosely mirrors the paper:
      - Q positively correlated with R (contemporaneous), b1 ≈ 0.4
      - R_nonQ has mild positive autocorrelation (momentum in residual)
      - Q has mild mean-reversion in its price impact (reversal)
    Returns: returns_df, cot_df, basis_df, basismom_df, shp_df
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1993-01-05", periods=n_weeks, freq="W-TUE")
    tickers = [f"CM{i:02d}" for i in range(n_assets)]

    # True orthogonal component with momentum: mild AR(1)
    r_nonq = np.zeros((n_weeks, n_assets))
    r_nonq[0] = rng.normal(0, 0.015, n_assets)
    for t in range(1, n_weeks):
        r_nonq[t] = 0.12 * r_nonq[t - 1] + rng.normal(0, 0.015, n_assets)

    # True Q: speculator flow (drives reversal next week via -0.4 loading)
    Q_true = rng.normal(0, 0.012, (n_weeks, n_assets))

    # Observed return = 0.4 * Q + R_nonQ  (paper: b1_avg = 0.41)
    ret = 0.40 * Q_true + r_nonq + rng.normal(0, 0.003, (n_weeks, n_assets))

    returns_df  = pd.DataFrame(ret,    index=dates, columns=tickers)
    cot_df      = pd.DataFrame(Q_true, index=dates, columns=tickers)

    # Basis: mild mean-reversion around zero
    basis = rng.normal(0, 0.02, (n_weeks, n_assets))
    basis_df = pd.DataFrame(basis, index=dates, columns=tickers)

    # Basis momentum: 52-wk difference in basis
    basismom_df = basis_df.rolling(52).mean().diff(4)

    # Smoothed hedging pressure
    hp_raw = rng.normal(0, 0.05, (n_weeks, n_assets))
    shp_df = pd.DataFrame(hp_raw, index=dates, columns=tickers) \
               .rolling(52).mean()

    return returns_df, cot_df, basis_df, basismom_df, shp_df


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CROSS-SECTIONAL DECOMPOSITION  (Fama-MacBeth style, one OLS per week)
# ══════════════════════════════════════════════════════════════════════════════

def decompose_returns(
    returns_df: pd.DataFrame,
    cot_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    For each week t, run a cross-sectional OLS:
        R_{i,t} = b0_t + b1_t * Q_{i,t} + eps_{i,t}

    Q_fitted_{i,t}  = b1_t * Q_{i,t}            (flow component)
    R_nonQ_{i,t}    = R_{i,t} - Q_fitted_{i,t}  (orthogonal component)

    Returns
    -------
    Q_fitted_df : DataFrame  — flow-driven return component
    R_nonQ_df   : DataFrame  — orthogonal return component
    betas       : Series     — time-series of weekly slope estimates b1_t
    """
    assert returns_df.shape == cot_df.shape
    assert (returns_df.index == cot_df.index).all()

    Q_fitted_rows, R_nonQ_rows, beta_rows = [], [], []

    for t in returns_df.index:
        R_t = returns_df.loc[t]
        Q_t = cot_df.loc[t]
        mask = R_t.notna() & Q_t.notna()
        n = mask.sum()

        if n < 5:
            Q_fitted_rows.append(pd.Series(np.nan, index=returns_df.columns, name=t))
            R_nonQ_rows.append(pd.Series(np.nan, index=returns_df.columns, name=t))
            beta_rows.append((t, np.nan))
            continue

        b1, b0, _, _, _ = stats.linregress(Q_t[mask], R_t[mask])
        q_fit = b1 * Q_t
        r_nonq = R_t - q_fit

        Q_fitted_rows.append(q_fit.rename(t))
        R_nonQ_rows.append(r_nonq.rename(t))
        beta_rows.append((t, b1))

    Q_fitted_df = pd.DataFrame(Q_fitted_rows)
    R_nonQ_df   = pd.DataFrame(R_nonQ_rows)
    betas       = pd.Series(
        {t: b for t, b in beta_rows}, name="b1_Q_on_R"
    )

    return Q_fitted_df, R_nonQ_df, betas


# ══════════════════════════════════════════════════════════════════════════════
# 1b. TIME-SERIES DECOMPOSITION  (rolling OLS per asset)
# ══════════════════════════════════════════════════════════════════════════════

def decompose_returns_ts(
    returns_df: pd.DataFrame,
    cot_df: pd.DataFrame,
    window: int = 52,
    min_periods: int = 26,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each asset i, run a rolling time-series OLS over a lookback window:
        R_{i,t} = b0_{i,t} + b1_{i,t} * Q_{i,t} + eps_{i,t}

    Q_fitted_{i,t}  = b1_{i,t} * Q_{i,t}            (flow component)
    R_nonQ_{i,t}    = R_{i,t} - Q_fitted_{i,t}       (orthogonal component)

    Unlike the cross-sectional version (which pools assets at each t), this
    version estimates a separate beta per asset using its own return history,
    allowing each commodity to have a distinct Q sensitivity.

    Parameters
    ----------
    returns_df  : (T x N) weekly returns
    cot_df      : (T x N) weekly Q (net speculator flow / OI), aligned
    window      : rolling window length in weeks (default 52)
    min_periods : minimum observations required to produce an estimate

    Returns
    -------
    Q_fitted_df : DataFrame  — flow-driven return component
    R_nonQ_df   : DataFrame  — orthogonal return component
    betas_df    : DataFrame  — rolling b1_{i,t} for each asset
    """
    assert returns_df.shape == cot_df.shape
    assert (returns_df.index == cot_df.index).all()
    assert (returns_df.columns == cot_df.columns).all()

    Q_fitted_df = pd.DataFrame(np.nan, index=returns_df.index, columns=returns_df.columns)
    R_nonQ_df   = pd.DataFrame(np.nan, index=returns_df.index, columns=returns_df.columns)
    betas_df    = pd.DataFrame(np.nan, index=returns_df.index, columns=returns_df.columns)

    for i, asset in enumerate(returns_df.columns):
        R_i = returns_df[asset]
        Q_i = cot_df[asset]

        for t_idx in range(len(returns_df)):
            # rolling window ending at t_idx (inclusive)
            start = max(0, t_idx - window + 1)
            R_win = R_i.iloc[start : t_idx + 1]
            Q_win = Q_i.iloc[start : t_idx + 1]

            mask = R_win.notna() & Q_win.notna()
            if mask.sum() < min_periods:
                continue

            b1, b0, _, _, _ = stats.linregress(Q_win[mask], R_win[mask])
            t = returns_df.index[t_idx]

            q_fit = b1 * Q_i.loc[t]
            Q_fitted_df.loc[t, asset] = q_fit
            R_nonQ_df.loc[t, asset]   = R_i.loc[t] - q_fit
            betas_df.loc[t, asset]    = b1

    return Q_fitted_df, R_nonQ_df, betas_df


# ══════════════════════════════════════════════════════════════════════════════
# 1c. ROLLING PANEL DECOMPOSITION  (pooled OLS across assets × weeks in window)
# ══════════════════════════════════════════════════════════════════════════════

def decompose_returns_panel(
    returns_df: pd.DataFrame,
    cot_df: pd.DataFrame,
    window: int = 52,
    min_obs: int = 100,
    fixed_effects: str = "none",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    For each week t, stack all (asset, week) observations within a rolling
    lookback window and run a single pooled OLS:

        R_{i,s} = b0 + b1_t * Q_{i,s} + [FE] + eps_{i,s}
                  for all i, s in [t-window+1, t]

    Q_fitted_{i,t}  = b1_t * Q_{i,t}            (flow component, current week)
    R_nonQ_{i,t}    = R_{i,t} - Q_fitted_{i,t}  (orthogonal component)

    A single beta b1_t is estimated from the full panel in the window and then
    applied to the current week's Q to produce the decomposition. This pools
    information across both the time and cross-sectional dimensions, giving a
    more stable beta estimate than either the pure cross-sectional or pure
    time-series versions.

    Parameters
    ----------
    returns_df    : (T x N) weekly returns
    cot_df        : (T x N) weekly Q (net speculator flow / OI), aligned
    window        : rolling window in weeks (default 52)
    min_obs       : minimum pooled observations required (default 100)
    fixed_effects : 'none'  — pooled OLS, no FE
                    'asset' — demean each asset's R and Q within the window
                              (within-estimator; removes asset-level intercepts)
                    'time'  — demean each week's R and Q within the window
                              (absorbs common weekly shocks)

    Returns
    -------
    Q_fitted_df : DataFrame  — flow-driven return component
    R_nonQ_df   : DataFrame  — orthogonal return component
    betas       : Series     — rolling pooled b1_t for each week t
    """
    assert returns_df.shape == cot_df.shape
    assert (returns_df.index == cot_df.index).all()
    assert (returns_df.columns == cot_df.columns).all()
    assert fixed_effects in ("none", "asset", "time"), \
        "fixed_effects must be 'none', 'asset', or 'time'"

    Q_fitted_df = pd.DataFrame(np.nan, index=returns_df.index, columns=returns_df.columns)
    R_nonQ_df   = pd.DataFrame(np.nan, index=returns_df.index, columns=returns_df.columns)
    beta_rows   = {}

    for t_idx, t in enumerate(returns_df.index):
        # ── 1. stack window into long format ──────────────────────────────────
        start = max(0, t_idx - window + 1)
        R_win = returns_df.iloc[start : t_idx + 1]   # (window x N)
        Q_win = cot_df.iloc[start : t_idx + 1]

        # melt to (obs x 2): drop any row with NaN in either series
        R_long = R_win.stack()
        Q_long = Q_win.stack()
        both   = pd.concat([R_long, Q_long], axis=1, keys=["R", "Q"]).dropna()

        if len(both) < min_obs:
            continue

        R_vec = both["R"].values
        Q_vec = both["Q"].values

        # ── 2. apply fixed effects via demeaning ──────────────────────────────
        if fixed_effects == "asset":
            asset_idx = both.index.get_level_values(1)
            for asset in asset_idx.unique():
                sel = asset_idx == asset
                R_vec[sel] -= R_vec[sel].mean()
                Q_vec[sel] -= Q_vec[sel].mean()

        elif fixed_effects == "time":
            time_idx = both.index.get_level_values(0)
            for week in time_idx.unique():
                sel = time_idx == week
                R_vec[sel] -= R_vec[sel].mean()
                Q_vec[sel] -= Q_vec[sel].mean()

        # ── 3. pooled OLS: slope only (intercept absorbed by FE or left in) ──
        if fixed_effects == "none":
            b1, b0, _, _, _ = stats.linregress(Q_vec, R_vec)
        else:
            # after demeaning, regress through origin for the within estimator
            b1 = (Q_vec @ R_vec) / (Q_vec @ Q_vec) if (Q_vec @ Q_vec) > 0 else np.nan

        if np.isnan(b1):
            continue

        beta_rows[t] = b1

        # ── 4. apply current-week beta to current-week Q ──────────────────────
        Q_t = cot_df.loc[t]
        R_t = returns_df.loc[t]
        q_fit  = b1 * Q_t
        r_nonq = R_t - q_fit

        Q_fitted_df.loc[t] = q_fit.values
        R_nonQ_df.loc[t]   = r_nonq.values

    betas = pd.Series(beta_rows, name="b1_panel")
    return Q_fitted_df, R_nonQ_df, betas


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FAMA-MACBETH PREDICTIVE REGRESSION  (with Newey-West SEs)
# ══════════════════════════════════════════════════════════════════════════════

def _newey_west_se(residuals: np.ndarray, x: np.ndarray, lags: int = 6) -> float:
    """
    Newey-West corrected standard error for a single OLS coefficient.
    residuals and x are aligned 1-D arrays (intercept already absorbed).
    """
    T = len(residuals)
    scores = residuals * x
    # Gamma_0
    S = np.dot(scores, scores) / T
    for h in range(1, lags + 1):
        weight = 1 - h / (lags + 1)
        S += 2 * weight * np.dot(scores[h:], scores[:-h]) / T
    xx = np.dot(x, x) / T
    return np.sqrt(S / (xx ** 2) / T)


def fama_macbeth_regression(
    dep_df: pd.DataFrame,         # next-week return R_{i,t+1}
    signals: dict[str, pd.DataFrame],   # {"R_nonQ": ..., "Q": ..., ...}
    nw_lags: int = 6,
) -> pd.DataFrame:
    """
    Fama-MacBeth cross-sectional regression:
        R_{i,t+1} = c0 + Σ_k c_k * signal_k_{i,t} + u_{i,t+1}

    Runs one OLS per week, then computes time-series mean and Newey-West t-stat.

    Parameters
    ----------
    dep_df   : forward return DataFrame (shifted -1 from returns_df)
    signals  : ordered dict of signal DataFrames; all same shape as dep_df
    nw_lags  : Newey-West lag order (paper uses 6)

    Returns
    -------
    DataFrame with columns [coef, t_stat, p_value] for each predictor + intercept
    """
    signal_names = list(signals.keys())
    coef_rows = {name: [] for name in ["intercept"] + signal_names}
    dates_used = []

    common_dates = dep_df.index
    for sig_df in signals.values():
        common_dates = common_dates.intersection(sig_df.index)

    for t in common_dates:
        y = dep_df.loc[t]
        Xs = [signals[k].loc[t] for k in signal_names]
        # Align on valid observations
        valid = y.notna()
        for x in Xs:
            valid &= x.notna()
        if valid.sum() < max(5, len(signal_names) + 2):
            continue

        y_v = y[valid].values
        X_v = np.column_stack([np.ones(valid.sum())] + [x[valid].values for x in Xs])

        try:
            coefs, _, _, _ = np.linalg.lstsq(X_v, y_v, rcond=None)
        except np.linalg.LinAlgError:
            continue

        coef_rows["intercept"].append(coefs[0])
        for i, name in enumerate(signal_names):
            coef_rows[name].append(coefs[i + 1])
        dates_used.append(t)

    results = []
    for name in ["intercept"] + signal_names:
        series = np.array(coef_rows[name])
        if len(series) == 0:
            results.append({"predictor": name, "coef": np.nan,
                            "t_stat": np.nan, "p_value": np.nan,
                            "N": 0})
            continue
        mean_c = series.mean()
        # Newey-West SE on the time-series of cross-sectional coefficients
        T = len(series)
        demeaned = series - mean_c
        S = np.dot(demeaned, demeaned) / T
        for h in range(1, nw_lags + 1):
            w = 1 - h / (nw_lags + 1)
            S += 2 * w * np.dot(demeaned[h:], demeaned[:-h]) / T
        se = np.sqrt(S / T)
        t_stat = mean_c / se if se > 0 else np.nan
        p_val = 2 * stats.t.sf(abs(t_stat), df=T - 1) if not np.isnan(t_stat) else np.nan
        results.append({"predictor": name, "coef": mean_c,
                        "t_stat": t_stat, "p_value": p_val, "N": T})

    return pd.DataFrame(results).set_index("predictor")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PORTFOLIO-SORTING BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_sort_backtest(
    signal_df: pd.DataFrame,
    fwd_returns_df: pd.DataFrame,
    n_quantiles: int = 3,
    direction: int = 1,
    transaction_cost: float = 0.0005,
) -> pd.DataFrame:
    """
    Equal-weighted long-short: top quantile long, bottom quantile short.

    direction = +1: long high signal  (momentum: R_nonQ)
    direction = -1: long low  signal  (reversal: Q  → long -Q)

    Returns DataFrame[date, long_ret, short_ret, ls_ret].
    """
    records = []
    common = signal_df.index.intersection(fwd_returns_df.index)

    for t in common:
        sig = (signal_df.loc[t] * direction).dropna()
        if len(sig) < n_quantiles * 2:
            continue
        fwd = fwd_returns_df.loc[t].dropna()
        sig = sig[sig.index.isin(fwd.index)]
        if len(sig) < n_quantiles * 2:
            continue

        labels = pd.qcut(sig, n_quantiles, labels=False, duplicates="drop")
        top = sig.index[labels == labels.max()]
        bot = sig.index[labels == labels.min()]

        long_r  = fwd[top].mean()  - transaction_cost
        short_r = fwd[bot].mean()  - transaction_cost
        records.append({"date": t,
                        "long_ret":  long_r,
                        "short_ret": short_r,
                        "ls_ret":    long_r - short_r})

    return pd.DataFrame(records).set_index("date")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  IMPROVED INTERMEDIATE-TERM MOMENTUM
#     Sort on cumulative R_nonQ over H-week window instead of raw R
# ══════════════════════════════════════════════════════════════════════════════

def improved_momentum_backtest(
    R_nonQ_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    formation_windows: list[int] = [13, 26, 39, 52],
    skip_week: int = 1,
    n_quantiles: int = 3,
    transaction_cost: float = 0.001,
) -> dict[str, pd.DataFrame]:
    """
    For each formation window H (weeks):
        Signal_t = cumulative R_nonQ over [t-H-skip, t-skip]
        Compare vs. same window using raw return R.

    Returns dict keyed by f"H{H}" with sub-dict {"R_nonQ": bt_df, "R_raw": bt_df}.
    """
    results = {}
    for H in formation_windows:
        # Cumulative R_nonQ signal
        cum_r_nonq = R_nonQ_df.rolling(H).sum().shift(skip_week)
        # Cumulative raw return signal
        cum_r_raw  = returns_df.rolling(H).sum().shift(skip_week)

        # Forward returns (next week)
        fwd = returns_df.shift(-1)

        bt_nonq = portfolio_sort_backtest(
            cum_r_nonq, fwd, n_quantiles=n_quantiles,
            direction=1, transaction_cost=transaction_cost
        )
        bt_raw = portfolio_sort_backtest(
            cum_r_raw, fwd, n_quantiles=n_quantiles,
            direction=1, transaction_cost=transaction_cost
        )

        results[f"H{H}"] = {"R_nonQ": bt_nonq, "R_raw": bt_raw}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  VARIANCE CHANNEL  (volatility regime conditioning)
# ══════════════════════════════════════════════════════════════════════════════

def variance_channel_backtest(
    R_nonQ_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    vol_window: int = 52,
    transaction_cost: float = 0.0005,
) -> dict[str, pd.DataFrame]:
    """
    Split sample each week into Low-Vol and High-Vol regimes (cross-sectional median).
    Run R_nonQ momentum within each sub-sample.

    Paper finding: momentum is ~2x stronger in Low-Vol regime.
    """
    vol_df = returns_df.rolling(vol_window).std()
    fwd    = returns_df.shift(-1)

    lo_records, hi_records = [], []
    common = R_nonQ_df.index.intersection(fwd.index).intersection(vol_df.index)

    for t in common:
        sig  = R_nonQ_df.loc[t].dropna()
        vol  = vol_df.loc[t]
        fwd_t = fwd.loc[t].dropna()

        # Align
        valid = sig.index.intersection(vol.dropna().index).intersection(fwd_t.index)
        if len(valid) < 6:
            continue
        sig_v = sig[valid]; vol_v = vol[valid]; fwd_v = fwd_t[valid]

        median_vol = vol_v.median()
        lo_mask = vol_v <= median_vol
        hi_mask = ~lo_mask

        def _ls(mask):
            s = sig_v[mask]; f = fwd_v[mask]
            if len(s) < 4:
                return np.nan
            labels = pd.qcut(s, 3, labels=False, duplicates="drop")
            if labels.isna().all():
                return np.nan
            top = s.index[labels == labels.max()]
            bot = s.index[labels == labels.min()]
            return (f[top].mean() - f[bot].mean()) - 2 * transaction_cost

        lo_records.append({"date": t, "ls_ret": _ls(lo_mask)})
        hi_records.append({"date": t, "ls_ret": _ls(hi_mask)})

    lo_bt = pd.DataFrame(lo_records).set_index("date")
    hi_bt = pd.DataFrame(hi_records).set_index("date")
    return {"LowVol": lo_bt, "HighVol": hi_bt}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CROWDING CHANNEL  (crowding regime conditioning)
# ══════════════════════════════════════════════════════════════════════════════

def crowding_channel_backtest(
    R_nonQ_df: pd.DataFrame,
    cot_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    crowding_window: int = 52,
    transaction_cost: float = 0.0005,
) -> dict[str, pd.DataFrame]:
    """
    Crowding_{i,t} = (NetLong_t / OI_t) - 52wk MA  (proxied here by Q cumsum)
    Split by Low/High crowding relative to signal direction.

    Paper finding: R_nonQ momentum nearly vanishes in the High-Crowding regime.
    """
    # Proxy crowding as deviation of cumulative Q from 52-wk MA
    cum_q  = cot_df.rolling(crowding_window).sum()
    mean_q = cot_df.rolling(crowding_window).mean() * crowding_window
    crowding_df = cum_q - mean_q

    fwd = returns_df.shift(-1)
    lo_records, hi_records = [], []
    common = R_nonQ_df.index.intersection(fwd.index).intersection(crowding_df.index)

    for t in common:
        sig  = R_nonQ_df.loc[t].dropna()
        crowd = crowding_df.loc[t]
        fwd_t = fwd.loc[t].dropna()

        valid = sig.index.intersection(crowd.dropna().index).intersection(fwd_t.index)
        if len(valid) < 6:
            continue
        sig_v   = sig[valid]
        crowd_v = crowd[valid]
        fwd_v   = fwd_t[valid]

        # Within winners (positive R_nonQ), split by crowding
        winner_mask = sig_v > 0
        loser_mask  = sig_v < 0

        def _momentum_in_regime(w_mask, crowd_thresh):
            """Among assets in w_mask, long those with crowding <= thresh (low crowding)."""
            lo_crowd = crowd_v[w_mask] <= crowd_thresh
            hi_crowd = ~lo_crowd
            lo_ret = fwd_v[w_mask][lo_crowd].mean() if lo_crowd.sum() > 0 else np.nan
            hi_ret = fwd_v[w_mask][hi_crowd].mean() if hi_crowd.sum() > 0 else np.nan
            return lo_ret, hi_ret

        median_crowd = crowd_v.median()
        lo_long, hi_long = _momentum_in_regime(winner_mask, median_crowd)
        lo_short, hi_short = _momentum_in_regime(loser_mask, median_crowd)

        lo_ls = ((lo_long or 0) - (lo_short or 0)) - 2 * transaction_cost
        hi_ls = ((hi_long or 0) - (hi_short or 0)) - 2 * transaction_cost
        lo_records.append({"date": t, "ls_ret": lo_ls})
        hi_records.append({"date": t, "ls_ret": hi_ls})

    lo_bt = pd.DataFrame(lo_records).set_index("date")
    hi_bt = pd.DataFrame(hi_records).set_index("date")
    return {"LowCrowding": lo_bt, "HighCrowding": hi_bt}


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MULTI-HORIZON DECOMPOSITION  (Table 9 replication)
# ══════════════════════════════════════════════════════════════════════════════

def multi_horizon_analysis(
    R_nonQ_df: pd.DataFrame,
    Q_fitted_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizons: dict[str, tuple[int, int]] = {
        "1-3w":  (1, 3),
        "4-13w": (4, 13),
        "14-26w": (14, 26),
        "27-52w": (27, 52),
    },
    n_quantiles: int = 3,
) -> pd.DataFrame:
    """
    Sort commodities into terciles by R_nonQ_t.
    Compute subsequent cumulative returns (total R, flow Q, orthogonal R_nonQ)
    over each horizon interval.

    Returns a summary DataFrame matching Table 9 in the paper.
    """
    rows = []
    dates = R_nonQ_df.dropna(how="all").index

    for t_idx, t in enumerate(dates):
        sig = R_nonQ_df.loc[t].dropna()
        if len(sig) < n_quantiles * 2:
            continue
        labels = pd.qcut(sig, n_quantiles, labels=["Low", "Med", "High"],
                         duplicates="drop")
        labels = labels.dropna()

        for name, (h_start, h_end) in horizons.items():
            for group in ["Low", "Med", "High"]:
                tickers = labels[labels == group].index
                # Cumulative returns over [t + h_start, t + h_end]
                future_slice = slice(t_idx + h_start, t_idx + h_end + 1)
                if t_idx + h_end >= len(dates):
                    continue

                cum_R = returns_df.iloc[future_slice][tickers].mean().mean()
                cum_Q = Q_fitted_df.iloc[future_slice][tickers].mean().mean()
                cum_RnonQ = R_nonQ_df.iloc[future_slice][tickers].mean().mean()

                rows.append({"date": t, "horizon": name, "group": group,
                             "cum_R": cum_R, "cum_Q": cum_Q, "cum_RnonQ": cum_RnonQ})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    summary = (
        df.groupby(["horizon", "group"])[["cum_R", "cum_Q", "cum_RnonQ"]]
          .mean()
          .round(4)
    )

    # Add High-Low spread
    hl_rows = []
    for horizon in summary.index.get_level_values("horizon").unique():
        high = summary.loc[(horizon, "High")]
        low  = summary.loc[(horizon, "Low")]
        hl   = (high - low).rename((horizon, "High-Low"))
        hl_rows.append(hl)

    if hl_rows:
        hl_df = pd.DataFrame(hl_rows)
        hl_df.index = pd.MultiIndex.from_tuples(hl_df.index, names=["horizon","group"])
        summary = pd.concat([summary, hl_df]).sort_index()

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def performance_summary(
    ls_series: pd.Series,
    periods_per_year: int = 52,
    label: str = "",
) -> pd.Series:
    s = ls_series.dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)
    ann_ret  = s.mean() * periods_per_year
    ann_vol  = s.std()  * np.sqrt(periods_per_year)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum      = (1 + s).cumprod()
    max_dd   = (cum / cum.cummax() - 1).min()
    hit      = (s > 0).mean()
    t_stat, p_val = stats.ttest_1samp(s, 0)
    return pd.Series({
        "Ann. Return":  f"{ann_ret:+.2%}",
        "Ann. Vol":     f"{ann_vol:.2%}",
        "Sharpe":       f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Hit Rate":     f"{hit:.2%}",
        "t-stat":       f"{t_stat:.2f}",
        "p-value":      f"{p_val:.3f}",
        "N (weeks)":    len(s),
    }, name=label or "Strategy")


def _regime_comparison(lo_bt, hi_bt, label_lo, label_hi):
    lo_s = performance_summary(lo_bt["ls_ret"].dropna(), label=label_lo)
    hi_s = performance_summary(hi_bt["ls_ret"].dropna(), label=label_hi)
    return pd.DataFrame({label_lo: lo_s, label_hi: hi_s})


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SEP = "=" * 70

    print(SEP)
    print("  Momentum & Reversal Decomposition Backtest")
    print("  Ding, Kang, Yu, Zhao (2026)")
    print(SEP)

    # ── 0. Load data ────────────────────────────────────────────────────────
    # Replace generate_sample_data() with real data loading, e.g.:
    #
    #   returns_df  = pd.read_parquet("data/commodity_returns_weekly.parquet")
    #   cot_df      = pd.read_parquet("data/cftc_net_change_scaled.parquet")
    #   basis_df    = pd.read_parquet("data/basis.parquet")          # optional
    #   basismom_df = pd.read_parquet("data/basis_momentum.parquet") # optional
    #   shp_df      = pd.read_parquet("data/smoothed_hedging_pressure.parquet") # optional
    #
    print("\n[0] Loading data ...")
    returns_df, cot_df, basis_df, basismom_df, shp_df = generate_sample_data(
        n_weeks=1664, n_assets=26
    )
    print(f"    {len(returns_df)} weeks × {returns_df.shape[1]} commodities "
          f"({returns_df.index[0].date()} – {returns_df.index[-1].date()})")

    # ── 1. Decompose ────────────────────────────────────────────────────────
    print("\n[1] Cross-sectional return decomposition (Fama-MacBeth OLS) ...")
    Q_fitted_df, R_nonQ_df, betas = decompose_returns(returns_df, cot_df)

    valid_betas = betas.dropna()
    print(f"    b1 (Q → R): mean={valid_betas.mean():.3f}, "
          f"std={valid_betas.std():.3f}, "
          f"t-stat={valid_betas.mean()/valid_betas.sem():.1f}")
    print(f"    R_nonQ std (cross-sectional avg): "
          f"{R_nonQ_df.stack().std():.4f}")

    # ── 2. Fama-MacBeth predictive regressions ──────────────────────────────
    print("\n[2] Fama-MacBeth predictive regressions ...")
    fwd_returns = returns_df.shift(-1)

    signals_panel_A = {"R_raw": returns_df}
    fm_A = fama_macbeth_regression(fwd_returns, signals_panel_A)
    print("\n  Panel A: univariate raw return (expect ~flat after controls)")
    print(fm_A.to_string(float_format="{:.4f}".format))

    signals_panel_B = {"R_nonQ": R_nonQ_df, "Q": Q_fitted_df}
    fm_B = fama_macbeth_regression(fwd_returns, signals_panel_B)
    print("\n  Panel B: R_nonQ and Q (core decomposition result)")
    print(fm_B.to_string(float_format="{:.4f}".format))
    print("\n  Paper predictions: coef(R_nonQ) > 0, coef(Q) < 0")

    # ── 3. Portfolio sorting: short-term momentum and reversal ───────────────
    print("\n[3] Portfolio sorting backtests ...")
    fwd_aligned = fwd_returns.loc[R_nonQ_df.index]

    bt_mom = portfolio_sort_backtest(R_nonQ_df,   fwd_aligned, direction= 1,
                                     n_quantiles=3, transaction_cost=0.0005)
    bt_rev = portfolio_sort_backtest(Q_fitted_df, fwd_aligned, direction=-1,
                                     n_quantiles=3, transaction_cost=0.0005)
    raw_sig_aligned = returns_df.loc[R_nonQ_df.index]
    bt_naive = portfolio_sort_backtest(raw_sig_aligned, fwd_aligned, direction=1,
                                       n_quantiles=3, transaction_cost=0.0005)

    summary_core = pd.DataFrame({
        "R_nonQ Momentum":  performance_summary(bt_mom["ls_ret"]),
        "Q Reversal":       performance_summary(bt_rev["ls_ret"]),
        "Naive R Momentum": performance_summary(bt_naive["ls_ret"]),
    })
    print("\n" + SEP)
    print("  CORE STRATEGY PERFORMANCE")
    print(SEP)
    print(summary_core.to_string())
    print("\n  Paper: R_nonQ momentum ≈ 6.2% p.a. (1-std-dev in R_nonQ → +11.6 bps/wk)")
    print("         Q reversal is significant and negative-direction (reversal = long -Q)")

    # ── 4. Improved intermediate-term momentum ───────────────────────────────
    print("\n[4] Improved intermediate-term momentum (R_nonQ vs. raw R) ...")
    im_results = improved_momentum_backtest(
        R_nonQ_df, returns_df,
        formation_windows=[13, 26, 39, 52],
        transaction_cost=0.001,
    )

    im_summary_rows = []
    for window_key, bts in im_results.items():
        perf_nonq = performance_summary(bts["R_nonQ"]["ls_ret"])
        perf_raw  = performance_summary(bts["R_raw"]["ls_ret"])
        im_summary_rows.append({
            "Formation": window_key,
            "R_nonQ Ann.Ret": perf_nonq["Ann. Return"],
            "R_nonQ Sharpe":  perf_nonq["Sharpe"],
            "R_raw Ann.Ret":  perf_raw["Ann. Return"],
            "R_raw Sharpe":   perf_raw["Sharpe"],
        })

    im_summary = pd.DataFrame(im_summary_rows).set_index("Formation")
    print("\n" + SEP)
    print("  IMPROVED INTERMEDIATE-TERM MOMENTUM")
    print(SEP)
    print(im_summary.to_string())
    print("\n  Paper: at H=26wk, R_nonQ gives 9.9% p.a. vs. 5.2% p.a. for raw R (+4.7%)")

    # ── 5. Variance channel ──────────────────────────────────────────────────
    print("\n[5] Variance channel: momentum by volatility regime ...")
    var_results = variance_channel_backtest(R_nonQ_df, returns_df,
                                            transaction_cost=0.0005)
    var_summary = _regime_comparison(
        var_results["LowVol"], var_results["HighVol"],
        "Low Vol (stronger MOM)", "High Vol (weaker MOM)"
    )
    print("\n" + SEP)
    print("  VARIANCE CHANNEL (momentum by volatility regime)")
    print(SEP)
    print(var_summary.to_string())
    print("\n  Paper: trend-chasing and momentum are ~2x stronger in Low-Vol regime")

    # ── 6. Crowding channel ──────────────────────────────────────────────────
    print("\n[6] Crowding channel: momentum by position crowding regime ...")
    crowd_results = crowding_channel_backtest(R_nonQ_df, cot_df, returns_df,
                                              transaction_cost=0.0005)
    crowd_summary = _regime_comparison(
        crowd_results["LowCrowding"], crowd_results["HighCrowding"],
        "Low Crowding (early-stage MOM)", "High Crowding (late-stage MOM)"
    )
    print("\n" + SEP)
    print("  CROWDING CHANNEL (momentum by position crowding regime)")
    print(SEP)
    print(crowd_summary.to_string())
    print("\n  Paper: momentum nearly vanishes in High-Crowding regime")

    # ── 7. Multi-horizon decomposition ──────────────────────────────────────
    print("\n[7] Multi-horizon return decomposition (Table 9 analog) ...")
    # Limit to last 500 dates for speed on synthetic data
    r_nonq_sub = R_nonQ_df.iloc[-500:]
    q_sub      = Q_fitted_df.iloc[-500:]
    ret_sub    = returns_df.iloc[-500:]
    mh_summary = multi_horizon_analysis(r_nonq_sub, q_sub, ret_sub)

    print("\n" + SEP)
    print("  MULTI-HORIZON DECOMPOSITION")
    print("  (sorted by R_nonQ: cumulative avg returns for High/Med/Low terciles)")
    print(SEP)
    if not mh_summary.empty:
        print(mh_summary.to_string(float_format="{:.4f}".format))
    print("\n  Paper Table 9: High-Low total return: +0.54% (wks 1-3), +0.31% (wks 4-13),")
    print("                  +0.73% (wks 14-26), +1.22% (wks 27-52)")

    # ── 8. Export equity curves ──────────────────────────────────────────────
    output_path = "equity_curves.csv"
    curves = pd.DataFrame({
        "R_nonQ_Momentum": (1 + bt_mom["ls_ret"]).cumprod(),
        "Q_Reversal":      (1 + bt_rev["ls_ret"]).cumprod(),
        "Naive_Momentum":  (1 + bt_naive["ls_ret"]).cumprod(),
    })
    curves.to_csv(output_path)
    print(f"\nEquity curves saved → {output_path}")
    print("\nDone.")

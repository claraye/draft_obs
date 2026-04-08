"""
model.py — Bayesian state-space model for Volume-Driven Time-of-Day intraday volatility.

Paper: "Volume-Driven Time-of-Day Effects in Intraday Volatility Models"
       Fruet Dias, Medeiros, Astorino, Pegoraro (2025)

Full model (Appendix B, eqns 2-7):
    log(y²_t) = h_t + u_t,          u_t ~ N(μ_{c_t}, σ²_{c_t})  [mixture approx]
    h_t = m₀ + x_t + tod^p_t + tod^v_t + so_t
    x_t = φ x_{t-1} + σ_x η_t,      η_t ~ N(0,1)
    tod^p_t = I'_t β^p,              Σ_k β^p_k = 0   (zero-sum restriction)
    tod^v_t = v_{t-1} I'_t β^v
    so_t    = H'_t β^{so}

where:
    v_{t-1} = (log V_{t-1} − V̄_{k*}) / σ_{V,k*}
              k* = intraday interval of bar t-1
              V̄ and σ_V estimated from the training (estimation) sample only

Estimation via 8-step Gibbs sampler (Appendix B) using Kim-Shephard-Chib (KSC 1998)
7-component Gaussian mixture approximation so the SV model is conditionally Gaussian.

Classes
-------
VolumeToD_Model       Bayesian MCMC estimator — matches the paper's algorithm.
VolumeToD_OlsModel    OLS + EWMA approximation retained as a fast diagnostic baseline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Kim-Shephard-Chib (1998) mixture approximation for log(χ²₁) ──────────────
# log(ε²_t) is approximated by a 7-component Gaussian mixture:
#   log(ε²_t) ≈ Σ_j p_j · N(μ_j, σ²_j)
# This makes the otherwise non-Gaussian SV measurement equation conditionally
# Gaussian, enabling closed-form Kalman filtering.
# Constants from Table 4 of Kim, Shephard & Chib (1998, ReStud).
KSC_PROBS = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
KSC_MEANS = np.array([-10.12999, -3.97281, -8.56686,  2.77786,  0.61942,  1.79518, -1.08819])
KSC_VARS  = np.array([  5.79596,   2.61369,  5.17950,  0.16735,  0.64009,  0.34023,  1.26261])

# Unconditional moments of the mixture — used for the OOS Kalman filter update
# where we cannot re-sample mixture indicators (no future data available).
LOG_CHI2_MEAN = float(np.dot(KSC_PROBS, KSC_MEANS))
LOG_CHI2_VAR  = float(np.dot(KSC_PROBS, KSC_VARS + KSC_MEANS ** 2) - LOG_CHI2_MEAN ** 2)


# ── Low-level sampling helpers ────────────────────────────────────────────────

def _safe_inverse_gamma(rng: np.random.Generator, shape: float, scale: float) -> float:
    """
    Sample X ~ IG(shape, scale) where E[X] = scale / (shape - 1).

    Uses the identity: if G ~ Gamma(shape, 1/scale) then 1/G ~ IG(shape, scale).
    numpy's rng.gamma(a, b) has mean=a·b, so rng.gamma(shape, 1/scale) ~ Gamma(shape, 1/scale).
    """
    return float(1.0 / rng.gamma(shape, 1.0 / max(scale, 1e-12)))


def _weighted_posterior_sample(
    rng:        np.random.Generator,
    X:          np.ndarray,
    y:          np.ndarray,
    obs_var:    np.ndarray,
    prior_mean: np.ndarray,
    prior_var:  np.ndarray,
) -> np.ndarray:
    """
    Sample from the conjugate posterior of a heteroskedastic Bayesian linear regression.

    Model:  y_t = X_t θ + ε_t,   ε_t ~ N(0, obs_var_t)
    Prior:  θ ~ N(prior_mean, diag(prior_var))
    Posterior: θ|y ~ N(post_mean, post_cov) where
        post_precision = diag(1/prior_var) + X'W X,   W = diag(1/obs_var)
        post_mean      = post_cov @ (diag(1/prior_var) prior_mean + X'W y)

    This implements Appendix B steps 5, 6, 7, 8 when called separately per block,
    or all at once for a joint block Gibbs draw (code uses the joint version for
    efficiency; stationary distribution is identical).
    """
    prior_precision = np.diag(1.0 / prior_var)
    w      = 1.0 / obs_var
    xt_w   = X.T * w
    post_precision = prior_precision + xt_w @ X
    post_cov       = np.linalg.inv(post_precision)
    rhs            = prior_precision @ prior_mean + xt_w @ y
    post_mean      = post_cov @ rhs
    # Symmetrise before Cholesky to prevent tiny numerical asymmetry failures.
    chol = np.linalg.cholesky(
        (post_cov + post_cov.T) / 2.0 + 1e-10 * np.eye(len(post_mean))
    )
    return post_mean + chol @ rng.standard_normal(len(post_mean))


def _sample_mixture_indicators(rng: np.random.Generator, residual: np.ndarray) -> np.ndarray:
    """
    Appendix B — Step 3: Sample mixture component indicators {c_t}.

    For each bar t, compute the posterior probability of belonging to each
    of the 7 Gaussian mixture components given the current residual:
        residual_t = log(y²_t) - h_t = log(ε²_t)

    Posterior: π(c_t=j | residual_t) ∝ p_j · N(residual_t; μ_j, σ²_j)

    Uses log-sum-exp trick for numerical stability.
    Returns integer array of component indices in {0,...,6}.
    """
    log_probs = np.empty((len(residual), len(KSC_PROBS)))
    for j in range(len(KSC_PROBS)):
        log_probs[:, j] = (
            np.log(KSC_PROBS[j])
            - 0.5 * np.log(KSC_VARS[j])
            - 0.5 * (residual - KSC_MEANS[j]) ** 2 / KSC_VARS[j]
        )
    # Stabilise: subtract row-max before exp
    log_probs -= log_probs.max(axis=1, keepdims=True)
    probs  = np.exp(log_probs)
    probs /= probs.sum(axis=1, keepdims=True)
    return np.array([rng.choice(len(KSC_PROBS), p=row) for row in probs], dtype=int)


def _ffbs_sample_x(
    rng:       np.random.Generator,
    y:         np.ndarray,
    obs_var:   np.ndarray,
    phi:       float,
    sigma2_x:  float,
    x0_mean:   float = 0.0,
    x0_var:    float | None = None,
) -> np.ndarray:
    """
    Appendix B — Step 4: Carter-Kohn Forward-Filter Backward-Sampler (FFBS).

    State-space form (after removing deterministic and mixture-mean):
        Measurement: y_t = x_t + ε_t,   ε_t ~ N(0, obs_var_t)
        State:       x_t = φ x_{t-1} + η_t,  η_t ~ N(0, σ²_x)

    The FFBS algorithm:
    ─ Forward pass (Kalman filter):
        Predict:  a_t = φ m_{t-1},  r_t = φ² c_{t-1} + σ²_x
        Update:   K_t = r_t / (r_t + obs_var_t)
                  m_t = a_t + K_t (y_t - a_t)
                  c_t = (1 - K_t) r_t
    ─ Backward pass (simulation smoother):
        x_T ~ N(m_T, c_T)
        For t = T-1,...,0:
            b_t = c_t φ / r_{t+1}
            x_t | x_{t+1} ~ N(m_t + b_t(x_{t+1} - a_{t+1}), c_t - b²_t r_{t+1})

    Returns one sampled path x_{1:T} from the joint conditional posterior.
    """
    n = len(y)
    a = np.zeros(n)   # predicted state mean
    r = np.zeros(n)   # predicted state variance
    m = np.zeros(n)   # filtered state mean
    c = np.zeros(n)   # filtered state variance

    # Stationary variance for initialisation (x₀ ~ N(x0_mean, x0_var))
    if x0_var is None:
        x0_var = sigma2_x / max(1.0 - phi ** 2, 1e-4)

    a[0] = x0_mean
    r[0] = max(x0_var, 1e-8)

    # ── Forward pass ─────────────────────────────────────────────────────────
    for t in range(n):
        if t > 0:
            a[t] = phi * m[t - 1]
            r[t] = phi ** 2 * c[t - 1] + sigma2_x
        q     = r[t] + obs_var[t]
        k     = r[t] / max(q, 1e-8)
        m[t]  = a[t] + k * (y[t] - a[t])
        c[t]  = max((1.0 - k) * r[t], 1e-8)

    # ── Backward pass ─────────────────────────────────────────────────────────
    x = np.zeros(n)
    x[-1] = rng.normal(m[-1], np.sqrt(c[-1]))
    for t in range(n - 2, -1, -1):
        b    = c[t] * phi / max(r[t + 1], 1e-8)
        mean = m[t] + b * (x[t + 1] - a[t + 1])
        var  = max(c[t] - b ** 2 * r[t + 1], 1e-8)
        x[t] = rng.normal(mean, np.sqrt(var))

    return x


def _sample_phi(
    rng:        np.random.Generator,
    x:          np.ndarray,
    sigma2_x:   float,
    prior_mean: float = 0.95,
    prior_var:  float = 0.25,
) -> float:
    """
    Appendix B — Step 1: Sample φ | Ψ, y from its conjugate Gaussian posterior.

    The AR(1) state equation x_t = φ x_{t-1} + η_t gives a linear regression:
        x_t = φ x_{t-1} + η_t,   η_t ~ N(0, σ²_x)

    Prior:     φ ~ N(φ₀=0.95, V_{φ₀}=0.25)   [paper Section 2.3]
    Posterior:
        V̂_φ = [1/V_{φ₀} + (1/σ²_x) Σ x²_{t-1}]⁻¹
        φ̂   = V̂_φ [φ₀/V_{φ₀} + (1/σ²_x) Σ x_t x_{t-1}]

    Stationarity rejection sampling: resample until |φ| < 0.999.
    """
    x_prev = x[:-1]
    x_curr = x[1:]
    post_var  = 1.0 / (1.0 / prior_var + np.dot(x_prev, x_prev) / sigma2_x)
    post_mean = post_var * (prior_mean / prior_var + np.dot(x_prev, x_curr) / sigma2_x)

    # Rejection sampling to enforce stationarity
    for _ in range(200):
        phi = rng.normal(post_mean, np.sqrt(post_var))
        if abs(phi) < 0.999:
            return float(phi)
    return float(np.clip(post_mean, -0.999, 0.999))


def _sample_sigma2_x(
    rng:    np.random.Generator,
    x:      np.ndarray,
    phi:    float,
    alpha0: float = 1.0,
    beta0:  float = 10.0,
) -> float:
    """
    Appendix B — Step 2: Sample σ²_x | Ψ, y from its conjugate inverse-Gamma posterior.

    Prior:     σ²_x ~ IG(α₀=1, β₀=10)   [paper Section 2.3]
    Posterior: σ²_x | Ψ ~ IG(α₀ + T/2, β₀ + Σ(x_t - φ x_{t-1})² / 2)

    Sampling: X ~ IG(a,b) ⟺ 1/X ~ Gamma(a, 1/b).
    """
    resid = x[1:] - phi * x[:-1]
    shape = alpha0 + 0.5 * len(resid)
    scale = beta0  + 0.5 * np.dot(resid, resid)
    return _safe_inverse_gamma(rng, shape, scale)


# ── Design matrix builders ────────────────────────────────────────────────────

def _build_static_design(interval_values: np.ndarray, intervals: list[int]) -> np.ndarray:
    """
    Build the design matrix for the static ToD component tod^p_t = I'_t β^p.

    Zero-sum restriction: Σ_k β^p_k = 0 (Appendix B step 6 and equation 3).
    Achieved by using the contrast coding: Ĩ_{t,k} = I_{t,k} - I_{t,K}
    for k = 1,...,K-1, with the K-th interval's coefficient set to
    β^p_K = -Σ_{k=1}^{K-1} β^p_k after sampling.

    Returns matrix of shape (T, K-1).
    """
    if len(intervals) <= 1:
        return np.empty((len(interval_values), 0))
    last = intervals[-1]
    return np.column_stack([
        (interval_values == iv).astype(float) - (interval_values == last).astype(float)
        for iv in intervals[:-1]
    ])


def _reconstruct_beta_p(beta_free: np.ndarray, intervals: list[int]) -> dict[int, float]:
    """
    Recover the K-th interval's β^p_K from the zero-sum restriction:
        β^p_K = -Σ_{k=1}^{K-1} β^p_k
    """
    if len(intervals) == 1:
        return {intervals[0]: 0.0}
    beta_last = -float(beta_free.sum())
    result = {iv: float(beta_free[i]) for i, iv in enumerate(intervals[:-1])}
    result[intervals[-1]] = beta_last
    return result


def _build_volume_design(vol_dev_lag: np.ndarray, interval_values: np.ndarray,
                         intervals: list[int]) -> np.ndarray:
    """
    Build the design matrix for the volume ToD component tod^v_t = v_{t-1} I'_t β^v.

    Column k is v_{t-1} · 1[bar t is in interval k].
    No sum-to-zero restriction needed because v_{t-1} is demeaned by construction.

    Returns matrix of shape (T, K).
    """
    return np.column_stack([
        vol_dev_lag * (interval_values == iv).astype(float)
        for iv in intervals
    ])


def _build_sunday_design(panel: pd.DataFrame, n_lags: int) -> np.ndarray:
    """
    Build the spillover design matrix H_t for the Sunday reopening effect.

    so_t = H'_t β^{so} captures the elevated volatility in the first m_lags
    bars after the Sunday 18:00 ET market reopen.

    H_t[:,j] = 1 if bar t is the j-th bar after the Sunday open, 0 otherwise.
    Requires panel["timestamp"] (raw tz-aware datetime) for weekday detection.

    Returns matrix of shape (T, n_lags). If n_lags=0, returns empty (T,0).
    """
    if n_lags <= 0:
        return np.empty((len(panel), 0))

    # "timestamp" column set by prepare_intraday from the raw datetime index
    ts = pd.to_datetime(panel["timestamp"])
    is_sunday = ts.dt.weekday.eq(6).to_numpy()     # Sunday = weekday 6
    bar_idx   = panel["bar_idx"].to_numpy(dtype=int)

    # bar_idx=0 is the 18:00 ET bar of the session; sessions start on Sunday
    # evening for Monday's trade date. Lag j captures bar j after the open.
    return np.column_stack([
        (is_sunday & (bar_idx == j)).astype(float)
        for j in range(n_lags)
    ])


def _vol_dev_lag(panel: pd.DataFrame) -> np.ndarray:
    """
    Compute v_{t-1}: the pure time-series lag of the standardised log-volume deviation.

    *** PAPER ALIGNMENT ***
    Equation (4): v_{t-1} = (V_{t-1} - V̄_{k*}) / σ_{V,k*}  where k* = w(t-1).
    The lag is a pure time-series shift — it does NOT reset at session boundaries.
    At the first bar of each new session, v_{t-1} takes the standardised volume
    from the last bar of the previous session (not zero).

    panel must be sorted by (date, bar_idx), which is global time order.
    Only the very first bar of the entire time series is padded with 0.
    """
    return panel["vol_dev"].shift(1).fillna(0.0).to_numpy(dtype=float)


def _build_design_parts(
    panel:            pd.DataFrame,
    intervals:        list[int],
    sunday_open_lags: int,
) -> dict[str, np.ndarray]:
    """Assemble all design matrices for a given panel."""
    interval_values = panel["interval"].to_numpy(dtype=int)
    vol_lag = _vol_dev_lag(panel)
    return {
        "static":           _build_static_design(interval_values, intervals),
        "volume":           _build_volume_design(vol_lag, interval_values, intervals),
        "sunday":           _build_sunday_design(panel, sunday_open_lags),
        "vol_dev_lag":      vol_lag,
        "interval_values":  interval_values,
    }


# ── Base class with shared forecast/diagnostics ───────────────────────────────

class _BaseVolumeModel:
    def __init__(self, log_var_clip: float = -20.0):
        self.log_var_clip = log_var_clip
        self._trained     = False

    def forecast_daily_vol(self, panel_pred: pd.DataFrame) -> pd.Series:
        """
        Aggregate bar-level log-variance forecasts to a daily vol forecast.

        Following Moreira & Muir (2017): sum bar-level variance forecasts across
        all intraday bars of the day, then take square root.
        σ̂_day = sqrt(Σ_t exp(ĥ_t))
        """
        if "h_pred" not in panel_pred.columns:
            raise ValueError("panel_pred must contain 'h_pred' column from predict()")
        daily_var = (
            panel_pred
            .assign(var_pred=lambda d: np.exp(d["h_pred"].clip(self.log_var_clip, 0.0)))
            .groupby("date")["var_pred"]
            .sum()
        )
        return np.sqrt(daily_var).rename("forecast_vol")

    def mz_r2(self, panel_pred: pd.DataFrame) -> float:
        """
        Mincer-Zarnowitz (1969) R²: regress realised log(y²_t) on forecast ĥ_t.

        R² measures the fraction of variation in the realised log-variance
        explained by the model forecast. Values of 0.55-0.75 are reported
        for the paper's models (Table 3, OOS evaluation).
        """
        common = panel_pred.dropna(subset=["log_r2", "h_pred"])
        if len(common) < 10:
            return np.nan
        X  = np.column_stack([np.ones(len(common)),
                               common["h_pred"].to_numpy(dtype=float)])
        y  = common["log_r2"].to_numpy(dtype=float)
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ b
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# ── Main Bayesian model ───────────────────────────────────────────────────────

class VolumeToD_Model(_BaseVolumeModel):
    """
    Bayesian state-space estimator matching the paper's Appendix B algorithm.

    8-step Gibbs sampler with:
    • KSC (1998) 7-component Gaussian mixture approximation for log(χ²₁)
    • Carter-Kohn FFBS for the latent AR(1) volatility component x_t
    • Conjugate Gaussian posteriors for regression blocks
    • Conjugate inverse-Gamma posterior for σ²_x

    Key design choices matching the paper:
    • v_{t-1} is a pure time-series lag (no session-boundary reset)
    • V̄_{k*} and σ_{V,k*} computed from training data ONLY (no look-ahead)
    • Zero-sum restriction on β^p for identification with m₀
    • Prior: φ~N(0.95, 0.25), σ²_x~IG(1,10), β^p~N(0,0.5I), β^v~N(0,0.5I),
             β^{so}~N(0,4I), m₀~N(ȳ*+1.27, 2)
    """

    def __init__(
        self,
        log_var_clip:     float = -20.0,
        n_iter:           int   = 600,    # total Gibbs iterations
        burn_in:          int   = 200,    # discard first burn_in draws
        thin:             int   = 2,      # keep every thin-th draw after burn-in
        sunday_open_lags: int   = 12,     # number of Sunday-open bars in so_t
        random_state:     int   = 42,
    ):
        super().__init__(log_var_clip=log_var_clip)
        self.n_iter           = n_iter
        self.burn_in          = burn_in
        self.thin             = thin
        self.sunday_open_lags = sunday_open_lags
        self.random_state     = random_state

        # Posterior mean estimates (set after fit)
        self.m0:       float           = 0.0
        self.phi:      float           = 0.95
        self.sigma2_x: float           = 0.05
        self.beta_p:   dict[int, float] = {}
        self.beta_v:   dict[int, float] = {}
        self.beta_so:  np.ndarray       = np.array([])

        # State initialisation for OOS prediction
        self._last_state_mean: float = 0.0
        self._last_state_var:  float = 1.0

        # Volume normalisation stats (computed from training data → applied OOS)
        # Shape: index=interval, columns=[vol_mean, vol_std]
        self._vol_stats_: pd.DataFrame | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_vol_normalization(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        (Re-)compute vol_dev using the stored training-sample statistics.

        BUG FIX: The volume standardisation V̄_{k*}, σ_{V,k*} must be computed
        from the estimation sample only (paper Section 2.2 and footnote 5).
        Calling build_volume_deviation() on the full panel before the train/test
        split would leak future volume information. This method always applies
        stored training stats, making OOS forecasting look-ahead-free.
        """
        if self._vol_stats_ is None:
            raise RuntimeError("Model not fit; call fit() before predict()")
        panel = panel.copy()
        mu  = panel["interval"].map(self._vol_stats_["vol_mean"])
        sig = panel["interval"].map(self._vol_stats_["vol_std"]).clip(lower=1e-8)
        panel["vol_dev"] = (panel["log_vol"] - mu) / sig
        return panel

    def _design_matrix(
        self, panel: pd.DataFrame
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Build the combined design matrix and return its components.

        Parameter order in γ = [m₀ | β^p_{1:K-1} | β^v_{1:K} | β^{so}_{1:m}]
        Column 0 is the intercept (= m₀).
        """
        parts = _build_design_parts(panel, self._intervals, self.sunday_open_lags)
        X = np.column_stack([
            np.ones(len(panel)),   # m₀
            parts["static"],       # β^p (K-1 free params; sum-to-zero coded)
            parts["volume"],       # β^v (K params)
            parts["sunday"],       # β^{so} (sunday_open_lags params)
        ])
        return X, parts

    def _gamma_to_components(
        self, gamma: np.ndarray
    ) -> tuple[float, dict[int, float], dict[int, float], np.ndarray]:
        """Unpack the flat γ vector into named model components."""
        K  = len(self._intervals)
        idx = 0
        m0           = float(gamma[idx]); idx += 1
        beta_p_free  = gamma[idx: idx + K - 1]; idx += K - 1
        beta_v_arr   = gamma[idx: idx + K];      idx += K
        beta_so      = gamma[idx:].copy()
        beta_p = _reconstruct_beta_p(beta_p_free, self._intervals)
        beta_v = {iv: float(beta_v_arr[i]) for i, iv in enumerate(self._intervals)}
        return m0, beta_p, beta_v, beta_so

    # ──────────────────────────────────────────────────────────────────────────
    # Fit
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, panel: pd.DataFrame) -> "VolumeToD_Model":
        """
        Estimate model on training panel via the 8-step Gibbs sampler.

        Implements Appendix B exactly, with one departure: steps 5-8 are drawn
        as a single joint Gaussian block rather than four sequential draws.
        The stationary distribution is the same; joint sampling is equivalent
        when the prior blocks are independent (which they are here).

        Parameters
        ----------
        panel : output of prepare_intraday(), must include 'timestamp' and 'log_vol'.
        """
        panel = (panel.copy()
                      .sort_values(["date", "bar_idx"])
                      .reset_index(drop=True))

        if "timestamp" not in panel.columns:
            raise ValueError("Panel must have a 'timestamp' column (from prepare_intraday).")

        self._intervals = sorted(panel["interval"].unique())
        K = len(self._intervals)
        if K < 2:
            raise ValueError("Need ≥2 intraday intervals to fit ToD model.")

        # ── Store training-sample volume normalisation stats ──────────────────
        # FIX: compute V̄_{k*} and σ_{V,k*} from training data ONLY.
        # These are reused in predict() so OOS vol_dev is computed with
        # estimation-sample parameters (matching the paper's protocol).
        self._vol_stats_ = (
            panel
            .groupby("interval")["log_vol"]
            .agg(vol_mean="mean", vol_std="std")
        )
        self._vol_stats_["vol_std"] = self._vol_stats_["vol_std"].clip(lower=1e-8)

        # Recompute vol_dev on the training panel using training stats only
        panel = self._apply_vol_normalization(panel)

        rng = np.random.default_rng(self.random_state)
        y   = panel["log_r2"].to_numpy(dtype=float)
        X, parts = self._design_matrix(panel)

        # ── Prior specification (paper Section 2.3) ───────────────────────────
        # m₀ ~ N(ȳ* + 1.27, 2)   where 1.27 = -E[log ε²] corrects for the
        #                         log-chi-squared transformation bias.
        mu_m0 = float(y.mean() + 1.27)
        prior_mean = np.concatenate([
            np.array([mu_m0]),                          # m₀
            np.zeros(max(K - 1, 0)),                    # β^p (N(0, 0.5·I))
            np.zeros(K),                                # β^v (N(0, 0.5·I))
            np.zeros(self.sunday_open_lags),            # β^{so} (N(0, 4·I))
        ])
        prior_var = np.concatenate([
            np.array([2.0]),
            np.full(max(K - 1, 0), 0.5),
            np.full(K,             0.5),
            np.full(self.sunday_open_lags, 4.0),
        ])

        # ── Initialise Markov chain ───────────────────────────────────────────
        gamma    = prior_mean.copy()
        phi      = 0.95
        sigma2_x = max(np.var(y), 1e-2) * 0.05
        x        = np.zeros(len(panel))

        kept_gamma  = []
        kept_phi    = []
        kept_sigma2 = []
        kept_x      = []

        for it in range(self.n_iter):

            # ── Deterministic component (current γ draw) ──────────────────────
            deterministic = X @ gamma

            # ─ STEP 3 (Appendix B): Sample mixture indicators {c_t} ──────────
            # Residual from the data after removing all deterministic and latent
            # components: residual_t = log(y²_t) - h_t = log(ε²_t)
            mix      = _sample_mixture_indicators(rng, y - deterministic - x)
            mix_mean = KSC_MEANS[mix]
            mix_var  = KSC_VARS[mix]

            # ─ STEP 4 (Appendix B): FFBS sample latent path {x_t} ────────────
            # Measurement after removing deterministic + mixture mean:
            #   y_t - deterministic_t - μ_{c_t} = x_t + ε_t,  ε_t ~ N(0, σ²_{c_t})
            x = _ffbs_sample_x(
                rng=rng,
                y=y - deterministic - mix_mean,
                obs_var=mix_var,
                phi=phi,
                sigma2_x=sigma2_x,
            )

            # ─ STEPS 5-8 (Appendix B, joint block): Sample γ = [m₀, β^p, β^v, β^{so}] ─
            # Conditioning on the sampled x_t, the model becomes a heteroskedastic
            # Bayesian linear regression:
            #   y_t - x_t - μ_{c_t} = X_t γ + ε_t,   ε_t ~ N(0, σ²_{c_t})
            # Prior: γ ~ N(prior_mean, diag(prior_var))
            # Posterior: conjugate Gaussian (see _weighted_posterior_sample).
            #
            # Note: the paper samples m₀ (step 5), β^p (step 6), β^v (step 7),
            # β^{so} (step 8) in four sequential draws, each conditioning on the
            # just-updated value of the others. This joint draw is equivalent
            # (same stationary distribution) because all blocks have independent
            # priors. It is marginally less efficient (slower mixing) but
            # considerably simpler to implement.
            gamma = _weighted_posterior_sample(
                rng=rng,
                X=X,
                y=y - x - mix_mean,
                obs_var=mix_var,
                prior_mean=prior_mean,
                prior_var=prior_var,
            )

            # ─ STEP 1 (Appendix B): Sample φ | Ψ, y ──────────────────────────
            phi = _sample_phi(rng, x, sigma2_x)

            # ─ STEP 2 (Appendix B): Sample σ²_x | Ψ, y ──────────────────────
            sigma2_x = _sample_sigma2_x(rng, x, phi)

            # Keep every thin-th draw after the burn-in period
            keep = it >= self.burn_in and ((it - self.burn_in) % max(self.thin, 1) == 0)
            if keep:
                kept_gamma.append(gamma.copy())
                kept_phi.append(phi)
                kept_sigma2.append(sigma2_x)
                kept_x.append(x.copy())

        if not kept_gamma:
            raise RuntimeError(
                f"No posterior draws kept. Increase n_iter (currently {self.n_iter}) "
                f"or lower burn_in (currently {self.burn_in})."
            )

        n_kept = len(kept_gamma)
        logger.info(f"  [model] Posterior mean from {n_kept} kept draws "
                    f"(n_iter={self.n_iter}, burn_in={self.burn_in}, thin={self.thin})")

        # ── Posterior mean estimates ──────────────────────────────────────────
        gamma_mean = np.mean(np.vstack(kept_gamma), axis=0)
        x_mean     = np.mean(np.vstack(kept_x),     axis=0)
        self.phi      = float(np.mean(kept_phi))
        self.sigma2_x = float(np.mean(kept_sigma2))
        self.m0, self.beta_p, self.beta_v, self.beta_so = self._gamma_to_components(gamma_mean)
        self._gamma_mean = gamma_mean

        # Store end-of-sample state for OOS prediction initialisation
        self._last_state_mean = float(x_mean[-1])
        self._last_state_var  = self.sigma2_x / max(1.0 - self.phi ** 2, 1e-4)

        self._trained = True

        # ── Variance decomposition diagnostics ───────────────────────────────
        panel_fit  = self.predict(panel, x0=0.0)
        total_var  = float(np.var(y))
        tod_p_var  = float(np.var(panel_fit["tod_p"].to_numpy()))
        tod_v_var  = float(np.var(panel_fit["tod_v"].to_numpy()))
        logger.info(f"  [model] Static  ToD (β^p)  : {tod_p_var / total_var:.1%} of log(y²) variance")
        logger.info(f"  [model] Volume  ToD (β^v)  : {tod_v_var / total_var:.1%} of log(y²) variance")
        logger.info(f"  [model] Combined R²         : {(tod_p_var + tod_v_var) / total_var:.1%}")

        return self

    # ──────────────────────────────────────────────────────────────────────────
    # Predict
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, panel: pd.DataFrame, x0: float | None = None) -> pd.DataFrame:
        """
        Compute one-step-ahead log-variance forecast ĥ_{t|t-1} for each bar.

        The forecast uses a real-time Kalman filter for the latent state x_t:
            ĥ_{t|t-1} = deterministic_t + φ x̂_{t-1|t-1}
        with an online filtered state update using the unconditional mixture
        moments (no mixture resampling needed OOS).

        Also applies stored training-sample volume normalisation to panel,
        so vol_dev is look-ahead-free regardless of how the caller prepared the data.

        Adds columns: tod_p (= I'_t β^p), tod_v (= v_{t-1} I'_t β^v),
                      so_t, x_t (filtered), h_pred (one-step-ahead forecast).
        """
        if not self._trained:
            raise RuntimeError("Model must be fit before predict().")

        # FIX: always recompute vol_dev using training-sample stats (look-ahead-free)
        panel = self._apply_vol_normalization(panel)
        panel = panel.sort_values(["date", "bar_idx"]).reset_index(drop=True)

        X, parts        = self._design_matrix(panel)
        deterministic   = X @ self._gamma_mean   # m₀ + tod^p + tod^v + so (all at once)

        # ── Decompose for diagnostics ─────────────────────────────────────────
        # tod_p: pure periodic component I'_t β^p  (NO m₀ — m₀ is separate)
        tod_p = np.zeros(len(panel), dtype=float)
        iv_arr = parts["interval_values"]
        for iv in self._intervals:
            tod_p[iv_arr == iv] = self.beta_p.get(iv, 0.0)

        # tod_v: volume-driven component v_{t-1} I'_t β^v
        tod_v = np.zeros(len(panel), dtype=float)
        for i, iv in enumerate(self._intervals):
            tod_v += parts["volume"][:, i] * self.beta_v.get(iv, 0.0)

        # so_t: Sunday-open spillover H'_t β^{so}
        so_t = (parts["sunday"] @ self.beta_so
                if self.beta_so.size else np.zeros(len(panel)))

        # ── Online Kalman filter for x_t (OOS) ───────────────────────────────
        # Measurement (using unconditional mixture moments as approximation):
        #   y_t - deterministic_t - LOG_CHI2_MEAN = x_t + ε_t,
        #   ε_t ~ N(0, LOG_CHI2_VAR)
        # State:
        #   x_t = φ x_{t-1} + η_t,  η_t ~ N(0, σ²_x)
        h_pred  = np.zeros(len(panel), dtype=float)
        x_filt  = np.zeros(len(panel), dtype=float)
        x_pred  = np.zeros(len(panel), dtype=float)

        state_mean = self._last_state_mean if x0 is None else float(x0)
        state_var  = self._last_state_var

        y = panel["log_r2"].to_numpy(dtype=float)

        for t in range(len(panel)):
            # One-step-ahead prediction
            a = self.phi * state_mean
            r = self.phi ** 2 * state_var + self.sigma2_x
            x_pred[t]  = a
            h_pred[t]  = deterministic[t] + a   # ĥ_{t|t-1} = det. + E[x_t | I_{t-1}]

            # Filtered state update using actual y_t
            innov      = y[t] - deterministic[t] - LOG_CHI2_MEAN - a
            q          = r + LOG_CHI2_VAR
            k          = r / max(q, 1e-8)
            state_mean = a + k * innov
            state_var  = max((1.0 - k) * r, 1e-8)
            x_filt[t]  = state_mean

        panel["tod_p"]  = tod_p        # pure I'_t β^p  (not including m₀)
        panel["tod_v"]  = tod_v        # v_{t-1} I'_t β^v
        panel["so_t"]   = so_t         # H'_t β^{so}
        panel["x_t"]    = x_filt       # filtered x_t (for state handover)
        panel["h_pred"] = h_pred       # one-step-ahead log-variance forecast
        return panel


# ── OLS + EWMA baseline ───────────────────────────────────────────────────────

class VolumeToD_OlsModel(_BaseVolumeModel):
    """
    Fast OLS + EWMA approximation (diagnostic baseline only; not the paper's model).

    Departs from the paper in three ways:
    1. OLS instead of Bayesian MCMC → no uncertainty quantification
    2. No zero-sum restriction on β^p (uses first-interval baseline instead)
    3. EWMA residual smoothing instead of Carter-Kohn FFBS for x_t
    """

    def __init__(self, ewma_lambda: float = 0.94, log_var_clip: float = -20.0):
        super().__init__(log_var_clip=log_var_clip)
        self.ewma_lambda = ewma_lambda
        self.beta_p: dict[int, float] = {}
        self.beta_v: dict[int, float] = {}
        self.intercept: float = 0.0
        self._vol_stats_: pd.DataFrame | None = None

    def _apply_vol_normalization(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Apply stored training-sample vol stats (same fix as VolumeToD_Model)."""
        if self._vol_stats_ is None:
            raise RuntimeError("Model not fit; call fit() before predict()")
        panel = panel.copy()
        mu  = panel["interval"].map(self._vol_stats_["vol_mean"])
        sig = panel["interval"].map(self._vol_stats_["vol_std"]).clip(lower=1e-8)
        panel["vol_dev"] = (panel["log_vol"] - mu) / sig
        return panel

    def fit(self, panel: pd.DataFrame) -> "VolumeToD_OlsModel":
        panel = (panel.copy()
                      .sort_values(["date", "bar_idx"])
                      .reset_index(drop=True))
        self._intervals = sorted(panel["interval"].unique())
        if len(self._intervals) < 2:
            raise ValueError("Need ≥2 intraday intervals.")

        # Store training vol stats and recompute vol_dev
        self._vol_stats_ = (
            panel
            .groupby("interval")["log_vol"]
            .agg(vol_mean="mean", vol_std="std")
        )
        self._vol_stats_["vol_std"] = self._vol_stats_["vol_std"].clip(lower=1e-8)
        panel = self._apply_vol_normalization(panel)

        interval_values = panel["interval"].to_numpy(dtype=int)

        # Step 1: OLS of log_r2 on interval dummies (baseline interval = first)
        X1 = np.column_stack([
            np.ones(len(panel)),
            *[(interval_values == iv).astype(float) for iv in self._intervals[1:]]
        ])
        y1 = panel["log_r2"].to_numpy(dtype=float)
        b1, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
        self.intercept = float(b1[0])
        self.beta_p    = {self._intervals[0]: 0.0}
        for i, iv in enumerate(self._intervals[1:]):
            self.beta_p[iv] = float(b1[1 + i])

        tod_p_hat = X1 @ b1
        resid1    = y1 - tod_p_hat

        # Step 2: OLS of residuals on v_{t-1} × interval dummies
        vol_lag = _vol_dev_lag(panel)   # uses global shift (same fix as Bayesian model)
        X2 = np.column_stack([
            vol_lag * (interval_values == iv).astype(float)
            for iv in self._intervals
        ])
        b2, _, _, _ = np.linalg.lstsq(X2, resid1, rcond=None)
        self.beta_v = {iv: float(b2[i]) for i, iv in enumerate(self._intervals)}

        # Step 3: EWMA of remaining residuals as proxy for x_t
        tod_v_hat = X2 @ b2
        resid2    = resid1 - tod_v_hat
        x         = np.zeros(len(panel))
        lam       = self.ewma_lambda
        x[0]      = resid2[0]
        for i in range(1, len(panel)):
            x[i] = lam * x[i - 1] + (1 - lam) * resid2[i]

        self._last_state_mean = float(x[-1])
        self._trained = True

        total_var = float(np.var(y1))
        logger.info(f"  [OLS] Static ToD: {np.var(tod_p_hat) / total_var:.1%}, "
                    f"Volume ToD: {np.var(tod_v_hat) / total_var:.1%}")
        return self

    def predict(self, panel: pd.DataFrame, x0: float = 0.0) -> pd.DataFrame:
        if not self._trained:
            raise RuntimeError("Model must be fit before predict().")

        # Recompute vol_dev using training stats (look-ahead-free)
        panel = self._apply_vol_normalization(panel)
        panel = panel.sort_values(["date", "bar_idx"]).reset_index(drop=True)

        interval_values = panel["interval"].to_numpy(dtype=int)

        tod_p = np.full(len(panel), self.intercept, dtype=float)
        for iv in self._intervals[1:]:
            tod_p[interval_values == iv] += self.beta_p.get(iv, 0.0)

        vol_lag = _vol_dev_lag(panel)
        tod_v   = np.zeros(len(panel), dtype=float)
        for iv in self._intervals:
            mask    = interval_values == iv
            tod_v[mask] = vol_lag[mask] * self.beta_v.get(iv, 0.0)

        log_r2 = panel["log_r2"].to_numpy(dtype=float)
        x      = np.zeros(len(panel), dtype=float)
        x[0]   = x0
        h_prev = tod_p[0] + tod_v[0] + x[0]
        lam    = self.ewma_lambda
        for i in range(1, len(panel)):
            prev_resid = log_r2[i - 1] - h_prev
            x[i]       = lam * x[i - 1] + (1 - lam) * prev_resid
            h_prev     = tod_p[i] + tod_v[i] + x[i]

        panel["tod_p"]  = tod_p - self.intercept   # pure I'_t β^p, consistent with Bayesian model
        panel["tod_v"]  = tod_v
        panel["x_t"]    = x
        panel["h_pred"] = tod_p + tod_v + x
        return panel
